# app.py — CrowdGuard: Intelligent Multi-CCTV Safe Routing System

import streamlit as st
import cv2
import tempfile
import time
import threading
import numpy as np
import pandas as pd
import heapq
from collections import defaultdict
from datetime import datetime

try:
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

from core.crowd_processor import process_video_frame, reset_processor

st.set_page_config(
    page_title="CrowdGuard — Safe Routing",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.stApp { background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%); color: #1b5e20; }
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #a5d6a7, #81c784);
    border-right: 1px solid #4caf50;
}
section[data-testid="stSidebar"] * { color: #111 !important; }
.badge-safe     { background:#1b5e20; color:#a5d6a7; padding:4px 12px; border-radius:20px; font-weight:700; }
.badge-moderate { background:#e65100; color:#fff59d; padding:4px 12px; border-radius:20px; font-weight:700; }
.badge-risk     { background:#b71c1c; color:#ffccbc; padding:4px 12px; border-radius:20px; font-weight:700; }
.badge-warmup   { background:#5d4037; color:#ffe0b2; padding:4px 12px; border-radius:20px; font-weight:700; }
.route-safe     { background:linear-gradient(135deg,#66bb6a,#43a047); border:1px solid #2e7d32; border-radius:12px; padding:14px 18px; margin:5px 0; color:#fff; }
.route-moderate { background:linear-gradient(135deg,#ffa726,#e65100); border:1px solid #bf360c; border-radius:12px; padding:14px 18px; margin:5px 0; color:#fff; }
.route-risk     { background:linear-gradient(135deg,#ef5350,#c62828); border:1px solid #b71c1c; border-radius:12px; padding:14px 18px; margin:5px 0; color:#fff; }
.route-exit-direct { background:linear-gradient(135deg,#e65100,#bf360c); border:2px solid #ff6d00; border-radius:12px; padding:14px 18px; margin:5px 0; color:#fff; }
.route-all-risk { background:linear-gradient(135deg,#b71c1c,#880e4f); border:2px solid #f44336; border-radius:12px; padding:16px 20px; margin:5px 0; color:#fff; }
.metric-card { background:#a5d6a7; border:1px solid #66bb6a; border-radius:10px; padding:14px; text-align:center; }
.metric-value { font-size:2rem; font-weight:800; color:#1b5e20; }
.metric-label { font-size:0.78rem; color:#2e7d32; text-transform:uppercase; }
.warmup-banner { background:#fff3e0; border:1px solid #ff9800; border-radius:8px; padding:10px 16px; color:#e65100; font-weight:600; margin-bottom:10px; }
.voice-banner  { background:#e3f2fd; border:1px solid #1976d2; border-radius:8px; padding:10px 16px; color:#0d47a1; font-weight:600; margin-bottom:10px; }
h2, h3 { color:#2e7d32 !important; }
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────
def _init():
    defaults = dict(
        step=1, zone_names=[], graph=defaultdict(list),
        exit_zones=[], running=False, routing_log=[],
        voice_enabled=True, temp_paths=[],
    )
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v
_init()
S = st.session_state

# ── Voice alerts ──────────────────────────────────────────────
_last_spoken:    dict  = {}
_VOICE_COOLDOWN: float = 15.0

def _speak(text: str):
    def _run():
        try:
            import pyttsx3
            engine = pyttsx3.init()
            engine.setProperty("rate", 155)
            engine.setProperty("volume", 1.0)
            engine.say(text)
            engine.runAndWait()
        except Exception:
            pass
    threading.Thread(target=_run, daemon=True).start()

def _maybe_speak(zone_names, zone_statuses, route_decisions, voice_enabled):
    if not voice_enabled:
        return None
    now = time.time()
    real = [s for s in zone_statuses if s != "WARMUP"]
    all_risk = bool(real) and all(s == "RISK" for s in real)

    if all_risk:
        if now - _last_spoken.get("ALL", 0) > _VOICE_COOLDOWN:
            _last_spoken["ALL"] = now
            _speak("Emergency. All zones critical. Full evacuation now.")
            return "🔊 ALL ZONES CRITICAL — evacuation spoken."
        return None
    _last_spoken.pop("ALL", None)

    announced = []
    for i, status in enumerate(zone_statuses):
        if status != "RISK" or now - _last_spoken.get(i, 0) < _VOICE_COOLDOWN:
            continue
        _last_spoken[i] = now
        rd   = route_decisions.get(i, {})
        path = rd.get("path", [i])
        sev  = rd.get("severity", "risk")
        if sev == "exit_direct":
            _speak(f"{zone_names[i]} is the exit zone at high risk. Evacuate directly now.")
            announced.append(f"🔊 {zone_names[i]}: IS exit — evacuate directly")
        elif len(path) > 1:
            route_str = " to ".join(zone_names[p] for p in path)
            _speak(f"{zone_names[i]} high risk. Move via {route_str}.")
            announced.append(f"🔊 {zone_names[i]}: via {route_str}")
        else:
            _speak(f"{zone_names[i]} high risk. No exit path. Secure zone.")
            announced.append(f"🔊 {zone_names[i]}: no exit — secure zone")
    return "  |  ".join(announced) if announced else None

# ── A* routing ────────────────────────────────────────────────
def _astar(graph, risks, start, exits):
    if start in exits:
        return [start]
    pq      = [(0.0, start, [start])]
    visited = {}
    while pq:
        g, node, path = heapq.heappop(pq)
        if node in visited and visited[node] <= g:
            continue
        visited[node] = g
        if node in exits:
            return path
        for nb in graph.get(node, []):
            g2 = g + (risks[node] + risks[nb]) / 2.0
            if nb not in visited or visited[nb] > g2:
                heapq.heappush(pq, (g2, nb, path + [nb]))
    return None

def compute_routes(graph, risks, statuses, exit_zones, num_zones, zone_names):
    exits    = exit_zones if exit_zones else [int(np.argmin(risks))]
    real     = [s for s in statuses if s != "WARMUP"]
    all_risk = bool(real) and all(s == "RISK" for s in real)
    results  = {}

    for src in range(num_zones):
        s = statuses[src]
        if s == "WARMUP":
            results[src] = dict(path=[src], severity="safe", message="⏳ Calibrating…")
            continue
        if s == "SAFE":
            msg = "✅ SAFE — exit directly." if src in exits else "✅ SAFE — no movement needed."
            results[src] = dict(path=[src], severity="safe", message=msg)
            continue
        if src in exits:
            if all_risk:
                results[src] = dict(path=[src], severity="all_risk",
                    message=f"🚨 ALL CRITICAL — {zone_names[src]} IS EXIT. Evacuate through here NOW.")
            elif s == "RISK":
                results[src] = dict(path=[src], severity="exit_direct",
                    message=f"🔴 HIGH RISK — {zone_names[src]} is EXIT ZONE. Clear and evacuate directly.")
            else:
                results[src] = dict(path=[src], severity="moderate",
                    message=f"⚠️ {zone_names[src]} is EXIT with moderate density. Direct people out.")
            continue

        path = _astar(graph, risks, src, exits)
        path_str = " → ".join(zone_names[p] for p in path) if path else "none"

        if path is None:
            sev = "all_risk" if all_risk else "risk"
            results[src] = dict(path=[src], severity=sev,
                message="🚨 ALL CRITICAL — no exit path. Alert authorities NOW." if all_risk
                        else "🔴 HIGH RISK — no exit path. Secure zone.")
        elif all_risk:
            results[src] = dict(path=path, severity="all_risk",
                message=f"🚨 ALL CRITICAL — evacuate via: {path_str}")
        elif s == "RISK":
            results[src] = dict(path=path, severity="risk",
                message=f"🔴 HIGH RISK — move people urgently via: {path_str}")
        else:
            results[src] = dict(path=path, severity="moderate",
                message=f"⚠️ Moderate — suggested route: {path_str}")
    return results

# ── Zone graph (Plotly) ───────────────────────────────────────
STATUS_HEX  = {"SAFE":"#4caf50","MODERATE":"#ffa726","RISK":"#ef5350","WARMUP":"#90a4ae"}
ROUTE_COLS  = ["#1565c0","#6a1b9a","#004d40","#bf360c","#37474f"]
SEVERITY_BOX = {
    "safe":"route-safe","moderate":"route-moderate","risk":"route-risk",
    "exit_direct":"route-exit-direct","all_risk":"route-all-risk",
}
SEVERITY_STATUS = {
    "safe":"SAFE","moderate":"MODERATE","risk":"RISK",
    "exit_direct":"RISK","all_risk":"RISK",
}

def draw_graph(zone_names, graph, risks, statuses, route_decisions, exit_zones):
    if not HAS_PLOTLY or not zone_names:
        return None
    n      = len(zone_names)
    angles = [2 * np.pi * i / n for i in range(n)]
    pos    = {i: (np.cos(a), np.sin(a)) for i, a in enumerate(angles)}
    fig    = go.Figure()
    drawn  = set()
    for src, dsts in graph.items():
        for dst in dsts:
            key = tuple(sorted((src, dst)))
            if key in drawn: continue
            drawn.add(key)
            x0,y0=pos[src]; x1,y1=pos[dst]
            fig.add_trace(go.Scatter(x=[x0,x1,None],y=[y0,y1,None],
                mode="lines",line=dict(color="#888",width=1.5),hoverinfo="none",showlegend=False))
    for zi, rd in route_decisions.items():
        path = rd.get("path", [])
        if len(path) > 1:
            col = ROUTE_COLS[zi % len(ROUTE_COLS)]
            for a, b in zip(path, path[1:]):
                x0,y0=pos[a]; x1,y1=pos[b]
                fig.add_trace(go.Scatter(x=[x0,x1,None],y=[y0,y1,None],
                    mode="lines",line=dict(color=col,width=3.5,dash="dot"),
                    hoverinfo="none",showlegend=False))
    nx=[pos[i][0] for i in range(n)]; ny=[pos[i][1] for i in range(n)]
    colors  = [STATUS_HEX.get(statuses.get(i,"WARMUP"),"#90a4ae") for i in range(n)]
    borders = ["#ffeb3b" if i in exit_zones else "#1b5e20" for i in range(n)]
    sizes   = [38 if i in exit_zones else 28 for i in range(n)]
    symbols = ["diamond" if i in exit_zones else "circle" for i in range(n)]
    hover   = [f"<b>{zone_names[i]}</b><br>Status: {statuses.get(i,'?')}<br>Risk: {risks[i]:.2f}"
               + (" 🚪 EXIT" if i in exit_zones else "") for i in range(n)]
    fig.add_trace(go.Scatter(x=nx,y=ny,mode="markers+text",
        marker=dict(size=sizes,color=colors,line=dict(width=3,color=borders),symbol=symbols),
        text=[zone_names[i] for i in range(n)],textposition="top center",
        hovertext=hover,hoverinfo="text",name="Zones"))
    fig.update_layout(paper_bgcolor="#e8f5e9",plot_bgcolor="#e8f5e9",
        showlegend=False,margin=dict(l=10,r=10,t=10,b=10),height=340,
        xaxis=dict(showgrid=False,zeroline=False,showticklabels=False),
        yaxis=dict(showgrid=False,zeroline=False,showticklabels=False))
    return fig

# ── Sidebar wizard ────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/48/shield.png", width=42)
    st.title("CrowdGuard")
    st.caption("Intelligent Multi-CCTV Safe Routing")
    st.divider()

    st.markdown("#### Step 1 · Upload CCTV Footage")
    uploaded = st.file_uploader(
        "Select video files (mp4 / avi / mov)",
        type=["mp4","avi","mov"], accept_multiple_files=True, key="uploader"
    )
    if uploaded and S.step == 1:
        if st.button("▶ Next: Name Zones", use_container_width=True):
            S.zone_names = [f"Zone {i+1}" for i in range(len(uploaded))]
            S.exit_zones = []; S.step = 2; st.rerun()

    if S.step >= 2 and uploaded:
        st.divider()
        st.markdown("#### Step 2 · Name Zones")
        new_names = []
        for i in range(len(uploaded)):
            nm = st.text_input(f"Zone {i+1}",
                value=S.zone_names[i] if i < len(S.zone_names) else f"Zone {i+1}",
                key=f"zname_{i}")
            new_names.append(nm)
        S.zone_names = new_names
        c1,c2 = st.columns(2)
        with c1:
            if st.button("◀ Back", use_container_width=True, key="b2b"):
                S.step=1; st.rerun()
        with c2:
            if st.button("▶ Connections", use_container_width=True, key="b2n"):
                S.step=3; st.rerun()

    if S.step >= 3 and uploaded:
        st.divider()
        st.markdown("#### Step 3 · Zone Connectivity")
        new_graph = defaultdict(list)
        for i, name in enumerate(S.zone_names):
            others = [z for z in S.zone_names if z != name]
            prev   = [S.zone_names[j] for j in S.graph.get(i, []) if j < len(S.zone_names)]
            conns  = st.multiselect(f"{name} → connects to:", options=others,
                                    default=prev, key=f"conn_{i}")
            for c in conns:
                j = S.zone_names.index(c)
                if j not in new_graph[i]: new_graph[i].append(j)
                if i not in new_graph[j]: new_graph[j].append(i)
        S.graph = new_graph

        st.divider()
        st.markdown("#### Step 4 · Exit Points")
        exit_sel = st.multiselect("Select exit zones:", options=S.zone_names,
            default=[S.zone_names[i] for i in S.exit_zones if i < len(S.zone_names)],
            key="exit_sel")
        S.exit_zones = [S.zone_names.index(e) for e in exit_sel]

        st.divider()
        st.markdown("#### Step 5 · Voice Alerts")
        S.voice_enabled = st.toggle("🔊 Enable voice alerts", value=S.get("voice_enabled", True))

        c1,c2 = st.columns(2)
        with c1:
            if st.button("◀ Back", use_container_width=True, key="b3b"):
                S.step=2; st.rerun()
        with c2:
            if st.button("🚀 Launch", use_container_width=True, type="primary", key="b3n"):
                S.step=4; S.running=True; S.routing_log=[]
                reset_processor()
                st.rerun()

    if S.step == 4:
        st.divider()
        S.voice_enabled = st.toggle("🔊 Voice alerts", value=S.get("voice_enabled", True), key="voice_live")
        if st.button("⏹ Stop System", use_container_width=True, type="secondary"):
            S.running=False; S.step=3; st.rerun()

# ── Main panel ────────────────────────────────────────────────
st.markdown("## 🛡️ CrowdGuard — Intelligent Safe Routing")
st.divider()

if not uploaded or S.step < 4 or not S.running:
    st.info("👈 Use the sidebar wizard to upload videos, configure zones and exits, then launch.")
    st.stop()

# ── Open video captures ───────────────────────────────────────
@st.cache_resource
def _open_caps(paths_tuple):
    return [cv2.VideoCapture(p) for p in paths_tuple]

if not S.temp_paths or len(S.temp_paths) != len(uploaded):
    temp_paths = []
    for vid in uploaded:
        tf = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tf.write(vid.read()); tf.flush()
        temp_paths.append(tf.name)
    S.temp_paths = temp_paths

caps      = _open_caps(tuple(S.temp_paths))
num_zones = len(caps)

vid_fps = []
for cap in caps:
    fps = cap.get(cv2.CAP_PROP_FPS)
    vid_fps.append(fps if fps and fps > 0 else 30.0)

# Target UI refresh: 10 fps regardless of source FPS
TARGET_UI_FPS  = 10
FRAME_WAIT     = 1.0 / TARGET_UI_FPS

# ── UI layout ─────────────────────────────────────────────────
MAX_COLS    = min(num_zones, 3)
rows_needed = (num_zones + MAX_COLS - 1) // MAX_COLS
vid_slots   = []
status_tags = []

for r in range(rows_needed):
    row_cols = st.columns(MAX_COLS)
    for c in range(MAX_COLS):
        idx = r * MAX_COLS + c
        if idx < num_zones:
            with row_cols[c]:
                exit_mark = " 🚪" if idx in S.exit_zones else ""
                st.markdown(f"**{S.zone_names[idx]}{exit_mark}**")
                vid_slots.append(st.empty())
                status_tags.append(st.empty())

st.markdown("---")
st.markdown("### 🔀 Zone-by-Zone Routing")
warmup_banner  = st.empty()
voice_banner   = st.empty()
routing_panels = [st.empty() for _ in range(num_zones)]
st.markdown("---")

r_col, g_col = st.columns([1, 1])
with r_col:
    st.markdown("### 📊 System Summary")
    summary_box = st.empty()

st.markdown("---")
metric_cols = st.columns(4)
m_slots     = [c.empty() for c in metric_cols]
log_slot    = st.empty()

# ── Main loop ─────────────────────────────────────────────────
statuses_map: dict  = {}
risks_arr           = np.zeros(num_zones)
last_frame_time     = [time.time() - 1.0] * num_zones  # force immediate first read

while S.running:
    loop_start = time.time()

    zone_density  = []
    zone_vul      = []
    zone_statuses = []
    zone_counts   = []
    zone_frames   = []

    for i, cap in enumerate(caps):
        # Skip ahead if we've fallen behind real time
        now          = time.time()
        elapsed_zone = now - last_frame_time[i]
        frames_due   = max(1, int(elapsed_zone * vid_fps[i]))

        if frames_due > 1:
            cur_pos   = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            total_frm = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            new_pos   = cur_pos + frames_due - 1
            if total_frm > 0:
                new_pos = new_pos % total_frm
            cap.set(cv2.CAP_PROP_POS_FRAMES, new_pos)

        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()

        last_frame_time[i] = time.time()

        if frame is None:
            zone_density.append(0.0)
            zone_vul.append(0.0)
            zone_statuses.append(statuses_map.get(i, "WARMUP"))
            zone_counts.append({"Child":0,"Adult":0,"Elderly":0})
            zone_frames.append(None)
            continue

        proc, density_pct, vul, status, counts = process_video_frame(
            frame, zone_id=i, video_fps=vid_fps[i]
        )
        statuses_map[i] = status

        zone_density.append(density_pct)
        zone_vul.append(vul)
        zone_statuses.append(status)
        zone_counts.append(counts)
        zone_frames.append(cv2.cvtColor(proc, cv2.COLOR_BGR2RGB))

    # Risk score: 70% density, 30% vulnerable
    d_arr     = np.array(zone_density, dtype=float) / 100.0
    v_arr     = np.array(zone_vul,     dtype=float)
    v_norm    = v_arr / (v_arr.max() + 1e-6)
    risks_arr = 0.70 * d_arr + 0.30 * v_norm

    route_decisions = compute_routes(
        graph=dict(S.graph), risks=risks_arr, statuses=zone_statuses,
        exit_zones=S.exit_zones, num_zones=num_zones, zone_names=S.zone_names,
    )

    voice_msg = _maybe_speak(S.zone_names, zone_statuses, route_decisions, S.get("voice_enabled", True))
    ts        = datetime.now().strftime("%H:%M:%S")
    in_warmup = any(s == "WARMUP" for s in zone_statuses)

    # ── Video + status badges ─────────────────────────────────
    for i in range(num_zones):
        if zone_frames[i] is not None:
            vid_slots[i].image(zone_frames[i], channels="RGB", use_container_width=True)

        status      = zone_statuses[i]
        density_pct = zone_density[i]
        counts      = zone_counts[i]
        badge_cls   = {"SAFE":"badge-safe","MODERATE":"badge-moderate",
                       "RISK":"badge-risk","WARMUP":"badge-warmup"}.get(status, "badge-safe")
        exit_tag    = "&nbsp;🚪 EXIT" if i in S.exit_zones else ""
        bar_color   = "#ef5350" if density_pct >= 70 else "#ffa726" if density_pct >= 40 else "#4caf50"
        density_bar = (
            f"<span style='display:inline-block;width:{min(density_pct,100):.0f}%;max-width:120px;"
            f"height:8px;background:{bar_color};border-radius:4px;vertical-align:middle'></span>"
            f"&nbsp;<b>{density_pct:.0f}%</b>"
        )
        status_tags[i].markdown(
            f"<span class='{badge_cls}'>{status}</span>{exit_tag}"
            f"&nbsp;&nbsp;👶 {counts['Child']}  👴 {counts['Elderly']}  🧍 {counts['Adult']}"
            f"&nbsp;|&nbsp;Density: {density_bar}",
            unsafe_allow_html=True
        )

    # ── Banners ───────────────────────────────────────────────
    if in_warmup:
        warmup_banner.markdown(
            "<div class='warmup-banner'>⏳ <b>WARMUP</b> — Calibrating (5 frames per zone).</div>",
            unsafe_allow_html=True)
    else:
        warmup_banner.empty()

    if voice_msg:
        voice_banner.markdown(f"<div class='voice-banner'>{voice_msg}</div>", unsafe_allow_html=True)
    else:
        voice_banner.empty()

    # ── Routing panels ────────────────────────────────────────
    for i in range(num_zones):
        rd      = route_decisions.get(i, {"message":"…","severity":"safe"})
        box_cls = SEVERITY_BOX.get(rd["severity"], "route-safe")
        routing_panels[i].markdown(
            f"<div class='{box_cls}'><b>{S.zone_names[i]}</b> — {rd['message']}</div>",
            unsafe_allow_html=True)

    # ── Summary ───────────────────────────────────────────────
    safe_z  = [S.zone_names[i] for i,s in enumerate(zone_statuses) if s=="SAFE"]
    mod_z   = [S.zone_names[i] for i,s in enumerate(zone_statuses) if s=="MODERATE"]
    risk_z  = [S.zone_names[i] for i,s in enumerate(zone_statuses) if s=="RISK"]
    warm_z  = [S.zone_names[i] for i,s in enumerate(zone_statuses) if s=="WARMUP"]
    exit_nm = [S.zone_names[i] for i in S.exit_zones if i < len(S.zone_names)]

    lines = []
    if warm_z:  lines.append(f"⏳ <b>WARMUP:</b> {', '.join(warm_z)}")
    if safe_z:  lines.append(f"🟢 <b>SAFE:</b> {', '.join(safe_z)}")
    if mod_z:   lines.append(f"🟡 <b>MODERATE:</b> {', '.join(mod_z)}")
    if risk_z:  lines.append(f"🔴 <b>RISK:</b> {', '.join(risk_z)}")
    if exit_nm: lines.append(f"🚪 <b>EXIT:</b> {', '.join(exit_nm)}")
    lines.append(f"🔊 <b>Voice:</b> {'ON' if S.get('voice_enabled') else 'OFF'}")
    lines.append(f"🕐 <b>Updated:</b> {ts}")

    all_risk_now = bool(risk_z) and len(risk_z) == num_zones and not in_warmup
    if all_risk_now:
        lines.insert(0, "🚨 <b>CRITICAL — ALL ZONES AT RISK. Emergency evacuation.</b>")
    sum_cls = "route-all-risk" if all_risk_now else "route-safe"
    summary_box.markdown(f"<div class='{sum_cls}'>" + "<br>".join(lines) + "</div>",
                         unsafe_allow_html=True)

    # ── Zone graph ────────────────────────────────────────────
    fig = draw_graph(S.zone_names, S.graph, risks_arr,
                     {i:s for i,s in enumerate(zone_statuses)},
                     route_decisions, S.exit_zones)
    if fig:
        g_col.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})

    # ── Metrics ───────────────────────────────────────────────
    safest  = int(np.argmin(risks_arr))
    riskest = int(np.argmax(risks_arr))
    vul_tot = sum(zone_vul)

    m_slots[0].markdown(f"<div class='metric-card'><div class='metric-value'>{vul_tot:.1f}</div><div class='metric-label'>Vulnerable Score</div></div>", unsafe_allow_html=True)
    m_slots[1].markdown(f"<div class='metric-card'><div class='metric-value' style='color:#2e7d32'>{S.zone_names[safest]}</div><div class='metric-label'>Safest Zone</div></div>", unsafe_allow_html=True)
    m_slots[2].markdown(f"<div class='metric-card'><div class='metric-value' style='color:#b71c1c'>{S.zone_names[riskest]}</div><div class='metric-label'>Highest Risk Zone</div></div>", unsafe_allow_html=True)
    m_slots[3].markdown(f"<div class='metric-card'><div class='metric-value'>{len(S.exit_zones)}</div><div class='metric-label'>Exit Zones</div></div>", unsafe_allow_html=True)

    # ── Log ───────────────────────────────────────────────────
    vul_per = vul_tot / max(num_zones, 1)
    vul_lbl = ("None" if vul_per==0 else "Low" if vul_per<2 else "Moderate" if vul_per<5 else "High")
    log_entry = {"Time": ts}
    for i in range(num_zones):
        rd = route_decisions.get(i, {})
        log_entry[S.zone_names[i]] = SEVERITY_STATUS.get(rd.get("severity","safe"), "SAFE")
    log_entry["Vulnerable"] = vul_lbl
    log_entry["Voice Alert"] = voice_msg.replace("🔊 ","") if voice_msg else "—"

    S.routing_log.append(log_entry)
    if S.routing_log:
        df_log = pd.DataFrame(S.routing_log[-20:][::-1])
        log_slot.dataframe(df_log, use_container_width=True, hide_index=True)

    # ── Frame pacing: target 10 UI fps ───────────────────────
    elapsed = time.time() - loop_start
    sleep_t = max(0.0, FRAME_WAIT - elapsed)
    time.sleep(sleep_t)

# ── Cleanup ───────────────────────────────────────────────────
for cap in caps:
    cap.release()