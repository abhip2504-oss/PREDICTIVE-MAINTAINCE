import streamlit as st
import pandas as pd
import joblib
import time
import os
import json
import plotly.graph_objects as go
from datetime import datetime
from collections import deque

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOCAL_DIR = os.path.join(BASE_DIR, "live_data")
MODEL_NAME = os.path.join(BASE_DIR, "model.pkl")

st.set_page_config(
    page_title="Predictive Maintenance",
    layout="wide",
    page_icon="⚙️",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp {
    background-color: #0d1117;
    color: #e2e8f0;
}

/* Hide Streamlit chrome, reduce padding */
#MainMenu, footer, header { visibility: hidden; }
.block-container {
    padding: 1rem 1.5rem 0.5rem 1.5rem !important;
    max-width: 100% !important;
}

/* ── Top bar ── */
.topbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding-bottom: 0.75rem;
    border-bottom: 1px solid #1e2535;
    margin-bottom: 0.75rem;
}
.topbar h1 {
    font-size: 1.1rem;
    font-weight: 600;
    color: #f1f5f9;
    margin: 0;
    letter-spacing: -0.3px;
}
.topbar p {
    font-size: 0.72rem;
    color: #475569;
    margin: 0.1rem 0 0 0;
}
.topbar-right {
    display: flex;
    align-items: center;
    gap: 12px;
    font-size: 0.72rem;
    color: #64748b;
}
.pill {
    display: flex; align-items: center; gap: 5px;
    background: #161b27;
    border: 1px solid #1e2535;
    border-radius: 20px;
    padding: 3px 10px;
    font-size: 0.7rem;
}
.dot { width: 7px; height: 7px; border-radius: 50%; display: inline-block; }
.dot-green { background: #22c55e; box-shadow: 0 0 5px #22c55e88; }
.dot-red   { background: #ef4444; box-shadow: 0 0 5px #ef444488; }

/* ── Status banner ── */
.status-ok {
    background: #052e16;
    border: 1px solid #16a34a;
    color: #4ade80;
    border-radius: 7px;
    padding: 0.45rem 1rem;
    font-size: 0.78rem;
    font-weight: 500;
    margin-bottom: 0.75rem;
}
.status-fail {
    background: #2d0a0a;
    border: 1px solid #dc2626;
    color: #f87171;
    border-radius: 7px;
    padding: 0.45rem 1rem;
    font-size: 0.78rem;
    font-weight: 500;
    margin-bottom: 0.75rem;
    animation: blink 1.6s ease-in-out infinite;
}
@keyframes blink { 0%,100%{opacity:1} 50%{opacity:0.55} }

/* ── KPI Cards ── */
.kpi-card {
    background: #161b27;
    border: 1px solid #1e2535;
    border-radius: 10px;
    padding: 0.75rem 1rem;
    position: relative;
    overflow: visible;
}
.kpi-accent {
    position: absolute;
    top: 0; left: 0;
    width: 3px; height: 100%;
    border-radius: 10px 0 0 10px;
}
.kpi-label {
    font-size: 0.65rem;
    color: #475569;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    font-weight: 500;
    margin-bottom: 0.25rem;
}
.kpi-value {
    font-size: 1.45rem;
    font-weight: 700;
    color: #f1f5f9;
    line-height: 1;
    margin-bottom: 0.25rem;
}
.kpi-unit  { font-size: 0.75rem; font-weight: 400; color: #64748b; }
.kpi-delta { font-size: 0.68rem; font-weight: 500; }
.kpi-delta.up      { color: #f87171; }
.kpi-delta.down    { color: #4ade80; }
.kpi-delta.neutral { color: #475569; }

/* ── Section label ── */
.sec-title {
    font-size: 0.65rem;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: #334155;
    font-weight: 600;
    margin: 0 0 0.5rem 0;
}

/* ── Panel box ── */
.panel {
    background: #161b27;
    border: 1px solid #1e2535;
    border-radius: 10px;
    padding: 0.75rem 1rem;
    height: 100%;
}

/* ── Log ── */
.log-item {
    display: flex; gap: 8px;
    padding: 0.35rem 0;
    border-bottom: 1px solid #1a2030;
    font-size: 0.7rem;
    line-height: 1.3;
}
.log-time { color: #334155; min-width: 50px; font-variant-numeric: tabular-nums; }
.log-fail { color: #f87171; }
.log-empty { color: #1e2535; font-size: 0.72rem; padding: 0.4rem 0; }

/* ── Gauge label ── */
.g-label {
    font-size: 0.62rem;
    color: #334155;
    text-align: center;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-top: -6px;
    margin-bottom: 4px;
}

/* ── Failure rate ── */
.rate-bar-wrap {
    background: #1e2535;
    border-radius: 4px;
    height: 5px;
    margin: 6px 0 4px;
}
.rate-bar-fill {
    height: 5px;
    border-radius: 4px;
    transition: width 0.4s;
}
.rate-row {
    display: flex;
    justify-content: space-between;
    font-size: 0.68rem;
    color: #334155;
}

/* Override Streamlit metric */
div[data-testid="metric-container"] {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}

/* Tooltip styling */
.tooltip { position: relative; display: inline-block; margin-left: 5px; cursor: help; color: #64748b; font-size: 0.75rem; vertical-align: middle; }
.tooltip .tooltiptext {
    visibility: hidden; width: 220px; background-color: #1e2535;
    color: #e2e8f0; text-align: left; border-radius: 6px; padding: 8px 10px;
    position: absolute; z-index: 999; top: 150%; left: 0%; opacity: 0;
    margin-left: -20px;
    transition: opacity 0.2s; font-size: 0.65rem; font-weight: 400;
    line-height: 1.4; text-transform: none; letter-spacing: normal;
    border: 1px solid #334155; box-shadow: 0 4px 12px rgba(0,0,0,0.5);
}
.tooltip .tooltiptext::after {
    content: ""; position: absolute; bottom: 100%; left: 24px; margin-bottom: -1px;
    border-width: 5px; border-style: solid; border-color: transparent transparent #334155 transparent;
}
.tooltip:hover .tooltiptext { visibility: visible; opacity: 1; }

/* Reduce plotly chart top margin injected by streamlit */
.element-container { margin-bottom: 0 !important; }
</style>
""", unsafe_allow_html=True)


# ── Load model ──────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_NAME):
        import train_model
        train_model.train_and_save_model()
        
    if os.path.exists(MODEL_NAME):
        return joblib.load(MODEL_NAME)
    return None

model = load_model()
if model is None:
    st.error(f"Model `{MODEL_NAME}` not found. Run `train_model.py` first.")
    st.stop()

# ── Session state ────────────────────────────────────────────────────────────
if 'history'            not in st.session_state: st.session_state.history            = deque(maxlen=50)
if 'alert_log'          not in st.session_state: st.session_state.alert_log          = []
if 'fail_count'         not in st.session_state: st.session_state.fail_count         = 0
if 'total_count'        not in st.session_state: st.session_state.total_count        = 0

# ── Constants ────────────────────────────────────────────────────────────────
ACCENT = {
    "Air temperature [K]":     "#38bdf8",
    "Process temperature [K]": "#818cf8",
    "Rotational speed [rpm]":  "#34d399",
    "Torque [Nm]":             "#fbbf24",
    "Tool wear [min]":         "#f87171",
}
LABELS = {
    "Air temperature [K]":     ("Air Temp",   "K"),
    "Process temperature [K]": ("Proc Temp",  "K"),
    "Rotational speed [rpm]":  ("Speed",      "RPM"),
    "Torque [Nm]":             ("Torque",     "Nm"),
    "Tool wear [min]":         ("Tool Wear",  "min"),
}
RANGES = {
    "Air temperature [K]":     (295, 305),
    "Process temperature [K]": (305, 315),
    "Rotational speed [rpm]":  (1100, 2900),
    "Torque [Nm]":             (3,   80),
    "Tool wear [min]":         (0,   250),
}
INVERSE = {   # True = higher is worse
    "Air temperature [K]":     True,
    "Process temperature [K]": True,
    "Rotational speed [rpm]":  False,
    "Torque [Nm]":             True,
    "Tool wear [min]":         True,
}
DESCRIPTIONS = {
    "Air temperature [K]":     "Ambient room temp. High temp limits machine cooling.",
    "Process temperature [K]": "Internal machine temp. Excess heat leads to degradation.",
    "Rotational speed [rpm]":  "Spindle speed. Drops indicate friction or mechanical load issues.",
    "Torque [Nm]":             "Rotational force applied. Spikes suggest machine is struggling.",
    "Tool wear [min]":         "Cumulative tool use. High wear causes poor quality and tool failure.",
}

# ── Top bar (static, no placeholder needed) ──────────────────────────────────
now_str = datetime.now().strftime("%d %b %Y  %H:%M")
st.markdown(f"""
<div class="topbar">
  <div>
    <h1>⚙️ &nbsp;Predictive Maintenance</h1>
    <p>Smart Manufacturing · Real-time Telemetry Monitor</p>
  </div>
  <div class="topbar-right">
    <span class="pill"><span class="dot dot-green"></span>ML Engine Online</span>
    <span class="pill"><span class="dot dot-green"></span>IoT Stream Active</span>
    <span style="color:#1e2535">|</span>
    <span>{now_str}</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Placeholders ─────────────────────────────────────────────────────────────
status_ph = st.empty()

# KPI row — 5 cards
kpi_cols  = st.columns(5, gap="small")
kpi_phs   = [c.empty() for c in kpi_cols]

st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

# Bottom row: 3 gauges | event log + failure rate
left_col, right_col = st.columns([3, 2], gap="medium")

with left_col:
    st.markdown("<p class='sec-title'>Live Sensor Gauges</p>", unsafe_allow_html=True)
    g1, g2, g3 = st.columns(3, gap="small")
    gauge_speed_ph  = g1.empty()
    gauge_label1_ph = g1.empty()
    gauge_torque_ph = g2.empty()
    gauge_label2_ph = g2.empty()
    gauge_wear_ph   = g3.empty()
    gauge_label3_ph = g3.empty()

with right_col:
    hc1, hc2 = st.columns([2, 1])
    with hc1:
        st.markdown("<p class='sec-title'>Event Log</p>", unsafe_allow_html=True)
    with hc2:
        # Give the Streamlit button a little negative top margin to align with the text
        st.markdown("<style>div[data-testid='stButton'] button { padding: 0.1rem 0.5rem; font-size: 0.70rem; }</style>", unsafe_allow_html=True)
        if st.button("Clear", key="clear_logs_btn", use_container_width=True):
            st.session_state.alert_log = []
            st.session_state.fail_count = 0
            st.rerun()

    log_ph = st.empty()
    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
    st.markdown("<p class='sec-title'>Session Failure Rate</p>", unsafe_allow_html=True)
    rate_ph = st.empty()


# ── Helpers ──────────────────────────────────────────────────────────────────
def get_failure_reason(data):
    reasons = []
    
    # Simple heuristics mimicking typical failure modes
    air = data.get('Air temperature [K]', 300)
    proc = data.get('Process temperature [K]', 310)
    speed = data.get('Rotational speed [rpm]', 1500)
    torque = data.get('Torque [Nm]', 40)
    wear = data.get('Tool wear [min]', 0)
    
    # Heat Dissipation Failure (HDF)
    if (proc - air) < 8.6 and speed < 1380:
        reasons.append("Heat Dissipation Failure (High Temp + Low Speed)")
        
    # Power Failure (PWF)
    power = speed * torque * 0.1047 # rad/s conversion
    if power < 3500 or power > 9000:
        reasons.append("Power Anomaly Detected (Abnormal Torque/Speed Ratio)")
        
    # Overstrain Failure (OSF)
    if wear * torque > 11000:
        reasons.append("Overstrain Failure (High Tool Wear × High Torque)")
        
    # Tool Wear Failure (TWF)
    if wear > 200:
        reasons.append("Tool Wear Failure (Exceeded Lifespan Limits)")

    if not reasons:
        # Fallback tracking highest outlier
        outlier, max_pct = None, 0
        for feat, (lo, hi) in RANGES.items():
            val = data.get(feat, lo)
            pct = (val - lo) / (hi - lo) if hi > lo else 0
            if pct > max_pct:
                max_pct = pct
                outlier = LABELS[feat][0]
        if outlier:
            reasons.append(f"Anomalous Sensor Reading in {outlier} subsystem")
        else:
            reasons.append("Complex Multivariate Anomaly Detected")
            
    return reasons[0]

def get_latest_data():
    if not os.path.exists(LOCAL_DIR):
        return None
    files = [os.path.join(LOCAL_DIR, f) for f in os.listdir(LOCAL_DIR) if f.endswith('.json')]
    if not files:
        return None
    latest = max(files, key=os.path.getctime)
    with open(latest, 'r') as f:
        return json.load(f)


def delta_class(val, inverse):
    if val == 0:  return "neutral"
    if inverse:   return "up" if val > 0 else "down"
    return "down" if val > 0 else "up"

def delta_sym(val):
    return "▲" if val > 0 else "▼" if val < 0 else "—"


def render_kpi(ph, key, data, last_data):
    label, unit = LABELS[key]
    val   = data.get(key, 0)
    delta = (val - last_data.get(key, 0)) if last_data else 0
    cls   = delta_class(delta, INVERSE[key])
    sym   = delta_sym(delta)
    fmt   = f"{val:.0f}" if unit == "RPM" or unit == "min" else f"{val:.1f}"
    desc = DESCRIPTIONS.get(key, "")
    ph.markdown(f"""
    <div class="kpi-card">
      <div class="kpi-accent" style="background:{ACCENT[key]};"></div>
      <div class="kpi-label">
        {label}
        <div class="tooltip">ⓘ<span class="tooltiptext">{desc}</span></div>
      </div>
      <div class="kpi-value">{fmt}<span class="kpi-unit"> {unit}</span></div>
      <div class="kpi-delta {cls}">{sym} {abs(delta):.1f} {unit}</div>
    </div>
    """, unsafe_allow_html=True)


def make_gauge(value, lo, hi, color):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        number=dict(font=dict(size=18, color="#f1f5f9", family="Inter"), valueformat=".1f"),
        gauge=dict(
            axis=dict(range=[lo, hi], tickfont=dict(size=8, color="#334155"), nticks=4),
            bar=dict(color=color, thickness=0.5),
            bgcolor="rgba(0,0,0,0)",
            borderwidth=0,
            steps=[
                dict(range=[lo, lo + (hi-lo)*0.6],  color="#1a2030"),
                dict(range=[lo + (hi-lo)*0.6,  lo + (hi-lo)*0.85], color="#222a3a"),
                dict(range=[lo + (hi-lo)*0.85, hi],  color="#2d1515"),
            ],
            threshold=dict(line=dict(color="#ef4444", width=2), thickness=0.8, value=lo + (hi-lo)*0.9)
        )
    ))
    fig.update_layout(
        height=160,
        margin=dict(l=10, r=10, t=10, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter", color="#475569"),
    )
    return fig


def render_log_html(alerts):
    if not alerts:
        return "<div class='log-empty'>No failure events recorded.</div>"
    rows = ""
    for a in alerts[:12]:
        # Format: "[HH:MM:SS] rest"
        if a.startswith("["):
            close = a.find("]")
            ts  = a[1:close] if close != -1 else ""
            msg = a[close+1:].strip() if close != -1 else a
        else:
            ts, msg = "", a
        rows += f"<div class='log-item'><span class='log-time'>{ts}</span><span class='log-fail'>{msg}</span></div>"
    return rows


# ── Main loop ─────────────────────────────────────────────────────────────────
last_timestamp = None
last_data      = None

while True:
    data = get_latest_data()

    if data and data.get('timestamp') != last_timestamp:
        last_timestamp = data.get('timestamp')
        st.session_state.total_count += 1
        st.session_state.history.append(data)

        # ── Prediction ──
        features_order = ['Air temperature [K]', 'Process temperature [K]',
                          'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']
        df_pred    = pd.DataFrame([data], columns=features_order)
        prediction = model.predict(df_pred)[0]

        if prediction == 1:
            st.session_state.fail_count += 1
            ts  = datetime.now().strftime("%H:%M:%S")
            reason = get_failure_reason(data)
            sensor_pfx = f"[{data.get('sensor_id', 'unknown')}]"
            msg = f"[{ts}] Failure {sensor_pfx} · {reason}"
            if not st.session_state.alert_log or st.session_state.alert_log[0] != msg:
                st.session_state.alert_log.insert(0, msg)

        # ── Status ──
        if prediction == 1:
            banner_reason = get_failure_reason(data)
            status_ph.markdown(
                f"<div class='status-fail'>⚠️ &nbsp;<strong>ALERT:</strong> {banner_reason} — Dispatch Maintenance Immediately</div>",
                unsafe_allow_html=True
            )
        else:
            status_ph.markdown(
                "<div class='status-ok'>✓ &nbsp;<strong>Healthy:</strong> All systems nominal — No anomalies detected</div>",
                unsafe_allow_html=True
            )

        # ── KPI Cards ──
        keys = list(LABELS.keys())
        for ph, key in zip(kpi_phs, keys):
            render_kpi(ph, key, data, last_data)

        # ── Gauges ──
        # Call plotly_chart DIRECTLY on each st.empty() placeholder.
        # This replaces the slot content in-place and bypasses Streamlit's
        # global element-ID registry entirely — no collision across iterations.
        def _gauge(feat, g_ph, lbl_ph):
            lo, hi = RANGES[feat]
            val    = data.get(feat, lo)
            pct    = (val - lo) / (hi - lo)
            color  = ACCENT[feat] if pct < 0.8 else "#ef4444"
            label, unit = LABELS[feat]
            g_ph.plotly_chart(
                make_gauge(val, lo, hi, color),
                use_container_width=True,
                config={"displayModeBar": False},
                key=f"gauge_{feat}_{st.session_state.total_count}"
            )
            lbl_ph.markdown(f"<p class='g-label'>{label} ({unit})</p>", unsafe_allow_html=True)

        _gauge("Rotational speed [rpm]", gauge_speed_ph,  gauge_label1_ph)
        _gauge("Torque [Nm]",            gauge_torque_ph, gauge_label2_ph)
        _gauge("Tool wear [min]",        gauge_wear_ph,   gauge_label3_ph)

        # ── Event log ──
        log_ph.markdown(render_log_html(st.session_state.alert_log), unsafe_allow_html=True)

        # ── Failure rate ──
        total  = st.session_state.total_count
        fails  = st.session_state.fail_count
        pct    = (fails / total * 100) if total > 0 else 0
        rc     = "#22c55e" if pct < 5 else "#fbbf24" if pct < 15 else "#ef4444"
        rate_ph.markdown(f"""
        <div class="rate-bar-wrap">
          <div class="rate-bar-fill" style="background:{rc};width:{min(pct,100):.1f}%;"></div>
        </div>
        <div class="rate-row">
          <span>{fails} failures</span>
          <span style="color:{rc};font-weight:600;">{pct:.1f}%</span>
          <span>{total} readings</span>
        </div>
        """, unsafe_allow_html=True)

        last_data = data

    elif not data:
        status_ph.markdown(
            "<div class='status-ok' style='border-color:#1e2535;color:#334155;background:#0d1117;'>"
            "⏳ &nbsp;Initializing — Waiting for telemetry data...</div>",
            unsafe_allow_html=True
        )

    time.sleep(1)
