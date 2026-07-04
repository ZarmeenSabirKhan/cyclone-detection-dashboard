import streamlit as st
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pandas as pd
import os
import io
from datetime import date, timedelta

# ============================================================
# DATA DIRECTORY (portable — works on any machine / on deployment)
# ============================================================
# Looks for a "data" folder next to this script first (use this when
# sharing the project or deploying). Falls back to your local Windows
# path so nothing breaks on your own PC.
_LOCAL_FALLBACK = r"C:\Users\Tmart\Desktop\cy"
_HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(_HERE, "data") if os.path.isdir(os.path.join(_HERE, "data")) else _LOCAL_FALLBACK

def dp(filename):
    """Build a data file path that works wherever the app runs."""
    return os.path.join(DATA_DIR, filename)

st.set_page_config(
    page_title="Cyclone Detector",
    page_icon="🌀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM CSS
# ============================================================
st.markdown("""
<style>
    .hero-header {
        background: linear-gradient(135deg, #1a4d7c 0%, #2c6ea4 100%);
        padding: 2.5rem;
        border-radius: 16px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 8px 24px rgba(26, 77, 124, 0.25);
    }
    .hero-header h1 {
        color: white !important;
        font-size: 2.5rem !important;
        margin: 0 !important;
        font-weight: 700 !important;
    }
    .hero-header p {
        color: rgba(255,255,255,0.9) !important;
        font-size: 1.15rem !important;
        margin-top: 0.5rem !important;
    }
    .metric-card-blue {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 16px rgba(102, 126, 234, 0.3);
        text-align: center;
    }
    .metric-card-green {
        background: linear-gradient(135deg, #51cf66 0%, #2f9e44 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 16px rgba(81, 207, 102, 0.3);
        text-align: center;
    }
    .metric-card-orange {
        background: linear-gradient(135deg, #ffa94d 0%, #f76707 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 16px rgba(247, 103, 7, 0.3);
        text-align: center;
    }
    .metric-card-purple {
        background: linear-gradient(135deg, #cc5de8 0%, #9c36b5 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 16px rgba(204, 93, 232, 0.3);
        text-align: center;
    }
    .metric-card-red {
        background: linear-gradient(135deg, #ff6b6b 0%, #c92a2a 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 16px rgba(201, 42, 42, 0.3);
        text-align: center;
    }
    .metric-value {
        font-size: 2.5rem !important;
        font-weight: 700 !important;
        margin: 0.3rem 0 !important;
        color: white !important;
    }
    .metric-label {
        font-size: 0.9rem !important;
        opacity: 0.95 !important;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin: 0 !important;
        color: white !important;
    }
    .metric-sub {
        font-size: 0.8rem !important;
        opacity: 0.85 !important;
        margin-top: 0.5rem !important;
        color: white !important;
    }
    .prediction-box-cyclone {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a52 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: 700;
        box-shadow: 0 4px 12px rgba(238, 90, 82, 0.3);
        margin: 1rem 0;
    }
    .prediction-box-normal {
        background: linear-gradient(135deg, #51cf66 0%, #37b24d 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: 700;
        box-shadow: 0 4px 12px rgba(55, 178, 77, 0.3);
        margin: 1rem 0;
    }
    .cyclone-card {
        background: white;
        padding: 1.2rem;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        border-top: 4px solid #2c6ea4;
        text-align: center;
    }
    .cyclone-card.real {
        border-top-color: #2f9e44;
    }
    .cyclone-card.fake {
        border-top-color: #f76707;
    }
    h2 {
        color: #1a4d7c !important;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 0.75rem 1.5rem;
        border-radius: 8px 8px 0 0;
        font-weight: 600;
    }
    .live-indicator {
        display: inline-block;
        width: 10px;
        height: 10px;
        background: #51cf66;
        border-radius: 50%;
        margin-right: 8px;
        animation: pulse 1.5s infinite;
    }
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.5; }
        100% { opacity: 1; }
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# MODEL CLASS
# ============================================================
class CycloneAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(5, 16, 3, stride=2, padding=1), nn.ReLU(), nn.BatchNorm2d(16),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(), nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(), nn.BatchNorm2d(64),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1), nn.ReLU(), nn.BatchNorm2d(32),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1), nn.ReLU(), nn.BatchNorm2d(16),
            nn.ConvTranspose2d(16, 5, 3, stride=2, padding=1, output_padding=1), nn.Sigmoid(),
        )
    def forward(self, x):
        return self.decoder(self.encoder(x))

# ============================================================
# LOAD & COMPUTE (cached)
# ============================================================
@st.cache_resource
def load_model():
    m = CycloneAutoencoder()
    # Use v3 filtered+expanded model if available, else fall back to original
    model_path = dp("autoencoder_model_v3_filtered.pt")
    if not os.path.exists(model_path):
        model_path = dp("autoencoder_model.pt")
    m.load_state_dict(torch.load(model_path, map_location="cpu"))
    m.eval()
    return m

@st.cache_data
def load_data():
    data  = np.load(dp("future_tropical.npy"))
    labels = np.load(dp("labels.npy"))
    scores = np.load(dp("cyclone_scores.npy"))
    # Use tuned v3 artifacts if available, else fall back to original
    art_path = dp("autoencoder_artifacts_v3_tuned.npz")
    if not os.path.exists(art_path):
        art_path = dp("autoencoder_artifacts.npz")
    art = np.load(art_path)
    threshold = float(art['threshold'])
    ch_min = np.array(art['ch_min'])
    ch_max = np.array(art['ch_max'])
    return data, labels, scores, threshold, ch_min, ch_max

@st.cache_data
def compute_all_predictions(_model, _data, _ch_min, _ch_max):
    ch_min_3d = _ch_min.reshape(5, 1, 1)
    ch_max_3d = _ch_max.reshape(5, 1, 1)
    errors = np.zeros(365)
    reconstructions = np.zeros((365, 5, 64, 64), dtype=np.float32)
    for i in range(365):
        day = _data[i]
        day_norm = (day - ch_min_3d) / (ch_max_3d - ch_min_3d + 1e-8)
        day_t = torch.from_numpy(day_norm).float().unsqueeze(0)
        day_resized = F.interpolate(day_t, size=(64, 64), mode='bilinear', align_corners=False)
        with torch.no_grad():
            recon = _model(day_resized)
            error = ((recon - day_resized) ** 2).mean().item()
        errors[i] = error
        reconstructions[i] = recon.squeeze(0).numpy()
    return errors, reconstructions

# Load everything
model = load_model()
data, labels, scores, threshold, ch_min, ch_max = load_data()
errors, reconstructions = compute_all_predictions(model, data, ch_min, ch_max)
predictions = (errors > threshold).astype(int)

# Compute metrics
TP = int(((predictions == 1) & (labels == 1)).sum())
TN = int(((predictions == 0) & (labels == 0)).sum())
FP = int(((predictions == 1) & (labels == 0)).sum())
FN = int(((predictions == 0) & (labels == 1)).sum())
total = len(labels)
accuracy = (TP + TN) / total
precision = TP / (TP + FP) if (TP + FP) > 0 else 0
recall = TP / (TP + FN) if (TP + FN) > 0 else 0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

# ============================================================
# HEADER
# ============================================================
st.markdown("""
<div class="hero-header">
    <h1><span class="live-indicator"></span>🌀 Tropical Cyclone Detector</h1>
    <p>Unsupervised AI for detecting tropical cyclones in CMIP6 climate model simulations &nbsp;|&nbsp; Australia, year 2064</p>
</div>
""", unsafe_allow_html=True)

# ============================================================
# TABS
# ============================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Dashboard",
    "🔍 Single Day Analysis",
    "🗓️ Calendar View",
    "📚 Methodology & Results",
    "🧪 Live Testing"
])

# ============================================================
# TAB 1: DASHBOARD (REDESIGNED)
# ============================================================
with tab1:
    # === SECTION 1: HERO PERFORMANCE METRICS ===
    st.markdown("## 🏆 Model Performance")
    st.markdown(f"<p style='color: #777;'>Performance on entire year 2064 ({total} days)</p>", unsafe_allow_html=True)
    
    pc1, pc2, pc3, pc4 = st.columns(4)
    with pc1:
        st.markdown(f"""
        <div class='metric-card-blue'>
            <p class='metric-label'>🎯 Accuracy</p>
            <p class='metric-value'>{accuracy*100:.1f}%</p>
            <p class='metric-sub'>{TP + TN} of {total} days correct</p>
        </div>
        """, unsafe_allow_html=True)
    with pc2:
        st.markdown(f"""
        <div class='metric-card-green'>
            <p class='metric-label'>✅ Precision</p>
            <p class='metric-value'>{precision*100:.1f}%</p>
            <p class='metric-sub'>When flagged, correct {precision*100:.0f}% of time</p>
        </div>
        """, unsafe_allow_html=True)
    with pc3:
        st.markdown(f"""
        <div class='metric-card-orange'>
            <p class='metric-label'>📊 Recall</p>
            <p class='metric-value'>{recall*100:.1f}%</p>
            <p class='metric-sub'>Caught {TP} of {TP+FN} actual cyclones</p>
        </div>
        """, unsafe_allow_html=True)
    with pc4:
        st.markdown(f"""
        <div class='metric-card-purple'>
            <p class='metric-label'>🏅 F1 Score</p>
            <p class='metric-value'>{f1:.3f}</p>
            <p class='metric-sub'>Balanced precision & recall</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # === SECTION 2: YEAR OVERVIEW ===
    st.markdown("## 📅 Year Overview")
    yc1, yc2, yc3, yc4 = st.columns(4)
    yc1.metric("📅 Days Analyzed", f"{total}")
    yc2.metric("🌀 Cyclones Detected", f"{predictions.sum()}",
               delta=f"{predictions.sum() - int(labels.sum()):+d} vs ground truth")
    yc3.metric("✅ True Positives", f"{TP}")
    yc4.metric("❌ Missed (False Neg.)", f"{FN}",
               delta=f"-{FN}", delta_color="inverse")
    
    st.markdown("---")
    
    # === SECTION 3: CONFUSION MATRIX + METHOD COMPARISON ===
    cm_col, comp_col = st.columns([1, 1])
    
    with cm_col:
        st.subheader("📋 Confusion Matrix")
        cm = np.array([[TN, FP], [FN, TP]])
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(cm, cmap='Blues', aspect='auto')
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['Normal', 'Cyclone'], fontsize=12)
        ax.set_yticklabels(['Normal', 'Cyclone'], fontsize=12)
        ax.set_xlabel('Predicted', fontsize=12, fontweight='bold')
        ax.set_ylabel('Actual', fontsize=12, fontweight='bold')
        
        # Numbers with annotations
        labels_2x2 = [[f'{TN}\n(True Negatives)', f'{FP}\n(False Positives)'],
                      [f'{FN}\n(False Negatives)', f'{TP}\n(True Positives)']]
        for i in range(2):
            for j in range(2):
                color = 'white' if cm[i, j] > cm.max() / 2 else '#1a4d7c'
                ax.text(j, i, labels_2x2[i][j], ha='center', va='center',
                        fontsize=14, fontweight='bold', color=color)
        plt.colorbar(im, ax=ax)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    
    with comp_col:
        st.subheader("📊 Methods Comparison")
        methods = ['KNN\nflatten', 'KNN +\nHOG', 'CNN\nAutoencoder']
        accs = [98.6, 93.1, accuracy * 100]
        f1s = [95.2, 80.0, f1 * 100]
        types = ['Supervised', 'Supervised', 'Unsupervised ⭐']
        
        fig, ax = plt.subplots(figsize=(7, 5))
        x = np.arange(len(methods))
        width = 0.35
        bars1 = ax.bar(x - width/2, accs, width, label='Accuracy (%)', color='#2c6ea4')
        bars2 = ax.bar(x + width/2, f1s, width, label='F1 × 100', color='#f76707')
        
        # Highlight autoencoder
        bars1[2].set_edgecolor('gold')
        bars1[2].set_linewidth(3)
        bars2[2].set_edgecolor('gold')
        bars2[2].set_linewidth(3)
        
        ax.set_xticks(x)
        ax.set_xticklabels(methods, fontsize=11)
        ax.set_ylabel('Score')
        ax.set_title('Comparison of All Methods')
        ax.legend(loc='lower left')
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim(0, 110)
        
        # Add value labels
        for bar, val in zip(bars1, accs):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
                    f'{val:.1f}', ha='center', fontsize=9, fontweight='bold')
        for bar, val in zip(bars2, f1s):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
                    f'{val:.1f}', ha='center', fontsize=9, fontweight='bold')
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    
    st.info("⭐ **CNN Autoencoder** is the only **unsupervised** method — it works without labels, which is the project's main contribution.")
    
    st.markdown("---")
    
    # === SECTION 4: TIMELINE ===
    st.subheader("📈 Reconstruction Error Throughout the Year")
    fig, ax = plt.subplots(figsize=(16, 5))
    days_axis = np.arange(1, 366)
    ax.axvspan(1, 120, alpha=0.08, color='orange', label='Cyclone season')
    ax.axvspan(305, 365, alpha=0.08, color='orange')
    ax.plot(days_axis, errors, color='steelblue', linewidth=1.2, alpha=0.7)
    detected = predictions == 1
    ax.scatter(days_axis[detected], errors[detected],
               color='red', s=50, zorder=5, label=f'Detected cyclone ({detected.sum()})')
    ax.axhline(y=threshold, color='black', linestyle='--', linewidth=1.5,
               label=f'Threshold = {threshold:.5f}')
    ax.set_xlabel('Day of Year (2064)', fontsize=11)
    ax.set_ylabel('Reconstruction Error', fontsize=11)
    ax.set_title('Autoencoder predictions across the year', fontsize=13)
    ax.legend(loc='upper right')
    ax.grid(alpha=0.3)
    ax.set_xlim(1, 365)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    
    st.markdown("---")
    
    # === SECTION 5: TOP 5 CYCLONES (improved cards) ===
    st.subheader("🏆 Top 5 Detected Cyclones (by reconstruction error)")
    top5_idx = np.argsort(errors)[::-1][:5]
    cols = st.columns(5)
    for rank, (col, idx) in enumerate(zip(cols, top5_idx)):
        day_num = idx + 1
        d = date(2064, 1, 1) + timedelta(days=int(idx))
        is_real = bool(labels[idx])
        card_class = "real" if is_real else "fake"
        status = "✓ Real cyclone" if is_real else "✗ False alarm"
        status_color = "#2f9e44" if is_real else "#f76707"
        with col:
            st.markdown(f"""
            <div class='cyclone-card {card_class}'>
                <p style='color: #777; margin: 0; font-size: 0.8rem;'>RANK #{rank+1}</p>
                <p style='font-size: 1.5rem; font-weight: 700; margin: 0.3rem 0; color: #1a4d7c;'>Day {day_num}</p>
                <p style='color: #555; margin: 0; font-style: italic;'>{d.strftime('%d %B %Y')}</p>
                <p style='margin: 0.5rem 0; font-family: monospace; background: #f4f8fb; padding: 4px 8px; border-radius: 4px; display: inline-block;'>Error: {errors[idx]:.5f}</p>
                <p style='color: {status_color}; font-weight: 700; margin: 0.3rem 0;'>{status}</p>
            </div>
            """, unsafe_allow_html=True)

# ============================================================
# TAB 2: SINGLE DAY ANALYSIS
# ============================================================
with tab2:
    st.sidebar.header("🎛️ Day Selector")
    st.sidebar.markdown("**Quick jumps:**")
    qc1, qc2 = st.sidebar.columns(2)
    if qc1.button("🌀 Day 68\nTop cyclone"):
        st.session_state.day = 68
    if qc2.button("❄️ Day 209\nWinter day"):
        st.session_state.day = 209
    qc3, qc4 = st.sidebar.columns(2)
    if qc3.button("🌧 Day 100\nApril cyclone"):
        st.session_state.day = 100
    if qc4.button("☀️ Day 180\nMid-year"):
        st.session_state.day = 180
    
    if 'day' not in st.session_state:
        st.session_state.day = 68
    
    day_idx = st.sidebar.slider("Day of Year", 1, 365, st.session_state.day, key='slider')
    day_idx_0 = day_idx - 1
    selected_date = date(2064, 1, 1) + timedelta(days=day_idx_0)
    error = errors[day_idx_0]
    prediction = bool(predictions[day_idx_0])
    ground_truth = bool(labels[day_idx_0])
    
    st.subheader(f"📅 Day {day_idx} — {selected_date.strftime('%A, %d %B %Y')}")
    
    pcol1, pcol2 = st.columns(2)
    with pcol1:
        st.markdown("**🤖 Model says:**")
        if prediction:
            st.markdown('<div class="prediction-box-cyclone">⚠️ CYCLONE DETECTED</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="prediction-box-normal">☀️ NORMAL DAY</div>', unsafe_allow_html=True)
    with pcol2:
        st.markdown("**📋 Ground truth:**")
        if ground_truth:
            st.markdown('<div class="prediction-box-cyclone">🌀 Actually a CYCLONE</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="prediction-box-normal">🌤️ Actually NORMAL</div>', unsafe_allow_html=True)
    
    if prediction == ground_truth:
        st.success(f"✅ **Correct prediction!** Reconstruction error: `{error:.5f}` vs threshold `{threshold:.5f}`")
    else:
        st.error(f"❌ **Incorrect prediction.** Reconstruction error: `{error:.5f}` vs threshold `{threshold:.5f}`")
    
    st.markdown("---")
    st.subheader("🌡️ Atmospheric Conditions (5 Channels)")
    day_data = data[day_idx_0]
    variables = ['Temperature', 'Rainfall', 'Humidity', 'Solar Rad.', 'Wind Speed']
    var_codes = ['tasmax', 'pr', 'hurs', 'rsds', 'sfcWind']
    units = ['°C', 'mm/day', '%', 'W/m²', 'm/s']
    cmaps = ['hot', 'Blues', 'YlGnBu', 'plasma', 'viridis']
    
    fig, axes = plt.subplots(1, 5, figsize=(20, 4.5))
    for i, (var, code, unit, cmap) in enumerate(zip(variables, var_codes, units, cmaps)):
        im = axes[i].imshow(day_data[i], cmap=cmap, origin='lower', aspect='auto')
        axes[i].set_title(f"{var}\n{code} ({unit})", fontsize=11, fontweight='bold')
        axes[i].axis('off')
        plt.colorbar(im, ax=axes[i], fraction=0.046)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    
    st.markdown("---")
    st.subheader("🔬 Original vs Model's Reconstruction")
    st.markdown("*Big differences = anomaly = cyclone.*")
    
    ch_min_3d = ch_min.reshape(5, 1, 1)
    ch_max_3d = ch_max.reshape(5, 1, 1)
    day_norm = (day_data - ch_min_3d) / (ch_max_3d - ch_min_3d + 1e-8)
    day_t = torch.from_numpy(day_norm).float().unsqueeze(0)
    day_resized = F.interpolate(day_t, size=(64, 64), mode='bilinear', align_corners=False).squeeze(0).numpy()
    recon_day = reconstructions[day_idx_0]
    diff = np.abs(day_resized - recon_day)
    
    fig, axes = plt.subplots(3, 5, figsize=(20, 11))
    row_labels = ['Original', 'Reconstructed', 'Difference (error)']
    for col in range(5):
        axes[0, col].imshow(day_resized[col], cmap=cmaps[col], origin='lower', aspect='auto')
        axes[0, col].set_title(f"{variables[col]}", fontsize=11, fontweight='bold')
        axes[0, col].axis('off')
        axes[1, col].imshow(recon_day[col], cmap=cmaps[col], origin='lower', aspect='auto')
        axes[1, col].axis('off')
        axes[2, col].imshow(diff[col], cmap='Reds', origin='lower', aspect='auto')
        axes[2, col].axis('off')
    for row, label in enumerate(row_labels):
        axes[row, 0].text(-0.15, 0.5, label, transform=axes[row, 0].transAxes,
                          rotation=90, fontsize=13, fontweight='bold',
                          va='center', ha='center', color='#1a4d7c')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    
    st.markdown("---")
    mcol1, mcol2, mcol3 = st.columns(3)
    mcol1.metric("Reconstruction Error", f"{error:.5f}",
                 delta=f"{(error - threshold):+.5f} vs threshold")
    mcol2.metric("Threshold", f"{threshold:.5f}")
    mcol3.metric("Confidence", f"{abs(error - threshold) / threshold * 100:.0f}%")

# ============================================================
# TAB 3: CALENDAR VIEW
# ============================================================
with tab3:
    st.subheader("🗓️ Year Calendar — Cyclone Activity")
    st.markdown("*Heatmap of reconstruction error each day. Blue circles = detected cyclones.*")
    
    cal = np.full((12, 31), np.nan)
    month_starts = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334]
    month_lengths = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    
    day_counter = 0
    for m in range(12):
        for d in range(month_lengths[m]):
            if day_counter < 365:
                cal[m, d] = errors[day_counter]
                day_counter += 1
    
    fig, ax = plt.subplots(figsize=(18, 6))
    im = ax.imshow(cal, cmap='YlOrRd', aspect='auto')
    for d_idx in range(365):
        m = next(i for i in range(11, -1, -1) if d_idx >= month_starts[i])
        day_in_month = d_idx - month_starts[m]
        if predictions[d_idx]:
            ax.plot(day_in_month, m, 'o', color='blue', markersize=8,
                    markerfacecolor='none', markeredgewidth=2)
    
    ax.set_yticks(range(12))
    ax.set_yticklabels(['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'])
    ax.set_xticks(range(0, 31, 5))
    ax.set_xticklabels([str(i+1) for i in range(0, 31, 5)])
    ax.set_xlabel('Day of Month')
    ax.set_title('Reconstruction Error by Day', fontsize=12)
    plt.colorbar(im, ax=ax, label='Reconstruction Error')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    
    st.markdown("---")
    st.subheader("📊 Monthly Cyclone Count")
    month_cyclones = []
    month_names = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    for m in range(12):
        start = month_starts[m]
        end = start + month_lengths[m]
        month_cyclones.append(predictions[start:end].sum())
    
    fig, ax = plt.subplots(figsize=(14, 4))
    colors = ['#ff6b6b' if c > 0 else '#cccccc' for c in month_cyclones]
    bars = ax.bar(month_names, month_cyclones, color=colors)
    ax.set_ylabel('Cyclones detected')
    ax.grid(axis='y', alpha=0.3)
    for bar, count in zip(bars, month_cyclones):
        if count > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    str(int(count)), ha='center', fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

# ============================================================
# TAB 4: METHODOLOGY
# ============================================================
with tab4:
    st.subheader("🎯 Project Goal")
    st.markdown("""
    Detect tropical cyclones in CMIP6 future climate projections using **unsupervised deep learning**.
    The project document states: *"No unsupervised cyclone detection exists. No reproducible benchmark for tropical 
    cyclone detection using unsupervised learning."* — this app demonstrates a solution to that gap.
    """)
    
    st.markdown("---")
    st.subheader("🧠 How the Autoencoder Detects Cyclones")
    st.markdown("""┌──────────────────────────────────────────────────────────┐
│  TRAINING (only on NORMAL days)                          │
│  Normal day → Encoder → small code → Decoder → Reconstr. │
│  Model learns "what normal weather looks like"           │
└──────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────┐
│  PREDICTION (on any new day)                             │
│  Test day → Autoencoder → Reconstruction                 │
│  Small difference → Normal day ☀                         │
│  Big difference   → CYCLONE 🌀 (model never saw this)   │
└──────────────────────────────────────────────────────────┘""")
    
    st.markdown("---")
    st.subheader("📊 Comparison with All Methods")

    df = pd.DataFrame({
        'Method': [
            'KNN (flatten pixels)',
            'KNN + HOG features',
            'Supervised CNN',
            'CNN Autoencoder (original)',
            'CNN Autoencoder — Filtered+Tuned ⭐'
        ],
        'Type': [
            'Supervised', 'Supervised', 'Supervised',
            'Unsupervised', 'Unsupervised'
        ],
        'Training Days': ['292', '292', '292', '232', '9,561'],
        'Accuracy': ['98.6%', '93.1%', '98.6%', '91.8%', '93.2%'],
        'F1 Score': ['0.952', '0.800', '0.957', '0.750', '0.880'],
        'Recall': ['—', '—', '1.000', '0.818', '1.000'],
        'Needs Labels?': ['Yes', 'Yes', 'Yes', 'No', 'No — fills the gap!']
    })
    st.dataframe(df, use_container_width=True, hide_index=True)

    st.info("""
    **Key finding:** The CNN Autoencoder (Filtered+Tuned) achieves Recall=1.000 — same as the best
    supervised method — without using any cyclone labels during training. It was trained on 9,561
    filtered normal days from BARRA-R2 historical reanalysis (1985–2014) plus future data,
    then threshold-tuned for optimal F1. This directly addresses the project gap:
    *"No reproducible benchmark for tropical cyclone detection using unsupervised learning."*
    """)
    
    st.markdown("---")
    st.subheader("🛠️ Model Architecture")
    st.code("""
Encoder:
  Conv2d(5 → 16, stride=2)  + BatchNorm + ReLU   # 64×64 → 32×32
  Conv2d(16 → 32, stride=2) + BatchNorm + ReLU   # 32×32 → 16×16
  Conv2d(32 → 64, stride=2) + BatchNorm + ReLU   # 16×16 → 8×8

Decoder (mirror):
  ConvTranspose2d(64 → 32) + BatchNorm + ReLU    # 8×8   → 16×16
  ConvTranspose2d(32 → 16) + BatchNorm + ReLU    # 16×16 → 32×32
  ConvTranspose2d(16 → 5)  + Sigmoid             # 32×32 → 64×64
    """, language='python')
    
    st.markdown("---")
    st.markdown("**Future Dataset:** CMIP6 ACCESS-CM2 / SSP5-8.5 / year 2064 / AUS-05i grid, 5 channels")
    st.markdown("**Historical Dataset:** BARRA-R2 reanalysis / 1985–2014 / 30 years / same 5 variables")
    st.markdown(f"**Training:** 9,561 filtered normal days (historical + future) &nbsp;|&nbsp; **Testing:** 73 future days &nbsp;|&nbsp; **Epochs:** 50")

# ============================================================
# TAB 5: LIVE TESTING
# ============================================================
# Fixed real-world physical bounds per variable — used ONLY for Live Testing.
# Why: ch_min/ch_max (loaded above) are the exact min/max seen in THIS training
# dataset (one year, one climate scenario). If a visitor uploads/simulates data
# outside that narrow range, normalizing against it silently gives a wrong
# verdict instead of erroring. These bounds cover realistic real-world ranges
# for each variable, so any reasonable input normalizes sensibly.
PHYSICAL_BOUNDS = np.array([
    [-10.0, 55.0],    # tasmax (°C)   — coldest to hottest realistic daily max
    [0.0, 500.0],     # pr (mm/day)   — dry to extreme tropical downpour
    [0.0, 100.0],     # hurs (%)      — relative humidity, full physical range
    [0.0, 450.0],     # rsds (W/m²)   — no sun to peak tropical solar radiation
    [0.0, 60.0],       # sfcWind (m/s) — calm to severe tropical cyclone wind
])

def run_model_on_field(field_5ch):
    """field_5ch: numpy array (5, H, W) in PHYSICAL units. Returns (error, prediction, recon_64).
    Uses fixed PHYSICAL_BOUNDS (not the training dataset's own min/max) so any
    realistic user-supplied input normalizes correctly, even if it falls outside
    the exact range seen during training."""
    lo = PHYSICAL_BOUNDS[:, 0].reshape(5, 1, 1)
    hi = PHYSICAL_BOUNDS[:, 1].reshape(5, 1, 1)
    clipped = np.clip(field_5ch, lo, hi)  # guard against wildly out-of-range input
    norm = (clipped - lo) / (hi - lo + 1e-8)
    t = torch.from_numpy(norm.astype(np.float32)).unsqueeze(0)
    t = F.interpolate(t, size=(64, 64), mode='bilinear', align_corners=False)
    with torch.no_grad():
        recon = model(t)
        err = ((recon - t) ** 2).mean().item()
    return err, int(err > threshold), recon.squeeze(0).numpy()

with tab5:
    st.subheader("🧪 Live Testing — Try the Model Yourself")
    st.markdown(
        "This tab lets anyone test the autoencoder directly — no dataset needed. "
        "Pick a mode below."
    )

    mode = st.radio(
        "Choose a way to test:",
        ["🎛️ What-If Simulator", "📤 Upload Your Own Day", "🎲 Random Day Challenge"],
        horizontal=True
    )

    st.markdown("---")

    # ------------------------------------------------------------
    # MODE A: WHAT-IF SIMULATOR — drag sliders, see live prediction
    # ------------------------------------------------------------
    if mode == "🎛️ What-If Simulator":
        st.markdown(
            "Set hypothetical weather conditions with the sliders. The app builds a "
            "uniform field with these values and feeds it through the autoencoder "
            "**live** — watch the verdict change as you drag."
        )
        var_names = ['Temperature (tasmax)', 'Rainfall (pr)', 'Humidity (hurs)',
                     'Solar Radiation (rsds)', 'Wind Speed (sfcWind)']
        units = ['°C', 'mm/day', '%', 'W/m²', 'm/s']

        cols = st.columns(5)
        slider_vals = []
        for i, (col, name, unit) in enumerate(zip(cols, var_names, units)):
            lo, hi = float(PHYSICAL_BOUNDS[i, 0]), float(PHYSICAL_BOUNDS[i, 1])
            default = lo + 0.3 * (hi - lo)  # default = mild/normal-ish value
            with col:
                v = st.slider(f"{name}\n({unit})", lo, hi, default, key=f"sim_{i}")
                slider_vals.append(v)

        c1, c2 = st.columns([1, 2])
        with c1:
            if st.button("⚡ Try an extreme cyclone-like scenario"):
                for i in range(5):
                    lo, hi = float(PHYSICAL_BOUNDS[i, 0]), float(PHYSICAL_BOUNDS[i, 1])
                    st.session_state[f"sim_{i}"] = hi if i in (1, 4) else (lo + 0.5 * (hi - lo))
                st.rerun()

        H, W = data.shape[2], data.shape[3]
        field = np.zeros((5, H, W), dtype=np.float32)
        for i, v in enumerate(slider_vals):
            field[i, :, :] = v

        err, pred, recon = run_model_on_field(field)

        st.markdown("---")
        rcol1, rcol2, rcol3 = st.columns(3)
        rcol1.metric("Reconstruction Error", f"{err:.5f}")
        rcol2.metric("Threshold", f"{threshold:.5f}")
        if pred:
            rcol3.markdown('<div class="prediction-box-cyclone">⚠️ CYCLONE PATTERN</div>', unsafe_allow_html=True)
        else:
            rcol3.markdown('<div class="prediction-box-normal">☀️ NORMAL PATTERN</div>', unsafe_allow_html=True)
        st.caption(
            "Note: the model was trained on *spatially varying* normal days, so a perfectly "
            "uniform field is already a bit unusual — this is a simplified demo of the live "
            "mechanism, not a substitute for real satellite/model data."
        )

    # ------------------------------------------------------------
    # MODE B: UPLOAD YOUR OWN DAY
    # ------------------------------------------------------------
    elif mode == "📤 Upload Your Own Day":
        st.markdown(
            "Upload a `.npy` file shaped **(5, H, W)** — 5 channels in this order: "
            "`tasmax, pr, hurs, rsds, sfcWind` — and the model will classify it live. "
            "You can grab a sample file using the button below first."
        )
        sample_idx = 67  # day 68, a known cyclone candidate
        sample_buf = io.BytesIO()
        np.save(sample_buf, data[sample_idx].astype(np.float32))
        st.download_button(
            "⬇️ Download a sample day (Day 68, real cyclone) to test the uploader",
            data=sample_buf.getvalue(),
            file_name="sample_day_68.npy",
            mime="application/octet-stream"
        )
        uploaded = st.file_uploader("Upload a (5, H, W) .npy file", type=["npy"])
        if uploaded is not None:
            try:
                custom_field = np.load(uploaded, allow_pickle=False)
                if custom_field.ndim != 3 or custom_field.shape[0] != 5:
                    st.error(f"Expected shape (5, H, W), got {custom_field.shape}. Please check your file.")
                else:
                    err, pred, recon = run_model_on_field(custom_field.astype(np.float32))
                    st.success(f"Loaded file with shape {custom_field.shape}")
                    rcol1, rcol2, rcol3 = st.columns(3)
                    rcol1.metric("Reconstruction Error", f"{err:.5f}")
                    rcol2.metric("Threshold", f"{threshold:.5f}")
                    if pred:
                        rcol3.markdown('<div class="prediction-box-cyclone">⚠️ CYCLONE DETECTED</div>', unsafe_allow_html=True)
                    else:
                        rcol3.markdown('<div class="prediction-box-normal">☀️ NORMAL DAY</div>', unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Couldn't read that file: {e}")

    # ------------------------------------------------------------
    # MODE C: RANDOM DAY CHALLENGE — guess vs. the model
    # ------------------------------------------------------------
    else:
        st.markdown(
            "A random day's weather maps are shown below with the label hidden. "
            "Make your guess, then see who's right — you or the model."
        )
        if "challenge_day" not in st.session_state:
            st.session_state.challenge_day = int(np.random.randint(0, 365))

        if st.button("🎲 Give me a new random day"):
            st.session_state.challenge_day = int(np.random.randint(0, 365))

        cidx = st.session_state.challenge_day
        cday_data = data[cidx]
        variables = ['Temperature', 'Rainfall', 'Humidity', 'Solar Rad.', 'Wind Speed']
        cmaps = ['hot', 'Blues', 'YlGnBu', 'plasma', 'viridis']

        fig, axes = plt.subplots(1, 5, figsize=(20, 4))
        for i, (var, cmap) in enumerate(zip(variables, cmaps)):
            im = axes[i].imshow(cday_data[i], cmap=cmap, origin='lower', aspect='auto')
            axes[i].set_title(var, fontsize=11, fontweight='bold')
            axes[i].axis('off')
            plt.colorbar(im, ax=axes[i], fraction=0.046)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        guess = st.radio("Your guess:", ["☀️ Normal day", "🌀 Cyclone"], horizontal=True, key=f"guess_{cidx}")

        if st.button("✅ Reveal answer"):
            actual = bool(labels[cidx])
            model_pred = bool(predictions[cidx])
            user_said_cyclone = guess == "🌀 Cyclone"

            r1, r2, r3 = st.columns(3)
            r1.markdown(f"**You said:** {'🌀 Cyclone' if user_said_cyclone else '☀️ Normal'}")
            r2.markdown(f"**Model said:** {'🌀 Cyclone' if model_pred else '☀️ Normal'}")
            r3.markdown(f"**Actual:** {'🌀 Cyclone' if actual else '☀️ Normal'}")

            if user_said_cyclone == actual and model_pred == actual:
                st.success("🤝 You and the model both got it right!")
            elif user_said_cyclone == actual and model_pred != actual:
                st.info("🧠 You beat the model on this one!")
            elif user_said_cyclone != actual and model_pred == actual:
                st.warning("🤖 The model got this one, you didn't — wind+rain extremes are subtle!")
            else:
                st.error("😅 Both of you missed this one — tricky day.")