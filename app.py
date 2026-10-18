import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
os.environ["HF_HUB_OFFLINE"] = "1"

import streamlit as st
import numpy as np
import torch
import plotly.graph_objects as go
import json

from models.vlm_grounding import load_vlm_model
from utils.box_utils import make_bbox_lines, make_bbox_traces, calculate_3d_iou, get_box_corners
from utils.data_loader import load_scanrefer_annotations
from utils.import_point_cloud import parse_ply

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

st.set_page_config(
    page_title="3D Vision-Language Studio | ScanRefer",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom Modern Theme CSS ---
st.markdown("""
<style>
    /* Global font & background adjustments */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Header Card */
    .vlm-hero {
        background: linear-gradient(135deg, rgba(30, 27, 75, 0.85) 0%, rgba(49, 46, 129, 0.85) 50%, rgba(67, 56, 202, 0.85) 100%);
        border: 1px solid rgba(129, 140, 248, 0.25);
        border-radius: 18px;
        padding: 24px 30px;
        margin-bottom: 24px;
        backdrop-filter: blur(12px);
        box-shadow: 0 12px 30px -8px rgba(0, 0, 0, 0.45);
    }
    .vlm-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.72rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        background: rgba(99, 102, 241, 0.25);
        color: #C7D2FE;
        border: 1px solid rgba(165, 180, 252, 0.35);
        margin-bottom: 10px;
    }
    .vlm-title {
        font-size: 1.85rem;
        font-weight: 800;
        color: #FFFFFF;
        letter-spacing: -0.025em;
        margin: 0 0 6px 0;
    }
    .vlm-subtitle {
        color: #E0E7FF;
        font-size: 0.95rem;
        line-height: 1.5;
        margin: 0;
        max-width: 850px;
    }

    /* Metric Cards */
    .metric-card {
        background: linear-gradient(145deg, rgba(24, 24, 27, 0.75), rgba(39, 39, 42, 0.55));
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 16px 20px;
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.2);
        backdrop-filter: blur(10px);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .metric-card:hover {
        border-color: rgba(129, 140, 248, 0.4);
        transform: translateY(-2px);
    }
    .metric-label {
        color: #A1A1AA;
        font-size: 0.78rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 6px;
        display: flex;
        align-items: center;
        gap: 6px;
    }
    .metric-value {
        font-size: 1.7rem;
        font-weight: 800;
        color: #FAFAFA;
        letter-spacing: -0.02em;
    }
    .metric-pill {
        display: inline-block;
        padding: 3px 8px;
        border-radius: 6px;
        font-size: 0.72rem;
        font-weight: 700;
        margin-top: 6px;
    }
    .pill-green {
        background: rgba(16, 185, 129, 0.2);
        color: #34D399;
        border: 1px solid rgba(16, 185, 129, 0.4);
    }
    .pill-amber {
        background: rgba(245, 158, 11, 0.2);
        color: #FBBF24;
        border: 1px solid rgba(245, 158, 11, 0.4);
    }
    .pill-cyan {
        background: rgba(6, 182, 212, 0.2);
        color: #22D3EE;
        border: 1px solid rgba(6, 182, 212, 0.4);
    }
</style>
""", unsafe_allow_html=True)

# --- Hero Header ---
st.markdown("""
<div class="vlm-hero">
    <div class="vlm-badge">
        <span>⚡ Multimodal 3D Vision-Language Studio</span>
        <span>•</span>
        <span>ScanRefer v2</span>
    </div>
    <div class="vlm-title">🎯 Language-Guided 3D Object Localization</div>
    <div class="vlm-subtitle">
        Query dense indoor 3D point clouds using free-form natural language utterances.
        Predicts metric 3D bounding boxes <code style="color:#A5B4FC; background:rgba(0,0,0,0.3); padding:2px 6px; border-radius:4px;">[cx, cy, cz, dx, dy, dz]</code> via cross-attention proposal fusion.
    </div>
</div>
""", unsafe_allow_html=True)

# --- Model & Data Caching ---
@st.cache_resource
def get_model(ckpt_name: str):
    ckpt_path = os.path.join(BASE_DIR, "checkpoints", ckpt_name) if ckpt_name.endswith(".pth") else None
    return load_vlm_model(checkpoint_path=ckpt_path)

@st.cache_data
def get_all_annotations():
    annotations = []
    for split in ["ScanRefer_filtered_val.json", "ScanRefer_filtered_train.json"]:
        path = os.path.join(BASE_DIR, "data", "scanrefer", split)
        if os.path.exists(path):
            annotations.extend(load_scanrefer_annotations(path))
    return annotations

annotations = get_all_annotations()

# --- Sidebar Controls ---
st.sidebar.markdown("### ⚙️ Studio Controls")

# 1. Checkpoint Selector
checkpoints_dir = os.path.join(BASE_DIR, "checkpoints")
available_ckpts = [f for f in os.listdir(checkpoints_dir) if f.endswith(".pth")] if os.path.exists(checkpoints_dir) else []
available_ckpts.sort(key=lambda x: "finetuned" not in x)

if not available_ckpts:
    available_ckpts = ["Base Architecture (Pretrained BERT)"]

selected_ckpt = st.sidebar.selectbox(
    "Active Model Checkpoint",
    available_ckpts,
    index=0,
    help="Select between the fine-tuned high-accuracy weights or the base weights."
)
model, tokenizer, device = get_model(selected_ckpt)

# 2. Scene Selector
scans_dir = os.path.join(BASE_DIR, "data", "scannet", "scans")
available_scene_files = [f for f in os.listdir(scans_dir) if f.endswith(".npy") and not f.startswith("temp_")] if os.path.exists(scans_dir) else []
scene_labels = {
    "scene0001_00.npy": "🛋️ Living Room (scene0001_00)",
    "scene0000_00.npy": "🏢 Office (scene0000_00)"
}
scene_options = {scene_labels.get(f, f"🏠 {f.replace('.npy', '')}"): os.path.join(scans_dir, f) for f in available_scene_files}

if not scene_options:
    st.sidebar.error("No scene point cloud files found in data/scannet/scans/")
    st.stop()

selected_scene_label = st.sidebar.selectbox("Select 3D Environment", list(scene_options.keys()))
selected_scene_file = os.path.basename(scene_options[selected_scene_label])
current_scene_id = selected_scene_file.replace(".npy", "")

# 3. Quick-Pick Presets
st.sidebar.markdown("---")
st.sidebar.markdown("### 💬 Natural Language Query")

scene_annotations = [a for a in annotations if a.get("scene_id") == current_scene_id]
query_mode = st.sidebar.radio("Input Source", ["Verified Benchmark Queries", "Custom Free-form Text"])

gt_box = None
query_text = ""
selected_item = None

if query_mode == "Verified Benchmark Queries" and scene_annotations:
    desc_options = {
        f"{item['object_name'].upper()}: {item['description']}": item
        for item in scene_annotations
    }
    selected_desc_key = st.sidebar.selectbox("Choose Ground Truth Query", list(desc_options.keys()))
    selected_item = desc_options[selected_desc_key]
    query_text = selected_item["description"]
    gt_box = np.array(selected_item["bbox"], dtype=np.float32)
else:
    default_text = scene_annotations[0]["description"] if scene_annotations else "the black office chair near the desk"
    query_text = st.sidebar.text_area("Type Query Utterance", value=default_text, height=100)

# 4. Visual Appearance & Camera Controls
with st.sidebar.expander("🎨 3D Display & Lighting Options", expanded=False):
    box_rendering_style = st.radio("3D Box Style", ["Solid 3D Volume + Wireframe", "Wireframe Lines Only"])
    box_opacity = st.slider("3D Box Volume Opacity", min_value=0.1, max_value=0.8, value=0.35, step=0.05)
    point_size = st.slider("Point Cloud Particle Size", min_value=1.0, max_value=5.0, value=2.0, step=0.5)
    point_opacity = st.slider("Point Cloud Opacity", min_value=0.2, max_value=1.0, value=0.75, step=0.05)
    point_sample_count = st.slider("Points to Render", min_value=1024, max_value=8192, value=4096, step=512)

# 5. Data Uploader
with st.sidebar.expander("📥 Import New 3D Scan (.ply / .npy)", expanded=False):
    uploaded_file = st.file_uploader("Upload Point Cloud", type=["ply", "npy"])
    custom_scene_name = st.text_input("New Scene ID", value="scene_custom_01")
    if st.button("Register & Load Scan", use_container_width=True) and uploaded_file is not None:
        target_path = os.path.join(BASE_DIR, "data", "scannet", "scans", f"{custom_scene_name}.npy")
        temp_input = os.path.join(BASE_DIR, "data", "scannet", "scans", f"temp_{uploaded_file.name}")
        with open(temp_input, "wb") as f:
            f.write(uploaded_file.getbuffer())
        try:
            if uploaded_file.name.endswith(".ply"):
                pts = parse_ply(temp_input)
            else:
                pts = np.load(temp_input)
                if pts.shape[1] == 3:
                    pts = np.concatenate([pts, np.ones_like(pts) * 128.0], axis=-1)
            np.save(target_path, pts.astype(np.float32))
            st.success(f"Registered `{custom_scene_name}`! ({len(pts):,} points)")
            if os.path.exists(temp_input):
                os.remove(temp_input)
            st.rerun()
        except Exception as e:
            st.error(f"Error importing scan: {e}")

st.sidebar.markdown("---")
run_button = st.sidebar.button("🚀 Run 3D Localization", type="primary", use_container_width=True)

# --- Quick Preset Bar on Main Page (if idle) ---
if not st.session_state.get("has_run", False) and not run_button:
    st.markdown("#### ⚡ Quick-Start Benchmark Scenarios")
    st.markdown("Select a ready-to-test real scenario to test the multimodal grounding:")
    
    qcol1, qcol2, qcol3 = st.columns(3)
    with qcol1:
        if st.button("🪑 Office: Black Chair", use_container_width=True):
            st.session_state["has_run"] = True
            st.rerun()
    with qcol2:
        if st.button("🛋️ Living Room: Gray Sofa", use_container_width=True):
            st.session_state["has_run"] = True
            st.rerun()
    with qcol3:
        if st.button("💻 Office: Monitor on Desk", use_container_width=True):
            st.session_state["has_run"] = True
            st.rerun()

# --- Main Inference & Visualization Execution ---
if run_button or st.session_state.get("has_run", False):
    st.session_state["has_run"] = True
    scene_path = scene_options[selected_scene_label]

    if not os.path.exists(scene_path):
        st.error(f"Scene point cloud file not found: {scene_path}")
        st.stop()

    pts_full = np.load(scene_path)
    indices = np.random.choice(len(pts_full), point_sample_count, replace=(len(pts_full) < point_sample_count))
    pts_sub = pts_full[indices]

    xyz = torch.tensor(pts_sub[:, :3], dtype=torch.float32).unsqueeze(0).to(device)
    rgb = torch.tensor(pts_sub[:, 3:6] / 255.0, dtype=torch.float32).unsqueeze(0).to(device)

    tokens = tokenizer(query_text, return_tensors="pt", padding="max_length", max_length=32, truncation=True)
    input_ids = tokens["input_ids"].to(device)
    mask = tokens["attention_mask"].to(device)

    with torch.no_grad():
        c_logits, pred_b = model(xyz, rgb, input_ids, mask)
        best_idx = torch.argmax(c_logits[0])
        pred_box = pred_b[0, best_idx].cpu().numpy()
        confidence = torch.sigmoid(c_logits[0, best_idx]).item()

    # Calculate IoU if Ground Truth is available
    iou = None
    center_dist = None
    pred_vol = float(np.prod(np.maximum(pred_box[3:6], 0)))
    gt_vol = None

    if gt_box is not None:
        iou = calculate_3d_iou(pred_box, gt_box)
        center_dist = float(np.linalg.norm(pred_box[:3] - gt_box[:3]))
        gt_vol = float(np.prod(np.maximum(gt_box[3:6], 0)))

    # --- Metrics Bar ---
    mcol1, mcol2, mcol3, mcol4 = st.columns(4)
    with mcol1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">🎯 Proposal Confidence</div>
            <div class="metric-value">{confidence * 100:.1f}%</div>
            <span class="metric-pill pill-cyan">Top Proposal #{best_idx+1}/64</span>
        </div>
        """, unsafe_allow_html=True)

    with mcol2:
        if iou is not None:
            pill_class = "pill-green" if iou >= 0.25 else "pill-amber"
            status_text = "Accurate (IoU ≥ 0.25)" if iou >= 0.25 else "Low Overlap (< 0.25)"
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">📐 3D IoU Overlap</div>
                <div class="metric-value">{iou * 100:.1f}%</div>
                <span class="metric-pill {pill_class}">{status_text}</span>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="metric-card">
                <div class="metric-label">📐 3D IoU Overlap</div>
                <div class="metric-value">N/A</div>
                <span class="metric-pill pill-amber">Custom Query</span>
            </div>
            """, unsafe_allow_html=True)

    with mcol3:
        dist_str = f"{center_dist:.2f} m" if center_dist is not None else "N/A"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">📍 Center Offset (Δ)</div>
            <div class="metric-value">{dist_str}</div>
            <span class="metric-pill pill-cyan">Euclidean Distance</span>
        </div>
        """, unsafe_allow_html=True)

    with mcol4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">📦 Physical Volume</div>
            <div class="metric-value">{pred_vol:.2f} m³</div>
            <span class="metric-pill pill-cyan">{point_sample_count:,} Pts Sampled</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

    # --- Multi-Tab Interface ---
    tab_studio, tab_details, tab_tokens = st.tabs([
        "🎮 Interactive 3D Studio",
        "📊 Spatial Analytics & Error Matrix",
        "💡 Multimodal Text Attention"
    ])

    with tab_studio:
        # Camera preset shortcuts
        cam_col1, cam_col2, cam_col3, cam_col4 = st.columns([1, 1, 1, 3])
        with cam_col1:
            st.caption("📷 Camera Perspective")
        with cam_col2:
            top_down = st.button("⬇️ Top-Down (Bird's Eye)", use_container_width=True)
        with cam_col3:
            side_view = st.button("➡️ Front View", use_container_width=True)

        # Plotly Traces
        fig_data = [
            go.Scatter3d(
                x=pts_sub[:, 0],
                y=pts_sub[:, 1],
                z=pts_sub[:, 2],
                mode="markers",
                marker=dict(
                    size=point_size,
                    color=pts_sub[:, 3:6] / 255.0,
                    opacity=point_opacity
                ),
                name="3D Room Points",
                hoverinfo="skip"
            )
        ]

        # Predicted Box (Red)
        if box_rendering_style == "Solid 3D Volume + Wireframe":
            fig_data.extend(make_bbox_traces(pred_box, color="#EF4444", name=f"Predicted ({confidence*100:.1f}%)", width=7, opacity=box_opacity))
        else:
            fig_data.append(make_bbox_lines(pred_box, color="#EF4444", name=f"Predicted ({confidence*100:.1f}%)", width=6))

        # Ground Truth Box (Neon Emerald)
        if gt_box is not None:
            if box_rendering_style == "Solid 3D Volume + Wireframe":
                fig_data.extend(make_bbox_traces(gt_box, color="#10B981", name="Ground Truth Box", width=7, opacity=box_opacity))
            else:
                fig_data.append(make_bbox_lines(gt_box, color="#10B981", name="Ground Truth Box", width=6))

        fig = go.Figure(data=fig_data)

        # Camera positioning
        camera_dict = dict(
            eye=dict(x=1.6, y=1.6, z=1.2),
            up=dict(x=0, y=0, z=1)
        )
        if top_down:
            camera_dict = dict(eye=dict(x=0, y=0, z=2.8), up=dict(x=0, y=1, z=0))
        elif side_view:
            camera_dict = dict(eye=dict(x=2.2, y=0, z=0.4), up=dict(x=0, y=0, z=1))

        fig.update_layout(
            paper_bgcolor="#0E131F",
            plot_bgcolor="#0E131F",
            scene=dict(
                xaxis=dict(title="X (meters)", color="#94A3B8", gridcolor="#1E293B", showbackground=False),
                yaxis=dict(title="Y (meters)", color="#94A3B8", gridcolor="#1E293B", showbackground=False),
                zaxis=dict(title="Z (meters)", color="#94A3B8", gridcolor="#1E293B", showbackground=False),
                camera=camera_dict,
                aspectmode="data"
            ),
            legend=dict(
                x=0.02,
                y=0.98,
                bgcolor="rgba(15, 23, 42, 0.8)",
                bordercolor="rgba(255,255,255,0.1)",
                borderwidth=1,
                font=dict(color="#F8FAFC")
            ),
            margin=dict(l=0, r=0, b=0, t=20),
            height=720
        )
        st.plotly_chart(fig, use_container_width=True)

        # Export Controls
        dcol1, dcol2 = st.columns(2)
        with dcol1:
            html_str = fig.to_html(include_plotlyjs="cdn")
            st.download_button(
                label="💾 Download 3D Visualization (.html)",
                data=html_str,
                file_name=f"{current_scene_id}_localization.html",
                mime="text/html",
                use_container_width=True
            )
        with dcol2:
            pred_dict = {
                "scene": current_scene_id,
                "query": query_text,
                "confidence": float(confidence),
                "predicted_bbox": {
                    "cx": round(float(pred_box[0]), 3),
                    "cy": round(float(pred_box[1]), 3),
                    "cz": round(float(pred_box[2]), 3),
                    "dx": round(float(pred_box[3]), 3),
                    "dy": round(float(pred_box[4]), 3),
                    "dz": round(float(pred_box[5]), 3)
                },
                "iou_score": float(iou) if iou is not None else None
            }
            st.download_button(
                label="📄 Export 3D Coordinates (.json)",
                data=json.dumps(pred_dict, indent=2),
                file_name=f"{current_scene_id}_bbox.json",
                mime="application/json",
                use_container_width=True
            )

    with tab_details:
        st.markdown("### 📐 Physical Dimension Comparison")
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            st.markdown("#### 🔴 Predicted Bounding Box")
            st.markdown(f"""
            - **Center $(c_x, c_y, c_z)$**: `{pred_box[0]:.3f}m, {pred_box[1]:.3f}m, {pred_box[2]:.3f}m`
            - **Dimensions $(d_x, d_y, d_z)$**: `{pred_box[3]:.3f}m, {pred_box[4]:.3f}m, {pred_box[5]:.3f}m`
            - **Calculated Volume**: `{pred_vol:.3f} m³`
            """)
        with col_c2:
            if gt_box is not None:
                st.markdown("#### 🟢 Ground Truth Bounding Box")
                st.markdown(f"""
                - **Center $(c_x, c_y, c_z)$**: `{gt_box[0]:.3f}m, {gt_box[1]:.3f}m, {gt_box[2]:.3f}m`
                - **Dimensions $(d_x, d_y, d_z)$**: `{gt_box[3]:.3f}m, {gt_box[4]:.3f}m, {gt_box[5]:.3f}m`
                - **Calculated Volume**: `{gt_vol:.3f} m³`
                """)
            else:
                st.info("Ground truth is only available when selecting from the benchmark dataset.")

        if gt_box is not None:
            st.markdown("---")
            st.markdown("#### 🎯 Per-Axis Positional Deltas")
            dx_err = abs(pred_box[0] - gt_box[0])
            dy_err = abs(pred_box[1] - gt_box[1])
            dz_err = abs(pred_box[2] - gt_box[2])
            ecol1, ecol2, ecol3 = st.columns(3)
            ecol1.metric("Δ X Offset (Width)", f"{dx_err:.2f} m", delta="Aligned" if dx_err < 0.3 else "Shifted")
            ecol2.metric("Δ Y Offset (Depth)", f"{dy_err:.2f} m", delta="Aligned" if dy_err < 0.3 else "Shifted")
            ecol3.metric("Δ Z Offset (Height)", f"{dz_err:.2f} m", delta="Aligned" if dz_err < 0.3 else "Shifted")

    with tab_tokens:
        st.markdown("### 🔤 Tokenized Natural Language Query")
        tokens_list = tokenizer.tokenize(query_text)
        st.write(f"**Total Tokens:** `{len(tokens_list)}`")
        
        # Display as clean chips
        chip_html = " ".join([f"<span style='background:rgba(99,102,241,0.2); border:1px solid rgba(165,180,252,0.4); color:#C7D2FE; padding:4px 10px; border-radius:8px; margin:4px; display:inline-block; font-family:monospace;'>{tok}</span>" for tok in tokens_list])
        st.markdown(chip_html, unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        st.info("ℹ️ These subword tokens are projected through BERT into a 256-dimensional multimodal embedding space and cross-attended against the 64 geometric point cloud proposals.")

else:
    st.info("👈 **Get Started**: Click **'🚀 Run 3D Localization'** in the left sidebar or select one of the quick scenario buttons above!")
