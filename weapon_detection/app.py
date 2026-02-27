import io
import os
import glob
import time
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import yaml
from PIL import Image
from ultralytics import YOLO


# -----------------------------
# UI config / styling
# -----------------------------
st.set_page_config(page_title="Weapon Detection Dashboard", layout="wide")

st.markdown(
    """
    <style>
      .block-container { padding-top: 1.2rem; }
      .metric-card {
        padding: 14px 14px 6px 14px;
        border: 1px solid rgba(49,51,63,0.2);
        border-radius: 14px;
        background: rgba(255,255,255,0.02);
      }
      .small-note { opacity: 0.8; font-size: 0.9rem; }
      .warn-note { opacity: 0.9; font-size: 0.9rem; color: #b45309; }
    </style>
    """,
    unsafe_allow_html=True,
)

ROOT = Path(__file__).resolve().parent

# Your local training YAML location (adjust if needed)
DEFAULT_YAML = ROOT / "weapon_training" / "data.yaml"


# -----------------------------
# Helpers
# -----------------------------
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)
    return p


def find_weight_candidates():
    patterns = [
        ROOT / "weapon_training" / "runs" / "detect" / "*" / "weights" / "best.pt",
        ROOT / "weapon_training" / "runs" / "detect" / "*" / "weights" / "last.pt",
    ]
    paths = []
    for p in patterns:
        paths.extend(glob.glob(str(p)))
    paths.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    return paths


def unzip_to_temp(uploaded_zip) -> Path:
    tmp_root = ensure_dir(ROOT / ".streamlit_tmp")
    run_dir = ensure_dir(tmp_root / f"dataset_{int(time.time())}")
    with zipfile.ZipFile(uploaded_zip, "r") as z:
        z.extractall(run_dir)
    return run_dir


def guess_yaml_in_folder(folder: Path) -> Path | None:
    for name in ["data.yaml", "dataset.yaml", "coco.yaml"]:
        cand = folder / name
        if cand.exists():
            return cand

    for cand in folder.rglob("*.yaml"):
        try:
            txt = cand.read_text(encoding="utf-8", errors="ignore")
            if "train:" in txt and "test:" in txt:
                return cand
        except Exception:
            pass
    return None


def latest_val_dir() -> Path | None:
    base = ROOT / "runs" / "detect"
    if not base.exists():
        return None
    val_dirs = sorted(base.glob("val*"), key=lambda p: p.stat().st_mtime, reverse=True)
    return val_dirs[0] if val_dirs else None


def load_yaml(yaml_path: Path) -> dict | None:
    if not yaml_path.exists():
        return None
    with open(yaml_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def normalize_names(names_obj):
    # names can be list or dict
    if names_obj is None:
        return None
    if isinstance(names_obj, dict):
        items = []
        for k, v in names_obj.items():
            try:
                ki = int(k)
            except Exception:
                continue
            items.append((ki, str(v)))
        items.sort(key=lambda x: x[0])
        return [v for _, v in items]
    if isinstance(names_obj, (list, tuple)):
        return [str(x) for x in names_obj]
    return None


def load_names_from_yaml(yaml_path: Path):
    d = load_yaml(yaml_path)
    if not d:
        return None
    return normalize_names(d.get("names"))


def pil_to_bgr(pil_img: Image.Image) -> np.ndarray:
    rgb = np.array(pil_img.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def bgr_to_pil(bgr: np.ndarray) -> Image.Image:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def draw_red_circles(image_bgr: np.ndarray, boxes_xyxy: np.ndarray, confs: np.ndarray, clss: np.ndarray, names):
    out = image_bgr.copy()
    h, w = out.shape[:2]

    for (x1, y1, x2, y2), conf, c in zip(boxes_xyxy, confs, clss):
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        dx = (x2 - x1)
        dy = (y2 - y1)
        radius = max(8, int(0.5 * np.sqrt(dx * dx + dy * dy)))

        cv2.circle(out, (cx, cy), radius, (0, 0, 255), 3)

        label = str(int(c))
        if names and int(c) < len(names):
            label = names[int(c)]
        text = f"{label} {float(conf):.2f}"

        tx = min(max(0, cx - radius), w - 1)
        ty = max(0, cy - radius - 10)
        cv2.putText(out, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)

    return out


def safe_metric(v):
    try:
        if v is None:
            return None
        return float(v)
    except Exception:
        return None


def results_summary(results) -> dict:
    """
    Extract key metrics from Ultralytics Results.
    """
    out = {
        "map50": None,
        "map": None,
        "mp": None,
        "mr": None,
    }
    try:
        out["map50"] = safe_metric(getattr(results.box, "map50", None))
        out["map"] = safe_metric(getattr(results.box, "map", None))
        out["mp"] = safe_metric(getattr(results.box, "mp", None))
        out["mr"] = safe_metric(getattr(results.box, "mr", None))
    except Exception:
        pass
    return out


def per_class_table(results, names):
    """
    Build per-class table with precision/recall/AP50/AP.
    """
    # Ultralytics commonly exposes per-class arrays on results.box:
    # p, r, ap50, ap (may vary by version)
    rows = []
    try:
        p = getattr(results.box, "p", None)
        r = getattr(results.box, "r", None)
        ap50 = getattr(results.box, "ap50", None)
        ap = getattr(results.box, "ap", None)

        if p is None or r is None:
            return pd.DataFrame()

        # Convert to numpy
        p = np.array(p).reshape(-1)
        r = np.array(r).reshape(-1)
        ap50 = np.array(ap50).reshape(-1) if ap50 is not None else None
        ap = np.array(ap).reshape(-1) if ap is not None else None

        n = len(p)
        for i in range(n):
            cls_name = names[i] if (names and i < len(names)) else str(i)
            rows.append(
                {
                    "class_id": i,
                    "class_name": cls_name,
                    "precision": float(p[i]),
                    "recall": float(r[i]),
                    "AP50": float(ap50[i]) if ap50 is not None and i < len(ap50) else None,
                    "AP50-95": float(ap[i]) if ap is not None and i < len(ap) else None,
                }
            )
    except Exception:
        return pd.DataFrame()

    return pd.DataFrame(rows)


def warn_missing_dirs_for_test_eval(eval_yaml: Path):
    """
    Helpful UI warning if val folders are missing (Ultralytics often requires them).
    This does NOT modify files; only warns.
    """
    d = load_yaml(eval_yaml)
    if not d:
        return

    # Determine base directory for relative resolution: YAML file directory
    ydir = eval_yaml.parent

    # Use "train/val/test" paths from yaml. If there's a "path" key, include it.
    base_path = d.get("path", None)
    base_dir = (ydir / base_path) if base_path else ydir

    val_rel = d.get("val", None)
    if not val_rel:
        return

    # If val is like "images/val", check that directory exists
    val_dir = (base_dir / val_rel).resolve()
    if not val_dir.exists():
        st.markdown(
            f'<div class="warn-note">Dataset warning: expected val images folder not found: <code>{val_dir}</code>. '
            f'Ultralytics may fail even for test-only evaluation. Create empty <code>images/val</code> and '
            f'<code>labels/val</code> folders (or add them in your dataset).</div>',
            unsafe_allow_html=True,
        )


# -----------------------------
# Sidebar: model loading
# -----------------------------
st.sidebar.title("Model")

st.sidebar.write("ROOT:", str(ROOT))
st.sidebar.write("DEFAULT_YAML:", str(DEFAULT_YAML))
st.sidebar.write("YAML exists:", bool(DEFAULT_YAML.exists()))

weight_candidates = find_weight_candidates()

weight_choice = st.sidebar.selectbox(
    "Select weights",
    options=(weight_candidates if weight_candidates else ["(no weights found)"]),
    index=0,
)

uploaded_weight = st.sidebar.file_uploader("Or upload weights (.pt)", type=["pt"])

conf_thres = st.sidebar.slider("Confidence threshold", 0.05, 0.95, 0.25, 0.05)
iou_thres = st.sidebar.slider("IoU threshold", 0.10, 0.95, 0.70, 0.05)

names = load_names_from_yaml(DEFAULT_YAML) if DEFAULT_YAML.exists() else None

model = None
weights_path = None

try:
    if uploaded_weight is not None:
        tmp_root = ensure_dir(ROOT / ".streamlit_tmp")
        weights_path = tmp_root / f"uploaded_{int(time.time())}.pt"
        weights_path.write_bytes(uploaded_weight.read())
        model = YOLO(str(weights_path))
    else:
        if weight_candidates and "(no weights found)" not in weight_choice:
            weights_path = Path(weight_choice)
            model = YOLO(str(weights_path))
except Exception as e:
    st.sidebar.error(f"Failed to load model: {e}")

if model is None:
    st.warning(
        "No model loaded. Put your `best.pt` under "
        "`weapon_training/runs/detect/.../weights/` or upload it in the sidebar."
    )
    st.stop()

st.sidebar.markdown("---")
st.sidebar.markdown(f"**YAML:** `{str(DEFAULT_YAML) if DEFAULT_YAML.exists() else 'not found'}`")
st.sidebar.markdown(f"**Weights:** `{weights_path.name if weights_path else 'uploaded'}`")
if names:
    st.sidebar.markdown(f"**Classes:** {', '.join(names)}")
else:
    st.sidebar.markdown("**Classes:** (not found in YAML)")


# -----------------------------
# Main layout
# -----------------------------
st.title("Weapon Detection Dashboard")
st.caption("Test evaluation + inference + KPI aggregation.")

tab_eval, tab_infer, tab_kpi = st.tabs(["1) Test Evaluation", "2) Inference", "3) KPI"])


# -----------------------------
# 1) Evaluation tab
# -----------------------------
with tab_eval:
    st.subheader("Test Dataset Evaluation (no training validation required)")

    colA, colB = st.columns([1.25, 1])

    with colA:
        st.markdown(
            """
            **Option A:** upload a ZIP that contains a YOLO dataset layout and a `data.yaml`.
            Example ZIP structure:
            - `data.yaml`
            - `images/test/...`
            - `labels/test/...`
            """
        )
        dataset_zip = st.file_uploader("Upload test dataset ZIP", type=["zip"], key="zip_eval")

        st.markdown("**Option B:** use local repo YAML (your project `weapon_training/data.yaml`).")
        use_local_yaml = st.checkbox(
            "Use local repo data.yaml instead",
            value=(dataset_zip is None and DEFAULT_YAML.exists()),
        )

        st.markdown("**Option C:** upload a YAML manually (if paths differ).")
        yaml_upload = st.file_uploader("Upload data.yaml (optional)", type=["yaml", "yml"], key="yaml_eval")

        eval_yaml = None
        extracted_dir = None

        # Priority: uploaded yaml > local yaml > zip yaml
        if yaml_upload is not None:
            tmp_root = ensure_dir(ROOT / ".streamlit_tmp")
            eval_yaml = tmp_root / f"uploaded_data_{int(time.time())}.yaml"
            eval_yaml.write_bytes(yaml_upload.read())
            st.info(f"Using uploaded YAML: {eval_yaml}")

        elif use_local_yaml and DEFAULT_YAML.exists():
            eval_yaml = DEFAULT_YAML
            st.info(f"Using local YAML: {eval_yaml}")

        elif dataset_zip is not None:
            extracted_dir = unzip_to_temp(dataset_zip)
            guessed = guess_yaml_in_folder(extracted_dir)
            if guessed is None:
                st.error("Could not find a usable YAML inside the ZIP. Include `data.yaml` in the ZIP.")
            else:
                eval_yaml = guessed
                st.info(f"Using YAML found in ZIP: {eval_yaml.relative_to(extracted_dir)}")

        if eval_yaml is not None:
            warn_missing_dirs_for_test_eval(eval_yaml)

        run_eval = st.button("Run TEST evaluation (metrics + confusion matrix)", type="primary", disabled=(eval_yaml is None))

    with colB:
        st.markdown("**Outputs:** confusion matrix + PR/F1/P/R curves + per-class table.")
        st.markdown(
            '<div class="small-note">'
            'Note: Ultralytics computes metrics via <b>model.val()</b>. Here it is executed on the <b>TEST split</b>. '
            'Ultralytics may still require empty <code>images/val</code> and <code>labels/val</code> folders to exist.'
            "</div>",
            unsafe_allow_html=True,
        )

    if run_eval and eval_yaml is not None:
        with st.spinner("Running evaluation on TEST dataset..."):
            try:
                results = model.val(
                    data=str(eval_yaml),
                    split="test",  # evaluate on TEST split
                    conf=conf_thres,
                    iou=iou_thres,
                )
                st.success("Test evaluation finished.")
                st.session_state["eval_results"] = results_summary(results)
                st.session_state["eval_per_class"] = per_class_table(results, names)
            except Exception as e:
                st.error(f"Evaluation failed: {e}")
                st.session_state["eval_results"] = None
                st.session_state["eval_per_class"] = pd.DataFrame()

        # Metrics cards
        s = st.session_state.get("eval_results") or {}
        mcol1, mcol2, mcol3, mcol4 = st.columns(4)
        with mcol1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("mAP50", f"{(s.get('map50') or float('nan')):.3f}")
            st.markdown("</div>", unsafe_allow_html=True)
        with mcol2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("mAP50-95", f"{(s.get('map') or float('nan')):.3f}")
            st.markdown("</div>", unsafe_allow_html=True)
        with mcol3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Precision", f"{(s.get('mp') or float('nan')):.3f}")
            st.markdown("</div>", unsafe_allow_html=True)
        with mcol4:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Recall", f"{(s.get('mr') or float('nan')):.3f}")
            st.markdown("</div>", unsafe_allow_html=True)

        # Per-class table
        pc = st.session_state.get("eval_per_class", pd.DataFrame())
        st.markdown("### Per-class performance (TEST)")
        if pc is not None and not pc.empty:
            st.dataframe(
                pc[["class_id", "class_name", "precision", "recall", "AP50", "AP50-95"]],
                use_container_width=True,
            )
        else:
            st.info("Per-class metrics not available from this Ultralytics version/run.")

        # Visuals from runs/detect/val*
        vdir = latest_val_dir()
        if vdir:
            cm = vdir / "confusion_matrix.png"
            pr = vdir / "PR_curve.png"
            f1 = vdir / "F1_curve.png"
            p_curve = vdir / "P_curve.png"
            r_curve = vdir / "R_curve.png"

            grid = [p for p in [cm, pr, f1, p_curve, r_curve] if p.exists()]

            st.markdown("### Visuals")
            if grid:
                cols = st.columns(2)
                for i, img_path in enumerate(grid):
                    with cols[i % 2]:
                        st.image(str(img_path), caption=img_path.name, use_container_width=True)

                # Store for KPI tab reuse
                st.session_state["eval_vdir"] = str(vdir)
            else:
                st.info("No plots found in the latest evaluation folder.")
        else:
            st.info("No evaluation folder found yet under runs/detect/.")


# -----------------------------
# 2) Inference tab
# -----------------------------
with tab_infer:
    st.subheader("Inference on user images (red-circle highlighting)")

    uploaded_imgs = st.file_uploader(
        "Upload one or more images",
        type=["jpg", "jpeg", "png", "webp"],
        accept_multiple_files=True,
        key="infer_imgs",
    )

    if uploaded_imgs:
        all_det_rows = []      # per detection rows
        per_image_rows = []    # per image summary rows

        for up in uploaded_imgs:
            pil_img = Image.open(io.BytesIO(up.read()))
            bgr = pil_to_bgr(pil_img)

            preds = model.predict(
                source=bgr,
                conf=conf_thres,
                iou=iou_thres,
                verbose=False,
            )[0]

            if preds.boxes is None or len(preds.boxes) == 0:
                out_pil = bgr_to_pil(bgr)
                st.image(out_pil, caption=f"{up.name} — no detections", use_container_width=True)
                per_image_rows.append({"image": up.name, "detections": 0, "avg_conf": None})
                continue

            xyxy = preds.boxes.xyxy.cpu().numpy()
            confs = preds.boxes.conf.cpu().numpy()
            clss = preds.boxes.cls.cpu().numpy()

            circled = draw_red_circles(bgr, xyxy, confs, clss, names)
            out_pil = bgr_to_pil(circled)

            det_count = len(xyxy)
            avg_conf = float(np.mean(confs)) if len(confs) else None
            per_image_rows.append({"image": up.name, "detections": int(det_count), "avg_conf": avg_conf})

            st.image(out_pil, caption=f"{up.name} — detections: {det_count}", use_container_width=True)

            for (x1, y1, x2, y2), conf, c in zip(xyxy, confs, clss):
                cls_id = int(c)
                cls_name = names[cls_id] if (names and cls_id < len(names)) else str(cls_id)
                all_det_rows.append(
                    {
                        "image": up.name,
                        "class_id": cls_id,
                        "class_name": cls_name,
                        "conf": float(conf),
                        "x1": float(x1),
                        "y1": float(y1),
                        "x2": float(x2),
                        "y2": float(y2),
                    }
                )

        # Save to session for KPI tab
        st.session_state["kpi_detections"] = all_det_rows
        st.session_state["kpi_images"] = per_image_rows

        st.success("Inference complete. Open the KPI tab for aggregated stats.")


# -----------------------------
# 3) KPI tab
# -----------------------------
with tab_kpi:
    st.subheader("KPI Dashboard")

    # --- Evaluation section (repeated here) ---
    st.markdown("## Model test evaluation (from Evaluation tab)")
    s = st.session_state.get("eval_results")
    pc = st.session_state.get("eval_per_class", pd.DataFrame())
    vdir_str = st.session_state.get("eval_vdir")

    if s:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("mAP50 (TEST)", f"{(s.get('map50') or float('nan')):.3f}")
        c2.metric("mAP50-95 (TEST)", f"{(s.get('map') or float('nan')):.3f}")
        c3.metric("Precision (TEST)", f"{(s.get('mp') or float('nan')):.3f}")
        c4.metric("Recall (TEST)", f"{(s.get('mr') or float('nan')):.3f}")

        if pc is not None and not pc.empty:
            st.markdown("### Per-class table (TEST)")
            st.dataframe(pc, use_container_width=True)
    else:
        st.info("Run the Evaluation tab to generate test metrics, confusion matrix, and curves.")

    if vdir_str:
        vdir = Path(vdir_str)
        cm = vdir / "confusion_matrix.png"
        pr = vdir / "PR_curve.png"
        f1 = vdir / "F1_curve.png"
        p_curve = vdir / "P_curve.png"
        r_curve = vdir / "R_curve.png"
        grid = [p for p in [cm, pr, f1, p_curve, r_curve] if p.exists()]

        if grid:
            st.markdown("### Evaluation visuals")
            cols = st.columns(2)
            for i, img_path in enumerate(grid):
                with cols[i % 2]:
                    st.image(str(img_path), caption=img_path.name, use_container_width=True)

    st.markdown("---")

    # --- Inference KPI section ---
    st.markdown("## User uploads KPI (inference)")

    det_rows = st.session_state.get("kpi_detections", [])
    img_rows = st.session_state.get("kpi_images", [])

    if not img_rows:
        st.info("Run Inference tab first to populate KPI for user-uploaded images.")
    else:
        img_df = pd.DataFrame(img_rows)
        det_df = pd.DataFrame(det_rows) if det_rows else pd.DataFrame()

        total_images = int(img_df["image"].nunique())
        images_with_det = int((img_df["detections"] > 0).sum())
        total_det = int(det_df.shape[0]) if not det_df.empty else 0
        det_rate = (images_with_det / total_images) if total_images else 0.0

        avg_conf_global = None
        if not det_df.empty and "conf" in det_df.columns:
            avg_conf_global = float(det_df["conf"].mean())

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Total images", total_images)
        k2.metric("Images with detections", images_with_det)
        k3.metric("Total detections", total_det)
        k4.metric("Detection rate", f"{det_rate:.1%}")

        if avg_conf_global is not None:
            st.metric("Average confidence (all detections)", f"{avg_conf_global:.3f}")

        st.markdown("### Per-image summary")
        st.dataframe(img_df.sort_values(["detections", "image"], ascending=[False, True]), use_container_width=True)

        if not det_df.empty:
            st.markdown("### Detections by class (count + avg confidence)")
            by_class = (
                det_df.groupby("class_name")
                .agg(detections=("class_name", "count"), avg_conf=("conf", "mean"))
                .sort_values("detections", ascending=False)
                .reset_index()
            )
            st.dataframe(by_class, use_container_width=True)

            # Simple visuals (Streamlit built-in)
            st.markdown("### Class distribution (user uploads)")
            chart_df = by_class.set_index("class_name")[["detections"]]
            st.bar_chart(chart_df)

            st.markdown("### Raw detection table")
            st.dataframe(det_df.sort_values(["image", "conf"], ascending=[True, False]), use_container_width=True)