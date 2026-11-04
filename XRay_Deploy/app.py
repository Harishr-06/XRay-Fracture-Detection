import os
import cv2
import numpy as np
import torch
import streamlit as st
from PIL import Image
from ultralytics import YOLO
from ultralytics.data.augment import LetterBox

# ============================================================
# PHASE 1 - X-RAY FRACTURE DETECTION + SEGMENTATION + EIGEN-CAM
# ============================================================
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "best.pt"

IMG_SIZE = 640
DEVICE = 0 if torch.cuda.is_available() else "cpu"


@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found:\n{MODEL_PATH}\n\n"
            "Change MODEL_PATH at the top of this file if your best.pt is elsewhere."
        )

    yolo = YOLO(MODEL_PATH)
    yolo.model.to(DEVICE)
    yolo.model.eval()
    return yolo


def prepare_tensor(image_rgb):
    """YOLO-style preprocessing -> BCHW float tensor."""
    image = np.asarray(image_rgb)
    lb = LetterBox(new_shape=(IMG_SIZE, IMG_SIZE), auto=False, stride=32)
    padded = lb(image=image)

    tensor = padded.transpose(2, 0, 1)
    tensor = np.ascontiguousarray(tensor)
    tensor = torch.from_numpy(tensor).float() / 255.0
    return tensor.unsqueeze(0).to(DEVICE)


def find_target_layer(yolo_model):
    """
    Pick a segmentation-relevant convolutional layer to explain.

    Preference order:
      1. The mask-prototype module (`proto`) inside the Segment head. This
         is the actual conv branch that produces your segmentation masks,
         so its activations are the most direct "segmentation-focused"
         explanation target.
      2. SPPF (a late, high-level backbone feature layer) as a fallback if
         `proto` isn't found for some reason (e.g. a differently-built head).
    """
    seg_head = yolo_model.model[-1]
    proto = getattr(seg_head, "proto", None)
    if proto is not None:
        return proto, "segmentation prototype layer (proto)"
    return yolo_model.model[9], "SPPF (fallback, backbone feature layer)"


class EigenCAMYOLOv8Seg:
    """
    Activation-based CAM (Eigen-CAM style) for YOLOv8-Seg.

    Needs no backward pass and no training-mode forward pass. Hooks a
    segmentation-relevant conv layer, captures its activation during a
    single plain no-grad forward pass, and computes the dominant
    activation pattern across channels via SVD. That pattern is the CAM.

    Trade-off: this is not class-discriminative the way true Grad-CAM is.
    It shows where the hooked feature layer activates most strongly, not
    "what pushed the fracture-class score up specifically."
    """

    def __init__(self, raw_model, target_layer):
        self.raw_model = raw_model
        self.target_layer = target_layer
        self.activations = None
        self.hook_handle = target_layer.register_forward_hook(self._save_activation)

    def _save_activation(self, module, inputs, output):
        if torch.is_tensor(output) and output.ndim == 4:
            self.activations = output.detach()
        elif isinstance(output, (tuple, list)):
            for item in output:
                if torch.is_tensor(item) and item.ndim == 4:
                    self.activations = item.detach()
                    return

    def remove(self):
        self.hook_handle.remove()

    def generate(self, x):
        self.activations = None

        with torch.inference_mode():
            self.raw_model(x)

        if self.activations is None:
            raise RuntimeError(
                "No activation was captured from the target layer. "
                "The hooked module did not run during this forward pass."
            )

        # activations: [1, C, H, W]
        acts = self.activations[0].float().cpu().numpy()  # [C, H, W]
        c, h, w = acts.shape
        flat = acts.reshape(c, h * w)  # [C, H*W]

        # Principal activation pattern across channels (Eigen-CAM).
        flat_centered = flat - flat.mean(axis=0, keepdims=True)
        try:
            _, _, vt = np.linalg.svd(flat_centered, full_matrices=False)
            principal = vt[0]  # [H*W]
        except np.linalg.LinAlgError:
            principal = flat.mean(axis=0)  # fallback: mean activation

        cam = principal.reshape(h, w)

        # SVD's principal component has an arbitrary sign (numpy gives no
        # guarantee it points toward "high activation"). The sign carries
        # no meaning for Eigen-CAM, only magnitude does, so use absolute
        # value rather than a ReLU-style clamp (which previously zeroed
        # out most of the map whenever the sign came out negative).
        cam = np.abs(cam)

        # A small region of relatively higher values can otherwise set
        # the ceiling for a plain min-max normalization, compressing the
        # rest of the map's real (but smaller-magnitude) variation down
        # near zero - which JET renders as near-black instead of a
        # visible heatmap. Clip to a percentile range first so the
        # ceiling reflects the bulk of the map's variation, not just its
        # most extreme pixels.
        low, high = np.percentile(cam, [1, 99])
        cam = np.clip(cam, low, high)

        cam -= cam.min()
        max_value = cam.max()
        if max_value > 1e-8:
            cam /= max_value

        # Even after clipping, this signal tends to sit mostly at low,
        # gently-varying magnitude with only a mild peak near the real
        # region of interest - a linear 0-1 map still renders as mostly
        # dark under JET (dark blue/black dominates its low end). Gamma
        # < 1 pushes low/mid values up without moving 0 or 1, so the
        # real (but modest) variation actually becomes visible as color
        # instead of reading as uniform near-black.
        cam = cam ** 0.4

        return cam


def resize_cam_to_image(cam, image_rgb):
    h, w = image_rgb.shape[:2]
    return cv2.resize(cam, (w, h), interpolation=cv2.INTER_LINEAR)


def overlay_cam(image_rgb, cam, alpha=0.45):
    cam_uint8 = np.uint8(np.clip(cam, 0, 1) * 255)
    heatmap_bgr = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(image_rgb, 1 - alpha, heatmap_rgb, alpha, 0)


def segmentation_overlay(result):
    plotted_bgr = result.plot(
        conf=True,
        labels=True,
        boxes=True,
        masks=True,
    )
    return cv2.cvtColor(plotted_bgr, cv2.COLOR_BGR2RGB)


# --------------------------- UI ----------------------------

st.set_page_config(
    page_title="X-Ray Fracture AI",
    page_icon="🦴",
    layout="wide",
)

st.title("🦴 A Deeplearning Framework for Detecting Subtle Hair-line Fracture from Scanned Medical Images ")
st.subheader(" X-Ray Module")

st.write(
    "Upload a new X-ray image. The system performs fracture detection, "
    "segmentation, confidence estimation and CAM-based explainability."
)

try:
    yolo = load_model()
except Exception as e:
    st.error(str(e))
    st.stop()

uploaded = st.file_uploader(
    "Upload X-Ray Image",
    type=["jpg", "jpeg", "png", "bmp"],
)

if uploaded is not None:
    pil_image = Image.open(uploaded).convert("RGB")
    image_rgb = np.array(pil_image)

    # ---------------- Normal YOLO prediction (unchanged) ----------------
    with st.spinner("Running YOLOv8m-Seg prediction..."):
        result = yolo.predict(
            source=image_rgb,
            imgsz=IMG_SIZE,
            conf=0.25,
            device=DEVICE,
            verbose=False,
        )[0]

    num_detections = 0 if result.boxes is None else len(result.boxes)

    if num_detections > 0:
        confs = result.boxes.conf.detach().cpu().numpy()
        best_conf = float(confs.max())
        prediction_text = "FRACTURE DETECTED"
    else:
        best_conf = 0.0
        prediction_text = "NO FRACTURE DETECTED"

    # ---------------- Eigen-CAM (separate, read-only pass) ----------------
    target_layer, target_layer_desc = find_target_layer(yolo.model)

    with st.spinner("Generating CAM explanation..."):
        cam_engine = EigenCAMYOLOv8Seg(yolo.model, target_layer)

        try:
            x = prepare_tensor(image_rgb)
            cam_small = cam_engine.generate(x)
            cam = resize_cam_to_image(cam_small, image_rgb)
            cam_overlay = overlay_cam(image_rgb, cam)
        finally:
            cam_engine.remove()

    # ---------------- Results ----------------
    st.markdown("---")
    st.header("Prediction")

    c1, c2, c3 = st.columns(3)

    with c1:
        if num_detections > 0:
            st.success(prediction_text)
        else:
            st.info(prediction_text)

    with c2:
        st.metric("Confidence", f"{best_conf * 100:.2f}%")

    with c3:
        st.metric("Detected Regions", num_detections)

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original X-Ray")
        st.image(image_rgb, width='stretch')

    with col2:
        st.subheader("Detection + Segmentation")
        st.image(
            segmentation_overlay(result),
            width='stretch',
        )

    col3, col4 = st.columns(2)

    with col3:
        st.subheader("CAM Heatmap")
        st.image(cam_overlay, width='stretch')
        st.caption(f"Explained layer: {target_layer_desc}")

    with col4:
        st.subheader("CAM Only")
        cam_uint8 = np.uint8(np.clip(cam, 0, 1) * 255)
        st.image(cam_uint8, width='stretch', clamp=True)

    st.markdown("---")
    st.header("AI Explanation")

    if num_detections > 0:
        st.write(
            f"The YOLOv8m-Seg model detected a suspected fracture with "
            f"a confidence of **{best_conf * 100:.2f}%**. "
            "The segmentation output highlights the predicted fracture region. "
            "The CAM heatmap highlights the image regions that most strongly "
            "activate the model's segmentation feature layer, giving a visual "
            "sense of where the network is 'looking' when it forms the mask."
        )
    else:
        st.write(
            "No fracture region was detected above the selected confidence "
            "threshold. The CAM heatmap is still displayed to visualize which "
            "regions most strongly activate the model's segmentation feature layer."
        )

    st.caption(
        "Research/demo system. The prediction is an AI model output and is not "
        "a clinical diagnosis. The CAM heatmap reflects feature-layer activation "
        "strength, not a class-discriminative gradient."
    )
