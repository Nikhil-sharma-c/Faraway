"""
phone_detect.py — High-precision, real-time multi-device & prohibited object detection.

Strict Model Class Mapping:
- COCO 67 ('cell phone') -> PHONE DETECTED
- COCO 63 ('laptop')     -> LAPTOP DETECTED
- COCO 74 ('clock')      -> SMARTWATCH DETECTED
- COCO 73 ('book')       -> UNAUTHORIZED NOTES DETECTED
- COCO 62 ('tv')         -> TABLET / SCREEN DETECTED
- COCO 65 ('remote')     -> PROHIBITED DEVICE DETECTED

NEVER falls back or reclassifies unknown/generic objects as 'phone'.
"""

import os
import numpy as np
import cv2
from ultralytics import YOLO

BASE = os.path.dirname(os.path.abspath(__file__))

# Strict COCO Class IDs
CLASS_PERSON = 0
CLASS_TV = 62          # Tablet / Secondary Screen
CLASS_LAPTOP = 63      # Laptop computer
CLASS_REMOTE = 65      # Remote control / Electronic peripheral
CLASS_PHONE = 67       # Cell phone / Mobile device
CLASS_BOOK = 73        # Book / Printed notes
CLASS_CLOCK = 74       # Clock / Watch / Smartwatch

TARGET_CLASSES = [CLASS_PHONE, CLASS_LAPTOP, CLASS_CLOCK, CLASS_BOOK, CLASS_TV, CLASS_REMOTE]

# Object-specific confidence thresholds
CONF_THRESHOLDS = {
    CLASS_PHONE: 0.25,    # High recall for partial and occluded mobile devices
    CLASS_LAPTOP: 0.35,   # High accuracy for laptops
    CLASS_CLOCK: 0.28,    # Smartwatch / Wristwatch
    CLASS_BOOK: 0.35,     # Study materials / Printed notes
    CLASS_TV: 0.35,       # Tablets / Screens
    CLASS_REMOTE: 0.30,   # Electronic devices
}

DEFAULT_CONF = 0.25
ROI_IMGSZ = 640              # High-resolution magnification for person/hand crop

# Cap this detector's ONNX Runtime thread pool. Uncapped, ORT grabs EVERY core,
# and when its inference overlapped the torch-based face model (which also grabs
# every core) the box oversubscribed and every worker stalled for seconds -- the
# detection freeze and "no signal" blackouts. Kept small (~3): the confirm pass
# only needs ~1 FPS (the per-frame fast pass handles instant phone response), so
# a tight cap leaves the bulk of the cores free for the face/tracking/stream
# threads and avoids the periodic multi-second spike when the two overlap.
_ORT_THREADS = max(2, min(3, (os.cpu_count() or 4) // 4)) if (os.cpu_count() or 4) >= 8 else 2

# Minimal COCO name map for the classes we care about (used on the raw-ONNX
# path, which does not carry ultralytics' full names dict).
_COCO_NAMES = {62: "tv", 63: "laptop", 65: "remote", 67: "cell phone",
               73: "book", 74: "clock"}

# --- Cost controls (see PhoneDetector.detect) ------------------------------
# Per-person ROI passes were the reason a detection cycle cost 1+N inferences
# (~1-2s with a few people). ai-exam-proctor runs exactly ONE inference per pass
# and stays responsive; "roundrobin" keeps that bound at 2 while retaining ROI
# magnification for distant/small phones. Set PHONE_ROI_MODE=none|roundrobin|all.
# Default "none" = exactly ai-exam-proctor's design (one inference per pass).
# Measured on a 3-person 1376x768 frame with yolo26s: all=552ms, roundrobin=306ms,
# none=118ms. Set PHONE_ROI_MODE=roundrobin if distant/small phones need the
# extra magnification and you can afford ~2x the cycle time.
DEFAULT_ROI_MODE = os.environ.get("PHONE_ROI_MODE", "none").lower()
# Stretch the whole frame to a square model input instead of letterboxing,
# matching ai-exam-proctor's drawImage(video,0,0,640,640).
SQUARE_STRETCH = os.environ.get("PHONE_SQUARE_STRETCH", "1").lower() not in ("0", "off", "false")
DEFAULT_WHOLE_IMGSZ = 640    # High-resolution whole-frame scanning

# Asymmetric person ROI padding:
# Expands generously downwards and sideways to enclose candidate hands, desk, and lap.
ROI_PAD_X = 0.35
ROI_PAD_Y_TOP = 0.20
ROI_PAD_Y_BOT = 0.45

# Plausibility limits
MIN_SIDE_PX = 8                  # minimum side length in pixels
MAX_AREA_FRAC_OF_PERSON = 0.85   # allows large laptops / tablets relative to person box
MIN_ASPECT = 0.15                # w/h; allows vertical, horizontal, and partial slivers
MAX_ASPECT = 6.0


def _iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / ua if ua > 0 else 0.0


def plausible(box, cls_id, person_box=None):
    """Geometric plausibility check on a candidate object bounding box."""
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    if w < MIN_SIDE_PX or h < MIN_SIDE_PX:
        return False, "too small"
    ar = w / max(h, 1e-6)
    if not (MIN_ASPECT <= ar <= MAX_ASPECT):
        return False, f"implausible aspect {ar:.2f}"
    if person_box is not None and cls_id in (CLASS_PHONE, CLASS_CLOCK, CLASS_REMOTE):
        px1, py1, px2, py2 = person_box
        parea = max((px2 - px1) * (py2 - py1), 1.0)
        if (w * h) / parea > 0.40:
            return False, "too large relative to person"
    return True, "ok"


class PhoneDetector:
    """Multi-device and prohibited object detector for real-time and replay CCTV."""

    def __init__(self, weights="yolo11s.pt", conf=DEFAULT_CONF):
        path = os.path.join(BASE, weights)
        if not os.path.exists(path):
            fallback = os.path.join(BASE, "yolo11n.pt")
            path = fallback if os.path.exists(fallback) else weights
        self.conf = conf
        self.weights = weights
        self.is_onnx = str(path).lower().endswith(".onnx")

        if self.is_onnx:
            # Raw, THREAD-CAPPED ONNX Runtime session. Using ultralytics here
            # would create an uncapped ORT session (all cores) -- the root cause
            # of the oversubscription freeze. yolo26s is exported end2end, so the
            # output is already-decoded [1, N, 6] = x1,y1,x2,y2,conf,cls.
            import onnxruntime as ort
            so = ort.SessionOptions()
            so.intra_op_num_threads = _ORT_THREADS
            so.inter_op_num_threads = 1
            so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            self.sess = ort.InferenceSession(path, sess_options=so,
                                             providers=["CPUExecutionProvider"])
            self.iname = self.sess.get_inputs()[0].name
            ishape = self.sess.get_inputs()[0].shape
            self.insz = ishape[2] if isinstance(ishape[2], int) else 640
            self.model = None
            self.names = _COCO_NAMES
            print(f"[PHONE] yolo26s ONNX on capped ORT ({_ORT_THREADS} threads, "
                  f"{self.insz}px end2end)")
        else:
            self.model = YOLO(path)
            self.sess = None
            self.names = self.model.names if hasattr(self.model, 'names') else {}

    def _detect_onnx(self, img):
        """Thread-capped end2end ONNX inference. Resizes the given image to the
        model's square input (a stretch, matching ai-exam-proctor) and maps the
        returned boxes back to the input image's own coordinates."""
        out = []
        try:
            H, W = img.shape[:2]
            if H == 0 or W == 0:
                return out
            blob = cv2.resize(img, (self.insz, self.insz), interpolation=cv2.INTER_LINEAR)
            blob = cv2.cvtColor(blob, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            blob = np.ascontiguousarray(np.transpose(blob, (2, 0, 1))[None])
            pred = self.sess.run(None, {self.iname: blob})[0]
            rows = pred[0] if pred.ndim == 3 else pred
            if rows.ndim != 2 or rows.shape[1] < 6:
                return out
            sx, sy = W / float(self.insz), H / float(self.insz)
            for row in rows:
                conf_val = float(row[4])
                cls_id = int(round(float(row[5])))
                if cls_id not in TARGET_CLASSES:
                    continue
                if conf_val < CONF_THRESHOLDS.get(cls_id, self.conf):
                    continue
                out.append([float(row[0]) * sx, float(row[1]) * sy,
                            float(row[2]) * sx, float(row[3]) * sy,
                            conf_val, cls_id, self.names.get(cls_id, str(cls_id))])
        except Exception as err:
            print(f"[DEVICE_DETECT] ONNX inference error: {err}")
        return out

    def _detect_objects(self, img, imgsz):
        if self.is_onnx:
            return self._detect_onnx(img)
        out = []
        try:
            for r in self.model(img, stream=True, verbose=False, imgsz=imgsz,
                                conf=self.conf, classes=TARGET_CLASSES):
                for b in r.boxes:
                    cls_id = int(b.cls[0])
                    conf_val = float(b.conf[0])
                    min_conf = CONF_THRESHOLDS.get(cls_id, self.conf)
                    if conf_val < min_conf:
                        continue
                    x1, y1, x2, y2 = map(float, b.xyxy[0])
                    cls_name = self.names.get(cls_id, str(cls_id))
                    out.append([x1, y1, x2, y2, conf_val, cls_id, cls_name])
        except Exception as err:
            print(f"[DEVICE_DETECT] Inference error: {err}")
        return out

    def detect(self, frame, person_boxes=None, whole_frame=True,
               whole_imgsz=DEFAULT_WHOLE_IMGSZ, ear_regions=None,
               roi_mode=None, roi_index=0):
        """Returns a list of dicts: {bbox:(x1,y1,x2,y2), conf, class_id, class_name, device_type, label, source}.

        Strictly maps returned model classes without defaulting non-phone objects to phone.

        roi_mode controls the per-person magnification passes, which are the
        dominant cost (each is a full extra inference):
          "none"       -- whole frame only: ONE inference per call. This is what
                          the ai-exam-proctor project does, and why it stays fast.
          "roundrobin" -- whole frame + exactly ONE person ROI per call, cycling
                          through people via roi_index. Bounded 2 inferences,
                          still gives distant/small-phone magnification over a
                          few passes. (default)
          "all"        -- legacy: whole frame + every person ROI = 1+N
                          inferences. Accurate but ~(1+N)x slower.
        """
        if frame is None or frame.size == 0:
            return []

        if roi_mode is None:
            roi_mode = DEFAULT_ROI_MODE

        H, W = frame.shape[:2]
        cands = []

        # 1. Whole-frame scan -- ONE inference.
        # Preprocessing mirrors ai-exam-proctor's dispatch, which draws the video
        # straight into a 640x640 canvas (drawImage(video,0,0,640,640)): a STRETCH
        # to square, not a letterbox. For a wide frame that matters -- letterboxing
        # 1376x768 into 640 leaves the content only ~357px tall (the rest is grey
        # padding), while the stretch uses the full 640 rows, putting ~1.8x more
        # pixels on a small phone. Boxes are mapped back to frame coordinates.
        if whole_frame:
            # The ONNX path already square-resizes inside _detect_onnx and maps
            # boxes back, so feed it the raw frame (double-resizing would just
            # lose precision). Only the ultralytics .pt path needs the manual
            # square stretch here.
            if SQUARE_STRETCH and not self.is_onnx:
                square = cv2.resize(frame, (whole_imgsz, whole_imgsz),
                                    interpolation=cv2.INTER_LINEAR)
                sx = W / float(whole_imgsz)
                sy = H / float(whole_imgsz)
                for b in self._detect_objects(square, whole_imgsz):
                    cands.append(([b[0] * sx, b[1] * sy, b[2] * sx, b[3] * sy,
                                   b[4], b[5], b[6]], None, "frame"))
            else:
                for b in self._detect_objects(frame, whole_imgsz):
                    cands.append((b, None, "frame"))

        # 2. Magnified Person-ROI scanning (hands, lap, desk region)
        rois = list(person_boxes or [])
        if roi_mode == "none":
            rois = []
        elif roi_mode == "roundrobin" and rois:
            rois = [rois[roi_index % len(rois)]]

        for pb in rois:
            px1, py1, px2, py2 = [int(v) for v in pb]
            pw, ph = px2 - px1, py2 - py1
            if pw < 24 or ph < 24:
                continue

            ex = int(pw * ROI_PAD_X)
            ey_top = int(ph * ROI_PAD_Y_TOP)
            ey_bot = int(ph * ROI_PAD_Y_BOT)

            cx1, cy1 = max(0, px1 - ex), max(0, py1 - ey_top)
            cx2, cy2 = min(W, px2 + ex), min(H, py2 + ey_bot)
            crop = frame[cy1:cy2, cx1:cx2]
            if crop.size == 0:
                continue

            for b in self._detect_objects(crop, ROI_IMGSZ):
                cands.append(([b[0] + cx1, b[1] + cy1, b[2] + cx1, b[3] + cy1,
                               b[4], b[5], b[6]], (px1, py1, px2, py2), "roi"))

        # 3. Geometric plausibility validation
        kept = []
        for b_data, pbox, src in cands:
            box = b_data[:4]
            cls_id = int(b_data[5])
            ok, _why = plausible(box, cls_id, pbox)
            if ok:
                kept.append((b_data, src))

        # 4. NMS & IoU deduplication across all detected objects
        kept.sort(key=lambda t: -t[0][4])
        final = []
        for b_data, src in kept:
            box = b_data[:4]
            conf = float(b_data[4])
            cls_id = int(b_data[5])
            cls_name = str(b_data[6])

            if all(_iou(box, f["bbox"]) < 0.40 for f in final):
                # Strict class identification
                if cls_id == CLASS_PHONE or cls_name == "cell phone":
                    dev_type = "phone"
                    label = "PHONE DETECTED"
                elif cls_id == CLASS_CLOCK or cls_name == "clock":
                    dev_type = "smartwatch"
                    label = "SMARTWATCH DETECTED"
                elif cls_id == CLASS_LAPTOP or cls_name == "laptop":
                    dev_type = "laptop"
                    label = "LAPTOP DETECTED"
                elif cls_id == CLASS_BOOK or cls_name == "book":
                    dev_type = "book"
                    label = "UNAUTHORIZED NOTES DETECTED"
                elif cls_id == CLASS_TV or cls_name == "tv":
                    dev_type = "tablet"
                    label = "TABLET / SCREEN DETECTED"
                elif cls_id == CLASS_REMOTE or cls_name == "remote":
                    dev_type = "device"
                    label = "PROHIBITED DEVICE DETECTED"
                else:
                    dev_type = "device"
                    label = f"{cls_name.upper()} DETECTED"

                final.append({
                    "bbox": (int(box[0]), int(box[1]), int(box[2]), int(box[3])),
                    "conf": conf,
                    "class_id": cls_id,
                    "class_name": cls_name,
                    "device_type": dev_type,
                    "label": f"{label} · {int(conf * 100)}%",
                    "source": src
                })

        return final


def persons_from_yolo_result(boxes, conf_min=0.35):
    """Extracts person boxes from an ultralytics Boxes object."""
    out = []
    for b in boxes:
        if int(b.cls[0]) == CLASS_PERSON and float(b.conf[0]) >= conf_min:
            x1, y1, x2, y2 = map(float, b.xyxy[0])
            out.append((x1, y1, x2, y2))
    return out
