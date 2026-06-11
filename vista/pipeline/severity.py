"""SeverityPipeline: crop-based zero-shot severity captioning for the VISTA challenge.

Architecture (three stages):
  1. YOLO (fine-tuned on VistaCrash) — detection + tracking.
  2. Crop-based VLM — per-track zero-shot severity captioning every N frames.
  3. Peak-severity aggregation — retains the worst observed state per track.

Constraints satisfied:
  * Subclasses VistaPipeline and implements forward() / reset().
  * The captioning stage is entirely zero-shot (no VISTA-label supervision).
  * The detector may be fine-tuned on VistaCrash (bbox-only, 1,000 frames).
"""
from __future__ import annotations

from collections import defaultdict, deque
from typing import Any

import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

from vista.pipeline.base import Detection, FrameResult, VistaPipeline
from vista.utils.severity import (
    YOLO_INITIAL_CAPTION,
    YOLO_TO_CATEGORY,
    peak_severity,
    vehicle_severity_prompt,
    person_severity_prompt,
    emergency_vehicle_prompt,
)
from vista.utils import log, IGNORE_CATEGORIES


class SeverityPipeline(VistaPipeline):
    """Severity-focused UAV pipeline: YOLO tracking + crop-based VLM captioning.

    The YOLO model (ideally fine-tuned on VistaCrash) provides an initial
    severity signal via its class labels (``crashed_car`` vs ``car``).  Each
    active track's bounding-box crop is then sent to a VLM every
    ``caption_stride`` frames with a category-specific zero-shot prompt so the
    model can refine the severity description (e.g. "overturned", "on fire",
    "injured, lying").  Captions are aggregated across a rolling window with a
    peak-severity strategy — the most critical state observed is preserved for
    emergency dispatch.

    Args:
        yolo_model:
            Loaded ``ultralytics.YOLO`` instance.  Should be fine-tuned on
            VistaCrash for best ``crashed_car`` recall.
        vlm:
            Any object that exposes ``caption_crop(crop: Image.Image,
            prompt: str) -> str``.  Pass ``None`` to operate in detector-only
            mode (captions derived solely from YOLO class labels).
        caption_stride:
            Run the VLM every this many frames.  Captions are propagated
            between calls.  Default is 30 (≈ 1 s at 30 FPS).
        yolo_conf:
            YOLO confidence threshold.
        history_len:
            Number of captions kept per track for peak-severity aggregation.
        check_emergency:
            When ``True``, trucks and buses are sent to the VLM with an
            emergency-vehicle probe before the severity prompt, allowing
            ambulances / police cars to be reclassified automatically.
        cfg:
            Full pipeline config dict (from YAML). Used to read user_prompt.
    """

    def __init__(
        self,
        yolo_model: YOLO,
        vlm: Any | None = None,
        caption_stride: int = 30,
        yolo_conf: float = 0.9,
        history_len: int = 5,
        check_emergency: bool = True,
        cfg: dict | None = None,
    ) -> None:
        self.yolo = yolo_model
        self.vlm = vlm
        self.caption_stride = caption_stride
        self.yolo_conf = yolo_conf
        self.history_len = history_len
        self.check_emergency = check_emergency
        self.cfg = cfg or {}

        # per-video state
        self._track_db: dict[int, dict] = {}
        self._caption_history: dict[int, deque[str]] = defaultdict(
            lambda: deque(maxlen=history_len)
        )

    # ── VistaPipeline interface ───────────────────────────────────────────────

    def reset(self) -> None:
        self._track_db.clear()
        self._caption_history.clear()

    def forward(self, frame: Image.Image, frame_idx: int) -> FrameResult:
        bgr = _pil_to_bgr(frame)

        # ── 1. YOLO detection + tracking ──────────────────────────────────────
        results = self.yolo.track(
            bgr, persist=True, verbose=False, conf=self.yolo_conf
        )[0]

        active: dict[int, dict] = {}
        if results.boxes.id is not None:
            for box, tid_t, cls_t, conf_t in zip(
                results.boxes.xyxy,
                results.boxes.id,
                results.boxes.cls,
                results.boxes.conf,
            ):
                tid      = int(tid_t.item())
                yolo_cat = results.names.get(int(cls_t.item()), "unknown")
                if yolo_cat in IGNORE_CATEGORIES:
                    continue

                cat      = YOLO_TO_CATEGORY.get(yolo_cat, "car")
                init_cap = YOLO_INITIAL_CAPTION.get(yolo_cat, yolo_cat)
                prev     = self._track_db.get(tid, {})

                active[tid] = {
                    "bbox":      box.cpu().numpy().tolist(),
                    "category":  prev.get("category", cat),
                    "caption":   prev.get("caption", init_cap),
                    "yolo_cls":  yolo_cat,
                    "conf":      float(conf_t.item()),
                }

        # purge tracks that are no longer detected
        for tid in set(self._track_db) - set(active):
            del self._track_db[tid]

        # ── 2. Crop-based severity captioning ─────────────────────────────────
        if self.vlm is not None and frame_idx % self.caption_stride == 0:
            log(
                f"[SeverityPipeline] VLM captioning at frame {frame_idx} "
                f"({len(active)} tracks)"
            )
            w, h = frame.size
            for tid, tr in active.items():
                crop = _safe_crop(frame, tr["bbox"], w, h)
                if crop is None:
                    continue

                caption = self._vlm_caption(crop, tr)

                # upgrade "car" to "emergency_vehicle" if VLM confirms
                if tr["category"] == "car" and _is_emergency_caption(caption):
                    active[tid]["category"] = "emergency_vehicle"
                    active[tid]["caption"] = caption
                    self._caption_history[tid].clear()
                    continue

                # update rolling history and apply peak-severity aggregation
                self._caption_history[tid].append(caption)
                active[tid]["caption"] = peak_severity(
                    list(self._caption_history[tid]),
                    active[tid]["category"],
                )

        # ── 3. Merge into track DB and emit FrameResult ───────────────────────
        for tid, tr in active.items():
            self._track_db[tid] = tr

        detections = [
            Detection(
                bbox=tuple(tr["bbox"]),
                category=tr["category"],
                confidence=tr["conf"],
                track_id=tid,
                caption=tr.get("caption"),
            )
            for tid, tr in self._track_db.items()
        ]
        return FrameResult(detections=detections, frame_idx=frame_idx)

    # ── private helpers ───────────────────────────────────────────────────────

    def _vlm_caption(self, crop: Image.Image, track: dict) -> str:
        """Choose the right prompt and call the VLM for one crop."""
        cat = track["category"]
        yolo_cls = track.get("yolo_cls", "")

        # for ambiguous vehicle classes probe for emergency status first
        if (
            self.check_emergency
            and cat == "car"
            and yolo_cls in {"truck", "bus"}
        ):
            try:
                emerg_reply = self.vlm.caption_crop(
                    crop, emergency_vehicle_prompt()
                ).strip()
                if _is_emergency_caption(emerg_reply):
                    return emerg_reply
            except Exception:
                pass

        # build prompt from severity.py + optional user_prompt from yaml
        user_prompt = self.cfg.get("qwen", {}).get("user_prompt", "")
        if cat == "person":
            prompt = person_severity_prompt()
        else:
            prompt = vehicle_severity_prompt()
        if user_prompt:
            prompt = prompt + "\n" + user_prompt

        try:
            return self.vlm.caption_crop(crop, prompt).strip()
        except Exception as e:
            log(f"[SeverityPipeline] VLM error for track {track}: {e}")
            return track.get("caption", "")


# ── module-level helpers ──────────────────────────────────────────────────────

def _pil_to_bgr(image: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)


def _safe_crop(
    frame: Image.Image,
    bbox: list[float],
    w: int,
    h: int,
) -> Image.Image | None:
    """Crop *bbox* from *frame*, clamping to image boundaries."""
    x1, y1, x2, y2 = (int(v) for v in bbox)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    return frame.crop((x1, y1, x2, y2))


def _is_emergency_caption(caption: str) -> bool:
    """Return True if the VLM identified an emergency vehicle."""
    kw = {"ambulance", "fire truck", "police", "emergency vehicle"}
    low = caption.lower()
    return any(k in low for k in kw) and "not emergency" not in low