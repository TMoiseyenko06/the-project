"""Face recognition: detect faces in source images and auto-tag "person".

Two halves that are deliberately kept separate:

* :class:`FaceRegistry` — persistent identity bookkeeping (which embedding
  belongs to which ``face_id``, and that face's optional display name). Pure
  Python/numpy, fully testable with synthetic embeddings, no ML dependencies.
* :class:`InsightFaceDetector` — the actual detection/embedding model
  (``insightface``, ONNX-based so it stays independent of the torch/diffusers
  dependency chain). A thin adapter; swappable for a fake in tests the same
  way :class:`~imagebatch.pipeline.MockPipeline` stands in for a real
  diffusers pipeline.

A face is matched against the registry by cosine similarity against known
faces' running-average embeddings. Above the threshold reuses that face's
stable ``face_id`` (recognizing the same person across different batches, not
just within one run); below it, a new ``face_id`` is created. Renaming a face
later only ever touches this registry, never the ``person`` tags already
applied — those store the stable ``face_id``, the same way albums store a
``slug`` separately from their renamable ``name``.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from .storage import StorageError, atomic_write_json

log = logging.getLogger(__name__)

FACES_FILE = "faces.json"
SCHEMA_VERSION = 1
DEFAULT_MATCH_THRESHOLD = 0.5  # cosine similarity; ArcFace embeddings, empirically ~0.4-0.6


def _cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    import numpy as np

    va, vb = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    denom = float(np.linalg.norm(va) * np.linalg.norm(vb))
    if denom == 0.0:
        return 0.0
    return float(np.dot(va, vb) / denom)


@dataclass
class Face:
    face_id: str
    name: str | None
    sample_count: int
    created_at: float

    @property
    def label(self) -> str:
        """What to show the user: the name once set, else the stable id."""
        return self.name if self.name else self.face_id


class FaceRegistry:
    """Thread-safe, persistent store of known faces and their embeddings."""

    def __init__(self, path: str | os.PathLike[str],
                match_threshold: float = DEFAULT_MATCH_THRESHOLD) -> None:
        self.path = Path(path)
        self.match_threshold = match_threshold
        self._lock = threading.RLock()
        if not self.path.exists():
            atomic_write_json(self.path, {"version": SCHEMA_VERSION, "next_id": 1, "faces": {}})

    # -- raw json -------------------------------------------------------------
    def _read(self) -> dict[str, Any]:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return {"version": SCHEMA_VERSION, "next_id": 1, "faces": {}}
        except json.JSONDecodeError as exc:
            raise StorageError(f"{self.path} is corrupt: {exc}") from exc
        data.setdefault("faces", {})
        data.setdefault("next_id", 1)
        return data

    def _write(self, data: dict[str, Any]) -> None:
        data["version"] = SCHEMA_VERSION
        atomic_write_json(self.path, data)

    @staticmethod
    def _to_face(face_id: str, meta: dict[str, Any]) -> Face:
        return Face(face_id=face_id, name=meta.get("name"),
                    sample_count=meta.get("sample_count", 1),
                    created_at=meta.get("created_at", 0.0))

    # -- queries ----------------------------------------------------------
    def list_faces(self) -> list[Face]:
        with self._lock:
            data = self._read()
        faces = [self._to_face(fid, meta) for fid, meta in data["faces"].items()]
        faces.sort(key=lambda f: f.label.lower())
        return faces

    def get(self, face_id: str) -> Face:
        with self._lock:
            data = self._read()
        meta = data["faces"].get(face_id)
        if meta is None:
            raise StorageError(f"no such face: {face_id}")
        return self._to_face(face_id, meta)

    # -- identity matching --------------------------------------------------
    def match_or_create(self, embedding: Sequence[float]) -> str:
        """Match *embedding* against known faces, or register a new one.

        Returns the (existing or freshly created) ``face_id``. On a match, the
        stored embedding is updated to a running average so the reference
        drifts gracefully toward the center of everything seen for that face
        rather than staying pinned to the first sample.
        """
        embedding = [float(x) for x in embedding]
        with self._lock:
            data = self._read()
            best_id, best_score = None, -1.0
            for face_id, meta in data["faces"].items():
                score = _cosine_similarity(embedding, meta["embedding"])
                if score > best_score:
                    best_id, best_score = face_id, score

            if best_id is not None and best_score >= self.match_threshold:
                meta = data["faces"][best_id]
                n = meta.get("sample_count", 1)
                meta["embedding"] = [
                    (old * n + new) / (n + 1)
                    for old, new in zip(meta["embedding"], embedding)
                ]
                meta["sample_count"] = n + 1
                self._write(data)
                return best_id

            face_id = f"face_{data['next_id']:03d}"
            data["next_id"] += 1
            data["faces"][face_id] = {
                "name": None, "embedding": embedding,
                "sample_count": 1, "created_at": time.time(),
            }
            self._write(data)
            log.info("Registered new face %s (best existing match: %.3f)",
                     face_id, best_score)
            return face_id

    # -- mutation -----------------------------------------------------------
    def rename(self, face_id: str, name: str | None) -> Face:
        """Set (or clear, with None/empty) a face's display name."""
        name = (name or "").strip() or None
        with self._lock:
            data = self._read()
            if face_id not in data["faces"]:
                raise StorageError(f"no such face: {face_id}")
            data["faces"][face_id]["name"] = name
            self._write(data)
        return self.get(face_id)

    def merge(self, keep_id: str, absorb_id: str) -> Face:
        """Merge *absorb_id* into *keep_id* (for when one person got split into two).

        Combines sample counts with a weighted-average embedding and removes
        *absorb_id*. Callers are responsible for re-tagging any images that
        referenced ``absorb_id`` as ``person`` — the registry doesn't know
        about tags at all, same separation as everywhere else in this file.
        """
        if keep_id == absorb_id:
            raise StorageError("cannot merge a face with itself")
        with self._lock:
            data = self._read()
            for face_id in (keep_id, absorb_id):
                if face_id not in data["faces"]:
                    raise StorageError(f"no such face: {face_id}")
            keep, absorb = data["faces"][keep_id], data["faces"][absorb_id]
            n1, n2 = keep.get("sample_count", 1), absorb.get("sample_count", 1)
            keep["embedding"] = [
                (a * n1 + b * n2) / (n1 + n2)
                for a, b in zip(keep["embedding"], absorb["embedding"])
            ]
            keep["sample_count"] = n1 + n2
            if not keep.get("name") and absorb.get("name"):
                keep["name"] = absorb["name"]
            del data["faces"][absorb_id]
            self._write(data)
        log.info("Merged face %s into %s", absorb_id, keep_id)
        return self.get(keep_id)

    def delete(self, face_id: str) -> None:
        with self._lock:
            data = self._read()
            if face_id not in data["faces"]:
                raise StorageError(f"no such face: {face_id}")
            del data["faces"][face_id]
            self._write(data)


class FaceDetectionError(RuntimeError):
    """Raised when the underlying detector fails to load or run."""


@dataclass
class Detection:
    embedding: list[float]
    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2


class InsightFaceDetector:
    """Wraps insightface's buffalo_l pack (SCRFD detection + ArcFace embeddings).

    CPU by default: this model is tiny next to the diffusion pipeline (well
    under a second per image even on CPU), and staying off GPU sidesteps
    another onnxruntime/CUDA version pairing to get wrong on top of everything
    else a GPU-heavy setup already has to match.
    """

    def __init__(self, use_gpu: bool = False) -> None:
        self.use_gpu = use_gpu
        self._app = None

    def load(self) -> None:
        if self._app is not None:
            return
        try:
            from insightface.app import FaceAnalysis
        except ImportError as exc:
            raise FaceDetectionError(
                "Face detection needs the `insightface` package (and `onnxruntime` "
                "or `onnxruntime-gpu`), which isn't installed. Run: pip install "
                "insightface onnxruntime"
            ) from exc

        app = FaceAnalysis(name="buffalo_l",
                           providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
                           if self.use_gpu else ["CPUExecutionProvider"])
        app.prepare(ctx_id=0 if self.use_gpu else -1)
        self._app = app
        log.info("Loaded insightface buffalo_l (%s)", "GPU" if self.use_gpu else "CPU")

    def detect(self, image) -> list[Detection]:
        """Detect faces in a PIL image, returning one Detection per face found."""
        if self._app is None:
            self.load()
        import numpy as np

        # insightface expects BGR, PIL gives RGB.
        rgb = np.array(image.convert("RGB"))
        bgr = rgb[:, :, ::-1]
        faces = self._app.get(bgr)
        return [
            Detection(embedding=[float(x) for x in face.embedding], bbox=tuple(face.bbox))
            for face in faces
        ]
