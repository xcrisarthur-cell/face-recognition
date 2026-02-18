"""
Engine Face Recognition menggunakan DeepFace (model ArcFace).
Fungsi: deteksi wajah, ekstraksi embedding, pendaftaran, identifikasi.
"""
import os
import numpy as np
from typing import List, Optional, Tuple
from deepface import DeepFace
import cv2
import config
import face_db


def _represent(image_input) -> List[dict]:
    """
    Dapatkan embedding untuk setiap wajah di gambar.
    image_input: path file (str) atau numpy array (BGR).
    Returns: list of {"embedding": [...], "facial_area": {"x","y","w","h"}, ...}
    """
    return DeepFace.represent(
        img_path=image_input,
        model_name=config.MODEL_NAME,
        detector_backend=config.DETECTOR_BACKEND,
        enforce_detection=False,
        align=True,
    )


def register_face(image_path: str, identity: str) -> bool:
    """
    Daftarkan wajah dari satu gambar ke database.
    Jika ada lebih dari satu wajah, hanya wajah pertama yang didaftarkan.
    Returns: True jika berhasil.
    """
    if not os.path.isfile(image_path):
        return False
    try:
        reps = _represent(image_path)
        if not reps:
            return False
        emb = reps[0].get("embedding")
        if emb is None:
            return False
        face_db.add_face(identity, emb, image_path)
        return True
    except Exception:
        return False


def register_face_from_folder(folder_path: str, identity: Optional[str] = None) -> int:
    """
    Daftarkan semua gambar di folder sebagai satu identitas.
    identity: nama; jika None, pakai nama folder.
    Returns: jumlah wajah yang berhasil didaftarkan.
    """
    name = identity or os.path.basename(os.path.normpath(folder_path))
    count = 0
    for f in os.listdir(folder_path):
        path = os.path.join(folder_path, f)
        if not os.path.isfile(path):
            continue
        ext = os.path.splitext(f)[1].lower()
        if ext not in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
            continue
        if register_face(path, name):
            count += 1
    return count


def recognize(image_input) -> List[dict]:
    """
    Kenali semua wajah di gambar.
    image_input: path (str) atau numpy array (BGR).
    Returns: list of {
        "identity": str or None,
        "similarity": float,
        "facial_area": {"x","y","w","h"},
    }
    """
    result = []
    try:
        reps = _represent(image_input)
    except Exception:
        return result

    for r in reps:
        emb = r.get("embedding")
        area = r.get("facial_area", {})
        if emb is None:
            continue
        identity, sim = face_db.find_closest(emb)
        result.append({
            "identity": identity,
            "similarity": sim,
            "facial_area": area,
        })
    return result


def verify_two_faces(image_path_1: str, image_path_2: str) -> dict:
    """
    Verifikasi apakah dua gambar berisi wajah yang sama.
    Returns: dict dari DeepFace.verify (verified, distance, threshold, ...).
    """
    return DeepFace.verify(
        img1_path=image_path_1,
        img2_path=image_path_2,
        model_name=config.MODEL_NAME,
        detector_backend=config.DETECTOR_BACKEND,
        distance_metric=config.DISTANCE_METRIC,
    )


def draw_results(frame: np.ndarray, recognitions: List[dict]) -> np.ndarray:
    """Gambar bbox dan label identitas di frame (untuk video/webcam)."""
    out = frame.copy()
    for r in recognitions:
        area = r.get("facial_area") or {}
        x = area.get("x", 0)
        y = area.get("y", 0)
        w = area.get("w", 0)
        h = area.get("h", 0)
        identity = r.get("identity") or "Unknown"
        sim = r.get("similarity", 0)
        label = f"{identity} ({sim:.2f})"
        color = (0, 255, 0) if r.get("identity") else (0, 0, 255)
        cv2.rectangle(out, (x, y), (x + w, y + h), color, 2)
        cv2.putText(
            out, label, (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2,
        )
    return out
