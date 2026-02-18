"""
Penyimpanan dan pencarian embedding wajah.
Menyimpan pasangan (identitas, embedding) dan mencari identitas terdekat.
"""
import os
import pickle
import numpy as np
from typing import List, Optional, Tuple
import config


def _load_db() -> List[dict]:
    """Muat database dari file pickle."""
    if not os.path.exists(config.FACE_DB_FILE):
        return []
    try:
        with open(config.FACE_DB_FILE, "rb") as f:
            return pickle.load(f)
    except Exception:
        return []


def _save_db(records: List[dict]) -> None:
    """Simpan database ke file pickle."""
    os.makedirs(config.FACE_DB_PATH, exist_ok=True)
    with open(config.FACE_DB_FILE, "wb") as f:
        pickle.dump(records, f)


def add_face(identity: str, embedding: np.ndarray, image_path: Optional[str] = None) -> None:
    """
    Tambahkan satu wajah ke database.
    - identity: nama atau ID orang
    - embedding: vektor dari DeepFace.represent()
    - image_path: path gambar sumber (opsional, untuk referensi)
    """
    records = _load_db()
    records.append({
        "identity": identity,
        "embedding": np.array(embedding, dtype=np.float32),
        "image_path": image_path or "",
    })
    _save_db(records)


def get_all() -> List[dict]:
    """Ambil semua record (identity, embedding, image_path)."""
    return _load_db()


def find_closest(
    embedding: np.ndarray,
    threshold: Optional[float] = None,
) -> Tuple[Optional[str], float]:
    """
    Cari identitas yang paling mirip dengan embedding.
    Menggunakan cosine similarity (semakin besar semakin mirip).
    Returns: (identity atau None, similarity score 0-1).
    """
    records = _load_db()
    if not records:
        return None, 0.0

    th = threshold if threshold is not None else config.MIN_SIMILARITY_THRESHOLD
    emb = np.array(embedding, dtype=np.float32).flatten()
    if emb.ndim > 1:
        emb = emb.reshape(1, -1)
    emb = emb / (np.linalg.norm(emb, axis=-1, keepdims=True) + 1e-8)

    best_identity = None
    best_sim = 0.0

    for r in records:
        e = np.array(r["embedding"], dtype=np.float32).flatten()
        if e.ndim > 1:
            e = e.reshape(1, -1)
        e = e / (np.linalg.norm(e, axis=-1, keepdims=True) + 1e-8)
        sim = float(np.dot(emb, e.T).squeeze())
        # Cosine similarity bisa sedikit > 1 karena numerik
        sim = min(1.0, max(0.0, sim))
        if sim > best_sim:
            best_sim = sim
            best_identity = r["identity"]

    if best_sim >= th:
        return best_identity, best_sim
    return None, best_sim


def clear_db() -> None:
    """Kosongkan database wajah."""
    _save_db([])


def count_faces() -> int:
    """Jumlah wajah (record) di database."""
    return len(_load_db())
