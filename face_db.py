"""
Penyimpanan dan pencarian embedding wajah.
Mendukung banyak foto per orang dengan strategi: closest, voting, centroid.
"""
import os
import pickle
import numpy as np
from typing import List, Optional, Tuple, Dict
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


def _normalize_emb(emb: np.ndarray) -> np.ndarray:
    """L2 normalize embedding untuk cosine similarity."""
    emb = np.array(emb, dtype=np.float32).flatten()
    if emb.ndim > 1:
        emb = emb.reshape(1, -1)
    return emb / (np.linalg.norm(emb, axis=-1, keepdims=True) + 1e-8)


def _cosine_sim(emb1: np.ndarray, emb2: np.ndarray) -> float:
    """Cosine similarity antara dua vektor (0-1)."""
    s = float(np.dot(emb1.flatten(), emb2.flatten()))
    return min(1.0, max(0.0, s))


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


def get_identities() -> List[str]:
    """Daftar unik identitas (nama orang) di database."""
    records = _load_db()
    return sorted(set(r["identity"] for r in records))


def get_count_by_identity() -> Dict[str, int]:
    """Jumlah embedding per identitas."""
    records = _load_db()
    out: Dict[str, int] = {}
    for r in records:
        out[r["identity"]] = out.get(r["identity"], 0) + 1
    return out


def _find_closest_single(emb_norm: np.ndarray, records: List[dict], th: float) -> Tuple[Optional[str], float]:
    """Strategi closest: bandingkan ke tiap embedding, ambil yang tertinggi."""
    best_identity = None
    best_sim = 0.0
    for r in records:
        e = _normalize_emb(r["embedding"])
        sim = _cosine_sim(emb_norm, e)
        if sim > best_sim:
            best_sim = sim
            best_identity = r["identity"]
    if best_sim >= th:
        return best_identity, best_sim
    return None, best_sim


def _find_closest_voting(emb_norm: np.ndarray, records: List[dict], th: float) -> Tuple[Optional[str], float]:
    """Strategi voting: per identitas ambil similarity terbaik, lalu pilih identitas dengan nilai terbaik."""
    by_identity: Dict[str, float] = {}
    for r in records:
        e = _normalize_emb(r["embedding"])
        sim = _cosine_sim(emb_norm, e)
        identity = r["identity"]
        by_identity[identity] = max(by_identity.get(identity, 0), sim)
    if not by_identity:
        return None, 0.0
    best_identity = max(by_identity, key=by_identity.get)
    best_sim = by_identity[best_identity]
    if best_sim >= th:
        return best_identity, best_sim
    return None, best_sim


def _find_closest_centroid(emb_norm: np.ndarray, records: List[dict], th: float) -> Tuple[Optional[str], float]:
    """Strategi centroid: hitung centroid embedding per identitas, bandingkan ke centroid."""
    by_identity: Dict[str, List[np.ndarray]] = {}
    for r in records:
        identity = r["identity"]
        e = np.array(r["embedding"], dtype=np.float32).flatten()
        by_identity.setdefault(identity, []).append(e)
    centroids = {}
    for identity, embs in by_identity.items():
        arr = np.stack(embs, axis=0)
        centroid = np.mean(arr, axis=0)
        centroids[identity] = _normalize_emb(centroid)
    best_identity = None
    best_sim = 0.0
    for identity, cen in centroids.items():
        sim = _cosine_sim(emb_norm, cen)
        if sim > best_sim:
            best_sim = sim
            best_identity = identity
    if best_sim >= th:
        return best_identity, best_sim
    return None, best_sim


def find_closest(
    embedding: np.ndarray,
    threshold: Optional[float] = None,
) -> Tuple[Optional[str], float]:
    """
    Cari identitas yang paling mirip dengan embedding.
    Menggunakan strategi dari config: closest, voting, atau centroid.
    Returns: (identity atau None, similarity score 0-1).
    """
    records = _load_db()
    if not records:
        return None, 0.0

    th = threshold if threshold is not None else config.MIN_SIMILARITY_THRESHOLD
    emb_norm = _normalize_emb(embedding)
    strategy = getattr(config, "MATCH_STRATEGY", "closest")

    if strategy == "centroid":
        return _find_closest_centroid(emb_norm, records, th)
    if strategy == "voting":
        return _find_closest_voting(emb_norm, records, th)
    return _find_closest_single(emb_norm, records, th)


def remove_identity(identity: str) -> int:
    """
    Hapus semua embedding untuk satu identitas (nama).
    Returns: jumlah record yang dihapus.
    """
    records = _load_db()
    before = len(records)
    records = [r for r in records if r["identity"] != identity]
    removed = before - len(records)
    if removed > 0:
        _save_db(records)
    return removed


def clear_db() -> None:
    """Kosongkan database wajah."""
    _save_db([])


def count_faces() -> int:
    """Jumlah wajah (record) di database."""
    return len(_load_db())
