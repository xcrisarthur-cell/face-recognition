"""
Engine Face Recognition menggunakan DeepFace (model ArcFace).
Preprocessing gambar, augmentasi saat registrasi, dan detektor RetinaFace untuk akurasi lebih baik.
"""
import os
import numpy as np
from typing import List, Optional, Tuple, Union
from deepface import DeepFace
import cv2
import config
import face_db


def _preprocess_image(img: np.ndarray) -> np.ndarray:
    """
    Normalisasi pencahayaan untuk meningkatkan konsistensi embedding.
    CLAHE pada channel L (Lab) mengurangi dampak pencahayaan berbeda.
    """
    if not getattr(config, "PREPROCESS_INPUT", True):
        return img
    try:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        lab = cv2.merge([l, a, b])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    except Exception:
        return img


def _load_and_preprocess(image_input: Union[str, np.ndarray]) -> np.ndarray:
    """Load gambar (dari path atau array), preprocess, return BGR array."""
    if isinstance(image_input, np.ndarray):
        img = image_input
    else:
        img = cv2.imread(image_input)
        if img is None:
            raise ValueError(f"Cannot read image: {image_input}")
    return _preprocess_image(img)


def _represent(image_input: Union[str, np.ndarray]) -> List[dict]:
    """
    Dapatkan embedding untuk setiap wajah di gambar.
    image_input: path file (str) atau numpy array (BGR). Preprocessing diterapkan jika aktif.
    Returns: list of {"embedding": [...], "facial_area": {"x","y","w","h"}, ...}
    """
    img = _load_and_preprocess(image_input)
    return DeepFace.represent(
        img_path=img,
        model_name=config.MODEL_NAME,
        detector_backend=config.DETECTOR_BACKEND,
        enforce_detection=False,
        align=True,
    )


def _augment_image(img: np.ndarray) -> List[np.ndarray]:
    """
    Hasilkan variasi gambar untuk augmentasi: asli, flip horizontal, brightness +/-.
    Dipakai saat registrasi agar satu foto memberi beberapa embedding.
    """
    out = [img]
    out.append(cv2.flip(img, 1))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    h, s, v = cv2.split(hsv)
    v_brighter = np.clip(v * 1.1, 0, 255).astype(np.uint8)
    v_darker = np.clip(v * 0.9, 0, 255).astype(np.uint8)
    out.append(cv2.cvtColor(cv2.merge([h, s, v_brighter]), cv2.COLOR_HSV2BGR))
    out.append(cv2.cvtColor(cv2.merge([h, s, v_darker]), cv2.COLOR_HSV2BGR))
    return out


def register_face(
    image_path: str,
    identity: str,
    all_faces: bool = False,
) -> int:
    """
    Daftarkan wajah dari satu gambar ke database.
    - all_faces=False: hanya wajah pertama yang didaftarkan.
    - all_faces=True: semua wajah di gambar didaftarkan sebagai identitas yang sama (mis. foto grup).
    Returns: jumlah wajah yang berhasil didaftarkan (0 atau lebih).
    """
    if not os.path.isfile(image_path):
        return 0
    try:
        reps = _represent(image_path)
        if not reps:
            return 0
        count = 0
        for r in reps:
            emb = r.get("embedding")
            if emb is None:
                continue
            face_db.add_face(identity, emb, image_path)
            count += 1
            if not all_faces:
                break
        return count
    except Exception:
        return 0


def register_face_from_folder(
    folder_path: str,
    identity: Optional[str] = None,
    augment: Optional[bool] = None,
) -> int:
    """
    Daftarkan semua gambar di folder sebagai satu identitas.
    - identity: nama; jika None, pakai nama folder.
    - augment: jika True, setiap gambar juga diproses dengan flip/brightness dan embedding tambahan disimpan.
    Returns: jumlah embedding yang ditambahkan (bukan jumlah file).
    """
    name = identity or os.path.basename(os.path.normpath(folder_path))
    use_augment = augment if augment is not None else getattr(config, "REGISTER_AUGMENT", False)
    count = 0
    for f in sorted(os.listdir(folder_path)):
        path = os.path.join(folder_path, f)
        if not os.path.isfile(path):
            continue
        ext = os.path.splitext(f)[1].lower()
        if ext not in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
            continue
        try:
            img = _load_and_preprocess(path)
            if use_augment:
                for aug_img in _augment_image(img):
                    reps = DeepFace.represent(
                        img_path=aug_img,
                        model_name=config.MODEL_NAME,
                        detector_backend=config.DETECTOR_BACKEND,
                        enforce_detection=False,
                        align=True,
                    )
                    for r in reps:
                        emb = r.get("embedding")
                        if emb is not None:
                            face_db.add_face(name, emb, path)
                            count += 1
                        break
            else:
                added = register_face(path, name, all_faces=False)
                count += added
        except Exception:
            continue
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
