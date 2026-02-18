"""
Konfigurasi untuk Face Recognition.
Model: ArcFace (akurasi tinggi). Detektor RetinaFace, strategi matching, dan
preprocessing untuk meningkatkan akurasi.
"""
import os

# Direktori basis
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Database wajah (embedding + metadata) disimpan di sini
FACE_DB_PATH = os.path.join(BASE_DIR, "face_database")
FACE_DB_FILE = os.path.join(FACE_DB_PATH, "representations.pkl")
KNOWN_FACES_DIR = os.path.join(BASE_DIR, "known_faces")  # Gambar wajah yang didaftarkan

# Model: "ArcFace" (sangat akurat), "Facenet512", "Facenet", "VGG-Face", "OpenFace", "DeepFace", "DeepID"
MODEL_NAME = "ArcFace"

# Backend deteksi wajah: "retinaface" (akurasi tinggi), "mtcnn", "opencv", "ssd", "mediapipe"
# RetinaFace lebih akurat untuk wajah kecil, samping, dan variasi pencahayaan
DETECTOR_BACKEND = "retinaface"

# Threshold similarity (0–1). Semakin tinggi semakin ketat; kurangi jika terlalu banyak "Unknown"
MIN_SIMILARITY_THRESHOLD = 0.55

# Strategi pencocokan saat satu orang punya banyak embedding (banyak foto):
# - "closest" : bandingkan ke tiap embedding, ambil identitas dengan similarity tertinggi
# - "voting"  : bandingkan ke semua embedding, per identitas ambil similarity terbaik lalu pilih identitas terbaik (lebih stabil)
# - "centroid": hitung centroid embedding per identitas, bandingkan ke centroid (cepat, bagus jika banyak foto per orang)
MATCH_STRATEGY = "voting"

DISTANCE_METRIC = "cosine"  # "cosine", "euclidean", "euclidean_l2"

# Preprocessing gambar sebelum ekstraksi embedding (normalisasi pencahayaan)
PREPROCESS_INPUT = True

# Saat daftar dari folder: tambah embedding dari versi augmentasi (flip, brightness) untuk variasi
REGISTER_AUGMENT = True

# Rekomendasi minimal jumlah foto per orang untuk akurasi lebih baik (hanya untuk peringatan di CLI)
MIN_IMAGES_PER_PERSON_RECOMMENDED = 3

# Ukuran frame untuk webcam/video
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

# Buat folder jika belum ada
os.makedirs(FACE_DB_PATH, exist_ok=True)
os.makedirs(KNOWN_FACES_DIR, exist_ok=True)
