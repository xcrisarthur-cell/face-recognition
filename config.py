"""
Konfigurasi untuk Face Recognition.
Model: ArcFace (akurasi tinggi) atau Facenet512.
"""
import os

# Direktori basis
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Database wajah (embedding + metadata) disimpan di sini
FACE_DB_PATH = os.path.join(BASE_DIR, "face_database")
FACE_DB_FILE = os.path.join(FACE_DB_PATH, "representations.pkl")
KNOWN_FACES_DIR = os.path.join(BASE_DIR, "known_faces")  # Gambar wajah yang didaftarkan

# Model: "ArcFace" (sangat akurat), "Facenet512", "Facenet", "VGG-Face", "OpenFace", "DeepFace", "DeepID"
# ArcFace direkomendasikan untuk akurasi terbaik
MODEL_NAME = "ArcFace"

# Backend deteksi wajah: "opencv", "ssd", "dlib", "retinaface", "mtcnn", "fastmtcnn", "mediapipe"
# retinaface atau mtcnn lebih akurat untuk wajah kecil/samping
DETECTOR_BACKEND = "opencv"  # Ganti ke "retinaface" untuk akurasi deteksi lebih baik

# Threshold similarity (jarak): di bawah ini = wajah sama
# ArcFace: threshold default ~0.68 (distance), atau gunakan cosine similarity
# DeepFace.verify menggunakan threshold per-model; kita bisa override
DISTANCE_METRIC = "cosine"  # "cosine", "euclidean", "euclidean_l2"

# Untuk identifikasi: minimal similarity (0-1) agar dianggap match
# Semakin tinggi semakin ketat. 0.6-0.7 biasanya bagus untuk ArcFace
MIN_SIMILARITY_THRESHOLD = 0.6

# Ukuran frame untuk webcam/video
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

# Buat folder jika belum ada
os.makedirs(FACE_DB_PATH, exist_ok=True)
os.makedirs(KNOWN_FACES_DIR, exist_ok=True)
