# Face Recognition (ProjectFaceRecog)

Program **face recognition** berbasis **DeepFace** dengan model **ArcFace** untuk akurasi tinggi. Mendukung pendaftaran wajah, identifikasi dari gambar, verifikasi dua gambar, dan deteksi real-time dari webcam.

## Fitur

- **Model ArcFace** – state-of-the-art untuk face recognition (akurasi tinggi)
- **Detektor RetinaFace** – deteksi wajah lebih akurat (wajah kecil, samping, pencahayaan beragam)
- **Banyak foto per orang** – dukung banyak embedding per identitas dengan strategi **voting** atau **centroid**
- **Preprocessing** – normalisasi pencahayaan (CLAHE) sebelum ekstraksi embedding
- **Augmentasi saat registrasi** – dari satu foto bisa generate variasi (flip, brightness) untuk data lebih kaya
- **Pendaftaran wajah** – dari satu gambar atau folder (satu folder = satu identitas)
- **Identifikasi** – kenali wajah dari gambar atau frame video
- **Verifikasi** – cek apakah dua foto adalah wajah yang sama
- **Webcam** – deteksi dan kenali wajah secara real-time

## Meningkatkan akurasi

1. **Tambahkan banyak foto per orang** (minimal 3–5, ideal 10+): variasi ekspresi, angle, dan pencahayaan.
2. **Daftar dari folder dengan `--augment`**:  
   `python app.py register --folder known_faces/John --augment`  
   Setiap foto akan menghasilkan beberapa embedding (asli + flip + variasi brightness), sehingga akurasi lebih stabil.
3. **Satu gambar berisi banyak wajah orang yang sama**:  
   `python app.py register --image grup.jpg --name "John" --all-faces`
4. **Di `config.py`**:
   - `DETECTOR_BACKEND = "retinaface"` (default) untuk deteksi lebih baik.
   - `MATCH_STRATEGY = "voting"` (default) atau `"centroid"` jika tiap orang punya banyak foto.
   - `PREPROCESS_INPUT = True` (default) untuk normalisasi pencahayaan.
   - `MIN_SIMILARITY_THRESHOLD`: naikkan (mis. 0.6) jika banyak false positive, turunkan (mis. 0.5) jika banyak false negative.

## Persyaratan

- **Python 3.9, 3.10, 3.11, atau 3.12** (TensorFlow belum mendukung Python 3.13+)
- Webcam (untuk mode webcam)

## Virtual environment (disarankan)

Agar dependency tidak bentrok dengan project lain, gunakan virtual environment.

**1. Buat venv dengan Python 3.12 atau 3.11 (penting):**
```bash
cd ProjectFaceRecog
# Gunakan salah satu, tergantung yang terpasang di sistem Anda:
python3.12 -m venv venv
# atau
python3.11 -m venv venv
```
Jika `python3.12` belum ada, install dulu (mis. lewat [python.org](https://www.python.org/downloads/) atau Homebrew: `brew install python@3.12`).

**2. Aktifkan venv:**

- **macOS/Linux:**
  ```bash
  source venv/bin/activate
  ```
- **Windows (CMD):**
  ```bash
  venv\Scripts\activate.bat
  ```
- **Windows (PowerShell):**
  ```bash
  venv\Scripts\Activate.ps1
  ```

Setelah aktif, prompt terminal akan diawali `(venv)`.

**3. Install dependency di dalam venv:**
```bash
pip install -r requirements.txt
```

**4. Menjalankan program:** gunakan `python` seperti biasa (pastikan venv masih aktif):
```bash
python app.py webcam
```

**5. Nonaktifkan venv saat selesai:**
```bash
deactivate
```

## Instalasi (tanpa venv)

```bash
cd ProjectFaceRecog
pip install -r requirements.txt
```

*Catatan: Pertama kali menjalankan, DeepFace akan mengunduh model ArcFace (~500MB). Pastikan koneksi internet aktif.*

## Penggunaan

### Langkah pertama (setelah instalasi berhasil)

Jika `python app.py list` menampilkan **"Database wajah kosong"**, lakukan:

1. **Daftarkan wajah** – siapkan foto wajah (JPG/PNG), lalu jalankan:
   ```bash
   python app.py register --image path/ke/foto.jpg --name "Nama Orang"
   ```
   Atau taruh beberapa foto satu orang di satu folder, lalu:
   ```bash
   python app.py register --folder path/ke/folder/NamaOrang
   ```
2. **Cek** – `python app.py list` akan menampilkan nama yang terdaftar.
3. **Kenali** – `python app.py recognize --image gambar_lain.jpg` atau `python app.py webcam` untuk real-time.

### 1. Daftarkan wajah

Dari satu gambar:

```bash
python app.py register --image path/ke/foto.jpg --name "Nama Orang"
```

Dari folder (semua gambar di folder = satu orang):

```bash
python app.py register --folder path/ke/folder/NamaOrang --name "Nama Orang"
# atau nama otomatis dari nama folder:
python app.py register --folder known_faces/John
# dengan augmentasi (flip + variasi brightness) untuk akurasi lebih baik:
python app.py register --folder known_faces/John --augment
```

Dari satu gambar, daftarkan **semua wajah** yang terdeteksi (mis. foto grup) sebagai satu nama:

```bash
python app.py register --image grup.jpg --name "Nama Orang" --all-faces
```

### 2. Kenali wajah dari gambar

```bash
python app.py recognize --image path/ke/gambar.jpg
```

### 3. Verifikasi dua gambar (apakah wajah sama?)

```bash
python app.py verify --image1 foto1.jpg --image2 foto2.jpg
```

### 4. Webcam (real-time)

```bash
python app.py webcam
```

Tekan **q** untuk keluar.

### 5. Lihat daftar wajah terdaftar

```bash
python app.py list
```

Menampilkan jumlah embedding per orang; jika di bawah rekomendasi (default 3), akan ada saran untuk menambah foto.

## Struktur folder disarankan

```
ProjectFaceRecog/
├── app.py                 # CLI utama
├── config.py              # Konfigurasi (model, threshold, path)
├── face_db.py             # Database embedding wajah
├── recognition_engine.py  # Engine DeepFace + ArcFace
├── requirements.txt
├── known_faces/           # Gambar wajah untuk pendaftaran
│   ├── John/
│   │   ├── 1.jpg
│   │   └── 2.jpg
│   └── Jane/
│       └── foto.jpg
└── face_database/         # File database (otomatis)
    └── representations.pkl
```

## Konfigurasi (`config.py`)

| Variabel | Deskripsi |
|----------|-----------|
| `MODEL_NAME` | Model: `"ArcFace"` (default), `"Facenet512"`, `"Facenet"`, `"VGG-Face"`, dll. |
| `DETECTOR_BACKEND` | Detektor wajah: `"retinaface"` (default), `"mtcnn"`, `"opencv"`, `"ssd"`, dll. |
| `MIN_SIMILARITY_THRESHOLD` | Ambang similarity (0–1). Semakin tinggi semakin ketat (default 0.55). |
| `MATCH_STRATEGY` | `"voting"` (default), `"centroid"`, atau `"closest"` saat satu orang punya banyak embedding. |
| `PREPROCESS_INPUT` | `True` (default): normalisasi pencahayaan sebelum ekstraksi embedding. |
| `REGISTER_AUGMENT` | `True` (default): saat daftar dari folder, tambah embedding dari augmentasi (flip, brightness). |
| `MIN_IMAGES_PER_PERSON_RECOMMENDED` | Rekomendasi minimal foto per orang (default 3); dipakai untuk saran di CLI. |

## Contoh di kode Python

```python
import recognition_engine as engine

# Daftar wajah
engine.register_face("foto.jpg", "Nama")

# Kenali
results = engine.recognize("gambar.jpg")
for r in results:
    print(r["identity"], r["similarity"])

# Verifikasi dua gambar
result = engine.verify_two_faces("foto1.jpg", "foto2.jpg")
print(result["verified"])
```

## Troubleshooting

**Error: `No matching distribution found for tensorflow>=2.15.0`**

Penyebab: TensorFlow belum mendukung Python 3.13 atau 3.14. Pastikan memakai **Python 3.9–3.12**.

1. Cek versi Python di venv saat ini: `python --version`
2. Nonaktifkan venv: `deactivate`
3. Hapus venv lama: `rm -rf venv`
4. Buat venv baru dengan Python 3.12 atau 3.11:
   ```bash
   python3.12 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```
5. Jika `python3.12` tidak ada, install dulu (macOS: `brew install python@3.12`, lalu gunakan `python3.12` atau path yang diberikan Homebrew).

## Lisensi

Dependencies (DeepFace, OpenCV, dll.) mengikuti lisensi masing-masing. Kode di repo ini bebas dipakai dan dimodifikasi.
