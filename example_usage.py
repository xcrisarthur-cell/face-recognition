#!/usr/bin/env python3
"""
Contoh penggunaan Face Recognition secara programatik.
Jalankan setelah menginstall dependencies: pip install -r requirements.txt
"""
import os
import recognition_engine as engine
import face_db
import config

# Pastikan folder contoh ada (opsional)
EXAMPLE_IMAGES = os.path.join(os.path.dirname(__file__), "known_faces")


def main():
    print("=== Contoh Face Recognition ===\n")
    print(f"Model: {config.MODEL_NAME}")
    print(f"Detector: {config.DETECTOR_BACKEND}\n")

    # 1) Daftarkan wajah dari folder known_faces
    # Struktur: known_faces/NamaOrang/foto1.jpg, foto2.jpg, ...
    if os.path.isdir(EXAMPLE_IMAGES):
        for name in os.listdir(EXAMPLE_IMAGES):
            folder = os.path.join(EXAMPLE_IMAGES, name)
            if os.path.isdir(folder):
                n = engine.register_face_from_folder(folder, name)
                print(f"Terdaftar: {name} ({n} gambar)")
    else:
        print("Folder 'known_faces' belum ada. Buat subfolder per orang dan isi foto.")
        print("Contoh: known_faces/John/image1.jpg")

    print(f"\nTotal wajah di database: {face_db.count_faces()}")

    # 2) Kenali wajah dari satu gambar (jika ada file contoh)
    test_image = os.path.join(EXAMPLE_IMAGES, "test.jpg")
    if not os.path.isfile(test_image):
        # Coba file mana saja di known_faces
        for root, _, files in os.walk(EXAMPLE_IMAGES):
            for f in files:
                if f.lower().endswith((".jpg", ".jpeg", ".png")):
                    test_image = os.path.join(root, f)
                    break
            if os.path.isfile(test_image):
                break

    if os.path.isfile(test_image):
        print(f"\nRecognize dari: {test_image}")
        results = engine.recognize(test_image)
        for i, r in enumerate(results):
            print(f"  Wajah {i+1}: {r.get('identity', 'Unknown')} (similarity: {r.get('similarity', 0):.3f})")
    else:
        print("\nTidak ada gambar tes. Letakkan gambar di known_faces/ atau jalankan:")
        print("  python app.py register --image path/foto.jpg --name Nama")
        print("  python app.py recognize --image path/foto.jpg")

    print("\nSelesai.")


if __name__ == "__main__":
    main()
