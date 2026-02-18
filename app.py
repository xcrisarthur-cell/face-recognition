#!/usr/bin/env python3
"""
Aplikasi Face Recognition - CLI.
Perintah:
  - register: daftarkan wajah dari gambar atau folder
  - recognize: kenali wajah dari gambar
  - verify: bandingkan dua gambar (apakah wajah sama)
  - webcam: deteksi & kenali wajah dari webcam
  - remove: hapus satu identitas dari database
"""
import argparse
import os
import sys
import cv2
import config
import face_db
import recognition_engine as engine


def cmd_register(args):
    if args.folder:
        count = engine.register_face_from_folder(
            args.folder,
            args.name,
            augment=getattr(args, "augment", None),
        )
        name = args.name or os.path.basename(os.path.normpath(args.folder))
        print(f"Terdaftar: {count} embedding dari folder '{args.folder}' sebagai '{name}'.")
        if getattr(config, "MIN_IMAGES_PER_PERSON_RECOMMENDED", 3) and count > 0:
            rec = config.MIN_IMAGES_PER_PERSON_RECOMMENDED
            if count < rec:
                print(f"  Tip: untuk akurasi lebih baik, tambah minimal {rec} foto per orang (atau gunakan --augment).")
    elif args.image and args.name:
        count = engine.register_face(args.image, args.name, all_faces=getattr(args, "all_faces", False))
        if count > 0:
            print(f"Berhasil mendaftarkan {count} wajah.")
        else:
            print("Gagal (pastikan file ada dan berisi wajah).")
    else:
        print("Untuk register: berikan --image PATH --name NAMA, atau --folder PATH [--name NAMA].")
        sys.exit(1)


def cmd_recognize(args):
    path = args.image
    if not path or not os.path.isfile(path):
        print("Berikan path gambar yang valid (--image PATH).")
        sys.exit(1)
    results = engine.recognize(path)
    if not results:
        print("Tidak ada wajah terdeteksi.")
        return
    for i, r in enumerate(results):
        identity = r.get("identity") or "Unknown"
        sim = r.get("similarity", 0)
        print(f"Wajah {i+1}: {identity} (similarity: {sim:.3f})")


def cmd_verify(args):
    if not args.image1 or not args.image2:
        print("Berikan --image1 PATH dan --image2 PATH.")
        sys.exit(1)
    try:
        result = engine.verify_two_faces(args.image1, args.image2)
        verified = result.get("verified", False)
        distance = result.get("distance", 0)
        threshold = result.get("threshold", 0)
        print(f"Verified: {verified}")
        print(f"Distance: {distance:.4f} (threshold: {threshold:.4f})")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


def cmd_webcam(args):
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print("Tidak dapat membuka kamera.")
        sys.exit(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)
    print("Webcam aktif. Tekan 'q' untuk keluar.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        recognitions = engine.recognize(frame)
        frame = engine.draw_results(frame, recognitions)
        cv2.imshow("Face Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()


def cmd_remove(args):
    name = getattr(args, "name", None) or getattr(args, "identity", None)
    if not name:
        print("Berikan --name NAMA yang akan dihapus.")
        sys.exit(1)
    removed = face_db.remove_identity(name)
    if removed > 0:
        print(f"Dihapus: {removed} embedding untuk '{name}'.")
    else:
        print(f"Tidak ada data untuk '{name}' di database.")


def cmd_list(args):
    records = face_db.get_all()
    if not records:
        print("Database wajah kosong.")
        return
    by_id = face_db.get_count_by_identity()
    rec = getattr(config, "MIN_IMAGES_PER_PERSON_RECOMMENDED", 3)
    print(f"Total {len(records)} embedding, {len(by_id)} orang:")
    for identity in face_db.get_identities():
        n = by_id.get(identity, 0)
        tip = f" (disarankan >= {rec} foto)" if n < rec else ""
        print(f"  - {identity}: {n} foto{tip}")


def main():
    parser = argparse.ArgumentParser(description="Face Recognition (DeepFace + ArcFace)")
    sub = parser.add_subparsers(dest="command", help="Perintah")

    # register
    p_register = sub.add_parser("register", help="Daftarkan wajah")
    p_register.add_argument("--image", "-i", help="Path gambar wajah")
    p_register.add_argument("--folder", "-f", help="Path folder berisi gambar satu orang")
    p_register.add_argument("--name", "-n", help="Nama identitas (wajib untuk --image)")
    p_register.add_argument("--augment", "-a", action="store_true", help="Dari folder: tambah embedding dari flip & variasi brightness (lebih akurat)")
    p_register.add_argument("--all-faces", action="store_true", help="Dari satu gambar: daftarkan semua wajah terdeteksi sebagai nama yang sama")
    p_register.set_defaults(func=cmd_register)

    # recognize
    p_rec = sub.add_parser("recognize", help="Kenali wajah dari gambar")
    p_rec.add_argument("--image", "-i", required=True, help="Path gambar")
    p_rec.set_defaults(func=cmd_recognize)

    # verify
    p_ver = sub.add_parser("verify", help="Bandingkan dua gambar (wajah sama atau tidak)")
    p_ver.add_argument("--image1", required=True, help="Path gambar pertama")
    p_ver.add_argument("--image2", required=True, help="Path gambar kedua")
    p_ver.set_defaults(func=cmd_verify)

    # webcam
    p_cam = sub.add_parser("webcam", help="Deteksi & kenali wajah dari webcam")
    p_cam.add_argument("--camera", "-c", type=int, default=0, help="Index kamera (default: 0)")
    p_cam.set_defaults(func=cmd_webcam)

    # list
    p_list = sub.add_parser("list", help="Tampilkan daftar wajah terdaftar")
    p_list.set_defaults(func=cmd_list)

    # remove
    p_remove = sub.add_parser("remove", help="Hapus satu identitas (nama) dari database")
    p_remove.add_argument("--name", "-n", required=True, help="Nama identitas yang akan dihapus")
    p_remove.set_defaults(func=cmd_remove)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(0)
    args.func(args)


if __name__ == "__main__":
    main()
