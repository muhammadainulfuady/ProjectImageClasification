
import os
from pathlib import Path

def rename_images(folder_path: str, prefix: str = None):
    """
    Fungsi untuk me-rename semua gambar di dalam folder secara berurutan.
    """

    target_dir = Path(folder_path)

    # Jika prefix tidak diisi, gunakan nama folder
    if prefix is None:
        prefix = target_dir.name.lower()

    # Ekstensi gambar yang didukung
    valid_extensions = {".jpg", ".jpeg", ".png", ".webp"}

    # Ambil semua file gambar
    files = [
        f for f in target_dir.iterdir()
        if f.is_file() and f.suffix.lower() in valid_extensions
    ]

    # Urutkan file berdasarkan nama
    files.sort(key=lambda x: x.name)

    print(f"--- Memproses folder: {target_dir.name} "
          f"(Total: {len(files)} gambar) ---")

    # Tahap 1: Rename ke nama sementara
    temp_files = []

    for index, file in enumerate(files, start=1):
        temp_name = target_dir / f"temp_{index}_{file.name}"
        file.rename(temp_name)
        temp_files.append(temp_name)

    # Tahap 2: Rename ke nama final
    for index, file in enumerate(temp_files, start=1):
        new_name = target_dir / f"{prefix}_{index:03d}{file.suffix.lower()}"
        file.rename(new_name)
        print(f"Renamed -> {new_name.name}")

    print(f"Selesai untuk folder '{target_dir.name}'!\n")


if __name__ == "__main__":

    # Mengambil lokasi folder utama project
    BASE_DIR = Path(__file__).resolve().parent.parent

    # Lokasi folder dataset
    DATA_DIR = BASE_DIR / "Data"

    # Daftar folder dataset
    folders = ["Berantakan", "Rapi"]

    for folder in folders:
        folder_path = DATA_DIR / folder

        if folder_path.exists() and folder_path.is_dir():
            rename_images(folder_path)
        else:
            print(f"Folder '{folder}' tidak ditemukan: {folder_path}")
