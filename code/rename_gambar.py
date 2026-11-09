import os
from pathlib import Path

def rename_images(folder_path: str, prefix: str = None):
    """
    Fungsi untuk me-rename semua gambar di dalam sebuah folder secara berurutan.
    
    Contoh hasil:
    prefix = "berantakan" -> berantakan_001.jpg, berantakan_002.jpg, dst.
    """
    target_dir = Path(folder_path)
    
    # Jika prefix tidak diisi, gunakan nama foldernya langsung
    if prefix is None:
        prefix = target_dir.name
        
    # Format ekstensi gambar yang didukung
    valid_extensions = {".jpg", ".jpeg", ".png", ".webp"}
    
    # Ambil semua file gambar di dalam folder target
    files = [
        f for f in target_dir.iterdir() 
        if f.is_file() and f.suffix.lower() in valid_extensions
    ]
    
    # Urutkan file agar urutan penamaan konsisten
    files.sort(key=lambda x: x.name)
    
    print(f"--- Memproses folder: {target_dir.name} (Total: {len(files)} gambar) ---")

    # TAHAP 1: Rename ke nama sementara (temp) 
    # Trik ini mencegah konflik jika file sebelumnya sudah memiliki nama dengan pola yang sama
    temp_files = []
    for index, file in enumerate(files, start=1):
        temp_name = target_dir / f"temp_{index}_{file.name}"
        file.rename(temp_name)
        temp_files.append(temp_name)
        
    # TAHAP 2: Rename ke format final (misal: berantakan_001.jpg)
    for index, file in enumerate(temp_files, start=1):
        # f"{index:03d}" artinya angka 1 jadi '001', 10 jadi '010', dsb.
        new_name = target_dir / f"{prefix}_{index:03d}{file.suffix.lower()}"
        file.rename(new_name)
        print(f"Renamed -> {new_name.name}")

    print(f"Selesai untuk folder '{target_dir.name}'!\n")


if __name__ == "__main__":
    # Daftar folder yang ingin di-rename
    folders = ["berantakan", "rapi"]
    
    for folder in folders:
        if os.path.exists(folder):
            rename_images(folder)
        else:
            print(f"Folder '{folder}' tidak ditemukan.")
