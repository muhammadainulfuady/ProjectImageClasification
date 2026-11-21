# 👁️ Citra Vision: Klasifikasi Kerapian Sandal (Rapi vs Berantakan)
### Sistem Klasifikasi Citra Real-Time Berbasis Deep Learning YOLOv8

**Citra Vision** adalah aplikasi Computer Vision cerdas untuk mendeteksi dan mengklasifikasikan kondisi penataan sandal (**Rapi** vs **Berantakan**) secara otomatis menggunakan model deep learning mutakhir **YOLOv8** (`yolov8n-cls`). Sistem ini dilengkapi dengan antarmuka web interaktif yang mendukung **Input Gambar** serta pemindaian **Kamera Real-Time (Live Webcam)**.

---

## 📊 Hasil Evaluasi Model YOLOv8 (Data Uji Independen)

Pengujian dilakukan pada 26 citra uji independen (*test set*):

| Metrik Evaluasi | Nilai Performa | Keterangan |
| :--- | :---: | :--- |
| **Akurasi (Accuracy)** | **92.31%** | 24 dari 26 citra terklasifikasi dengan tepat |
| **Presisi (Precision)** | **85.71%** | Tingkat ketepatan deteksi kelas target |
| **Recall (Sensitivitas)**| **100.00%** | **Sempurna**: Seluruh 12 citra sandal rapi terdeteksi tepat tanpa kesalahan |
| **F1-Score** | **92.31%** | Keseimbangan optimal antara presisi dan sensitivitas |
| **Kecepatan Inferensi** | **~15.4 ms** | Ringan dan responsif dijalankan pada CPU standar |

---

## 📂 Struktur Direktori Proyek

```text
ProjectImageClasification/
├── data/                             # Dataset Citra Asli
│   ├── Berantakan/                   # 86 citra sandal berantakan
│   └── Rapi/                         # 75 citra sandal rapi
├── dataset_yolo/                     # Dataset Format YOLO (70:15:15)
│   ├── train/ (Berantakan, Rapi)     # 112 citra latih
│   ├── val/   (Berantakan, Rapi)     # 23 citra validasi
│   └── test/  (Berantakan, Rapi)     # 26 citra uji
├── models/                           # Bobot Model
│   ├── best_yolo_sandal.pt           # Model terbaik YOLOv8 (Akurasi 92.31%)
│   └── yolov8n-cls.pt                # Base pre-trained weights
├── code/                             # Notebook & Skrip
│   ├── main.ipynb                    # Notebook Utama Pipeline YOLOv8
│   ├── yolo.ipynb                    # Notebook Eksplorasi & Evaluasi YOLO
│   └── rename_gambar.py              # Utilitas standarisasi nama berkas
├── web/                              # Aplikasi Web "Citra Vision"
│   ├── app.py                        # Server Flask REST API (/predict, /status)
│   ├── templates/
│   │   └── index.html                # Antarmuka Web (Input Gambar & Kamera)
│   └── static/
│       ├── css/style.css             # Desain Glassmorphism & Cyberpunk Reticle HUD
│       ├── js/app.js                 # Logika Kamera Webcam & AJAX
│       └── samples/                  # Sampel gambar untuk uji cepat 1-klik
└── README.md                         # Dokumentasi Lengkap Proyek
```

---

## 🚀 Panduan Menjalankan Sistem

### 1. Menjalankan Jupyter Notebook
Buka file [`code/main.ipynb`](file:///D:/KULIAH/SEMESTER%205/psd/uas/ProjectImageClasification/code/main.ipynb) atau [`code/yolo.ipynb`](file:///D:/KULIAH/SEMESTER%205/psd/uas/ProjectImageClasification/code/yolo.ipynb) pada VS Code / Jupyter Lab dan jalankan seluruh sel untuk melihat alur kerja lengkap dari eksplorasi data, evaluasi model, hingga inferensi.

### 2. Menjalankan Aplikasi Web "Citra Vision"

1. Buka terminal pada folder `web`:
   ```bash
   cd "D:\KULIAH\SEMESTER 5\psd\uas\ProjectImageClasification\web"
   ```
2. Jalankan server aplikasi:
   ```bash
   python app.py
   ```
3. Buka peramban di: **`http://127.0.0.1:5000`**

### 3. Fitur Utama Web Citra Vision:
1. **Mode 1: Input Gambar**:
   - Tarik & lepas (*drag and drop*) gambar sandal atau gunakan penjelajah berkas.
   - Tombol uji cepat (*Quick Sample Buttons*) untuk langsung menguji sampel sandal Rapi & Berantakan.
   - Menampilkan badge hasil status `RAPI` / `BERANTAKAN`, persentase keyakinan (*confidence*), dan rincian probabilitas.
2. **Mode 2: Kamera Real-Time**:
   - Aliran video langsung dari webcam dengan *Futuristic AI Scanner Reticle HUD*.
   - Pilihan: **Ambil Snapshot** atau aktifkan **Deteksi Real-Time Otomatis** (looping inferensi setiap ~350ms).
   - Dilengkapi tabel riwayat deteksi (*Detection Log History*) dengan stempel waktu dan latensi inferensi.
