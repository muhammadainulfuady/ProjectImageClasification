# 👡 Klasifikasi Kerapian Sandal (Rapi vs Berantakan) — Ekstraksi Fitur Klasik Ringkas

Proyek ini mendokumentasikan dataset dan proses **ekstraksi 34 fitur klasik terbaik (*Handcrafted Features*)** pada **6 kategori fitur citra** untuk membedakan sandal dalam kondisi **Rapi** dan **Berantakan**.

---

## 📂 Struktur Direktori Dataset & File

```text
uas/ProjectImageClasification/
├── Data/
│   ├── Berantakan/                  # Citra sandal berantakan
│   └── Rapi/                        # Citra sandal rapi
├── code/
│   ├── ekstraksi_fitur.py           # Modul utama ekstraksi 34 fitur klasik terbaik
│   ├── ekstraksi_fitur_sandal.ipynb # Notebook ekstraksi, visualisasi, & seleksi fitur
│   └── rename_gambar.py             # Utilitas penamaan ulang gambar berurutan
├── hasil_ekstraksi/
│   ├── fitur_ekstraksi_sandal.csv   # Output 34 fitur klasik terbaik
│   └── fitur_terpilih_sandal.csv    # Output Top-15 fitur paling diskriminatif (dari notebook)
└── README.md                        # Dokumentasi proyek
```

---

## 🔬 34 Fitur Klasik Terbaik (6 Kategori)

Fitur diringkas agar tidak berlebihan/redundan serta seimbang dengan jumlah sampel citra:

1. **Warna (9 fitur)**:
   `mean_red`, `std_red`, `mean_green`, `std_green`, `mean_blue`, `std_blue`, `mean_gray`, `std_gray`, `color_ccv_coherent_ratio`
2. **Tekstur (7 fitur)**:
   `glcm_contrast`, `glcm_homogeneity`, `glcm_energy`, `glcm_correlation`, `glcm_entropy`, `lbp_uniform_ratio`, `gabor_orientation_dominance`
3. **Tepi & Gradien (6 fitur)**:
   `hog_entropy`, `hog_dominant_ratio`, `edge_laplacian_var`, `edge_canny_density`, `edge_canny_spatial_spread`, `edge_canny_num_contours`
4. **Bentuk & Geometri (6 fitur)**:
   `shape_area_ratio`, `shape_solidity`, `shape_circularity`, `shape_eccentricity`, `shape_num_objects`, `hu_moment_1`
5. **Titik Kunci (3 fitur)**:
   `kp_density`, `kp_spatial_spread`, `kp_desc_diversity`
6. **Domain Frekuensi (3 fitur)**:
   `fft_high_energy_ratio`, `fft_spectral_entropy`, `dwt_detail_energy_ratio`

---

## 🚀 Cara Menjalankan

### 1. Menjalankan Skrip Python
Dari folder `uas/ProjectImageClasification`:
```bash
python code/ekstraksi_fitur.py
```

### 2. Menjalankan Jupyter Notebook
Buka file [`code/ekstraksi_fitur_sandal.ipynb`](code/ekstraksi_fitur_sandal.ipynb) melalui VS Code / Jupyter Lab dan jalankan seluruh sel untuk menghasilkan `hasil_ekstraksi/fitur_ekstraksi_sandal.csv`, `hasil_ekstraksi/fitur_terpilih_sandal.csv`, serta melihat perbandingan akurasi *5-Fold Cross-Validation* dan visualisasi Top-15 fitur terbaik.

