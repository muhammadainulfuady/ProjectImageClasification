from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage, signal
from scipy.spatial import ConvexHull

# Ukuran standar untuk ekstraksi fitur spasial, tekstur, bentuk, keypoint, dan frekuensi
STANDARD_SIZE = (256, 256)
RESAMPLE_BILINEAR = getattr(Image, "Resampling", Image).BILINEAR


def _build_lbp_uniform_lut() -> np.ndarray:
    """Lookup table untuk mendeteksi pola LBP uniform (transisi bit <= 2)."""
    lut = np.zeros(256, dtype=np.uint8)
    for code in range(256):
        bits = [(code >> k) & 1 for k in range(8)]
        transitions = sum(bits[k] != bits[(k + 1) % 8] for k in range(8))
        lut[code] = 1 if transitions <= 2 else 0
    return lut


_LBP_UNIFORM_LUT = _build_lbp_uniform_lut()


def _build_gabor_bank() -> list:
    """Bank filter Gabor 2D (4 orientasi) untuk mengukur dominasi arah tekstur."""
    kernels = []
    ksize = 15
    half = ksize // 2
    y, x = np.mgrid[-half:half + 1, -half:half + 1]
    sigma, lam, gamma = 3.0, 6.0, 0.5
    for deg in (0, 45, 90, 135):
        theta = np.deg2rad(deg)
        x_theta = x * np.cos(theta) + y * np.sin(theta)
        y_theta = -x * np.sin(theta) + y * np.cos(theta)
        gb = np.exp(-0.5 * ((x_theta ** 2) / (sigma ** 2) + (y_theta ** 2) / ((sigma / gamma) ** 2))) * np.cos(
            2.0 * np.pi * x_theta / lam
        )
        kernels.append((deg, gb - np.mean(gb)))
    return kernels


_GABOR_BANK = _build_gabor_bank()


def find_dataset_dir(start_dir: str | Path | None = None) -> Path:
    """Mencari folder 'Data' yang berisi tepat 2 kelas ('Rapi' dan 'Berantakan')."""
    search_roots = []
    if start_dir is not None:
        search_roots.append(Path(start_dir))
    script_base = Path(__file__).resolve().parent.parent
    cwd = Path.cwd()
    search_roots.extend([
        script_base / "Data",
        script_base,
        cwd / "Data",
        cwd.parent / "Data",
        cwd / "uas" / "ProjectImageClasification" / "Data",
        cwd / "uas" / "SandalClasification" / "Data",
    ])

    for root in search_roots:
        if not root.exists() or not root.is_dir():
            continue
        subdirs = {d.name.lower(): d for d in root.iterdir() if d.is_dir()}
        if "rapi" in subdirs and "berantakan" in subdirs:
            return root
        data_sub = root / "Data"
        if data_sub.exists() and data_sub.is_dir():
            data_subdirs = {d.name.lower(): d for d in data_sub.iterdir() if d.is_dir()}
            if "rapi" in data_subdirs and "berantakan" in data_subdirs:
                return data_sub

    raise FileNotFoundError(
        "Direktori dataset 'Data' berisi kelas 'Rapi' dan 'Berantakan' tidak ditemukan."
    )


def extract_color_features(img_orig_rgb: Image.Image, arr_orig: np.ndarray, gray_std: np.ndarray) -> dict:
    """
    1. Fitur Warna Ringkas (9 fitur):
       - Rata-rata & Standar Deviasi per channel (Red, Green, Blue, Grayscale)
       - Color Coherence Ratio (proporsi area warna yang utuh/koheren)
    """
    r_orig, g_orig, b_orig = arr_orig[:, :, 0], arr_orig[:, :, 1], arr_orig[:, :, 2]
    gray_orig = 0.299 * r_orig + 0.587 * g_orig + 0.114 * b_orig

    # Color Coherence Ratio (8 level intensitas pada gambar standar 256x256)
    q8 = np.clip((gray_std // 32).astype(np.int32), 0, 7)
    struct = np.ones((3, 3), dtype=int)
    tau = max(1, int(0.01 * q8.size))
    total_coh = 0
    for b_idx in range(8):
        mask = q8 == b_idx
        if np.any(mask):
            labeled, _ = ndimage.label(mask, structure=struct)
            counts = np.bincount(labeled.ravel())[1:]
            total_coh += int(np.sum(counts[counts >= tau]))

    return {
        "width": int(img_orig_rgb.width),
        "height": int(img_orig_rgb.height),
        "mean_red": round(float(np.mean(r_orig)), 4),
        "std_red": round(float(np.std(r_orig)), 4),
        "mean_green": round(float(np.mean(g_orig)), 4),
        "std_green": round(float(np.std(g_orig)), 4),
        "mean_blue": round(float(np.mean(b_orig)), 4),
        "std_blue": round(float(np.std(b_orig)), 4),
        "mean_gray": round(float(np.mean(gray_orig)), 4),
        "std_gray": round(float(np.std(gray_orig)), 4),
        "color_ccv_coherent_ratio": round(total_coh / float(q8.size), 6),
    }


def extract_texture_features(gray_std: np.ndarray) -> dict:
    """
    2. Fitur Tekstur Terbaik (7 fitur):
       - GLCM (Contrast, Homogeneity, Energy, Correlation, Entropy)
       - LBP Uniform Ratio (proporsi mikro-tekstur teratur)
       - Gabor Orientation Dominance (kekuatan arah tekstur dominan)
    """
    q16 = np.clip((gray_std // 16).astype(np.int32), 0, 15)
    H, W = q16.shape
    i_idx, j_idx = np.indices((16, 16))
    offsets = ((0, 1), (-1, 1), (-1, 0), (-1, -1))

    contrasts, homogeneities, energies, correlations, entropies = [], [], [], [], []
    for dy, dx in offsets:
        r1_s, r1_e = max(0, -dy), H - max(0, dy)
        c1_s, c1_e = max(0, -dx), W - max(0, dx)
        a = q16[r1_s:r1_e, c1_s:c1_e].ravel()
        b = q16[r1_s + dy:r1_e + dy, c1_s + dx:c1_e + dx].ravel()

        P = np.bincount(a * 16 + b, minlength=256).reshape(16, 16).astype(np.float64)
        P = P + P.T
        P = P / (P.sum() + 1e-12)

        contrasts.append(float(np.sum(P * ((i_idx - j_idx) ** 2))))
        homogeneities.append(float(np.sum(P / (1.0 + (i_idx - j_idx) ** 2))))
        energies.append(float(np.sqrt(np.sum(P ** 2))))
        entropies.append(-float(np.sum(P * np.log2(P + 1e-12))))

        mu_i = float(np.sum(i_idx * P))
        mu_j = float(np.sum(j_idx * P))
        std_i = float(np.sqrt(np.sum(((i_idx - mu_i) ** 2) * P)))
        std_j = float(np.sqrt(np.sum(((j_idx - mu_j) ** 2) * P)))
        if std_i * std_j > 1e-12:
            corr = float(np.sum(((i_idx - mu_i) * (j_idx - mu_j)) * P) / (std_i * std_j))
        else:
            corr = 1.0
        correlations.append(corr)

    # LBP Uniform Ratio
    center = gray_std[1:H - 1, 1:W - 1]
    codes = np.zeros_like(center, dtype=np.uint8)
    nbr_offsets = ((-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1))
    for bit_idx, (dy, dx) in enumerate(nbr_offsets):
        nbr = gray_std[1 + dy:H - 1 + dy, 1 + dx:W - 1 + dx]
        codes |= ((nbr >= center).astype(np.uint8) << bit_idx)
    lbp_uniform_ratio = float(np.mean(_LBP_UNIFORM_LUT[codes]))

    # Gabor Orientation Dominance
    gray_128 = gray_std[::2, ::2] / 255.0
    orient_energies = [float(np.mean(signal.fftconvolve(gray_128, k, mode="same") ** 2)) for _, k in _GABOR_BANK]
    gabor_dom = float(np.max(orient_energies) / (np.mean(orient_energies) + 1e-12))

    return {
        "glcm_contrast": round(float(np.mean(contrasts)), 6),
        "glcm_homogeneity": round(float(np.mean(homogeneities)), 6),
        "glcm_energy": round(float(np.mean(energies)), 6),
        "glcm_correlation": round(float(np.mean(correlations)), 6),
        "glcm_entropy": round(float(np.mean(entropies)), 6),
        "lbp_uniform_ratio": round(lbp_uniform_ratio, 6),
        "gabor_orientation_dominance": round(gabor_dom, 6),
    }


def _canny_like_edges(gray_norm: np.ndarray) -> np.ndarray:
    """Deteksi tepi gaya Canny (Gaussian + Sobel + NMS + Double Threshold + Hysteresis)."""
    smooth = ndimage.gaussian_filter(gray_norm, sigma=1.2)
    gx = ndimage.sobel(smooth, axis=1)
    gy = ndimage.sobel(smooth, axis=0)
    mag = np.hypot(gx, gy)
    ang = np.rad2deg(np.arctan2(gy, gx)) % 180.0

    p = np.pad(mag, 1, mode="constant", constant_values=0.0)
    nms = np.zeros_like(mag)

    d0 = (ang < 22.5) | (ang >= 157.5)
    d45 = (ang >= 22.5) & (ang < 67.5)
    d90 = (ang >= 67.5) & (ang < 112.5)
    d135 = (ang >= 112.5) & (ang < 157.5)

    keep = (
        (d0 & (mag >= p[1:-1, :-2]) & (mag >= p[1:-1, 2:]))
        | (d45 & (mag >= p[:-2, 2:]) & (mag >= p[2:, :-2]))
        | (d90 & (mag >= p[:-2, 1:-1]) & (mag >= p[2:, 1:-1]))
        | (d135 & (mag >= p[:-2, :-2]) & (mag >= p[2:, 2:]))
    )
    nms[keep] = mag[keep]

    pos_vals = nms[nms > 0]
    if pos_vals.size == 0:
        return np.zeros_like(mag, dtype=bool)
    high_thr = float(np.percentile(pos_vals, 85))
    if high_thr <= 1e-12:
        return np.zeros_like(mag, dtype=bool)
    return ndimage.binary_propagation(
        nms >= high_thr,
        structure=np.ones((3, 3), dtype=bool),
        mask=(nms >= 0.4 * high_thr),
    )


def extract_edge_gradient_features(gray_std: np.ndarray) -> dict:
    """
    3. Fitur Tepi & Gradien Terbaik (6 fitur):
       - HOG Entropy & HOG Dominant Ratio (keteraturan arah gradien)
       - Laplacian Variance (ketajaman/detail tepi)
       - Canny Edge Density, Spatial Spread, & Number of Edge Contours
    """
    g = gray_std / 255.0
    H, W = g.shape

    gx = ndimage.sobel(g, axis=1)
    gy = ndimage.sobel(g, axis=0)
    mag_sobel = np.hypot(gx, gy)
    lap_var = float(np.var(ndimage.laplace(g)))

    # HOG 9-bin orientasi
    angles = np.rad2deg(np.arctan2(gy, gx)) % 180.0
    bins = np.clip((angles // 20.0).astype(np.int32), 0, 8)
    hog_hist = np.bincount(bins.ravel(), weights=mag_sobel.ravel(), minlength=9).astype(np.float64)
    hog_prob = hog_hist / (hog_hist.sum() + 1e-12)

    edges = _canny_like_edges(g)
    ey, ex = np.where(edges)
    if len(ex) > 1:
        spread = float(np.hypot(np.std(ex) / W, np.std(ey) / H))
    else:
        spread = 0.0
    _, n_contours = ndimage.label(edges, structure=np.ones((3, 3), dtype=int))

    return {
        "hog_entropy": round(-float(np.sum(hog_prob * np.log2(hog_prob + 1e-12))), 6),
        "hog_dominant_ratio": round(float(np.max(hog_prob)), 6),
        "edge_laplacian_var": round(lap_var, 6),
        "edge_canny_density": round(float(np.mean(edges)), 6),
        "edge_canny_spatial_spread": round(spread, 6),
        "edge_canny_num_contours": int(n_contours),
    }


def _otsu_foreground_mask(gray_std: np.ndarray) -> np.ndarray:
    """Segmentasi objek otomatis dengan Otsu Thresholding + deteksi polaritas background dari tepi."""
    gray_u8 = np.clip(np.round(gray_std), 0, 255).astype(np.uint8)
    hist = np.bincount(gray_u8.ravel(), minlength=256).astype(np.float64)
    bins = np.arange(256, dtype=np.float64)
    sum_total = float(np.dot(bins, hist))

    w_b = np.cumsum(hist)
    w_f = float(gray_u8.size) - w_b
    sum_b = np.cumsum(bins * hist)
    valid = (w_b > 0) & (w_f > 0)

    mu_b = np.zeros(256, dtype=np.float64)
    mu_f = np.zeros(256, dtype=np.float64)
    mu_b[valid] = sum_b[valid] / w_b[valid]
    mu_f[valid] = (sum_total - sum_b[valid]) / w_f[valid]
    thresh = int(np.argmax(w_b * w_f * ((mu_b - mu_f) ** 2)))

    mask = gray_u8 <= thresh
    border = np.concatenate([mask[:5, :].ravel(), mask[-5:, :].ravel(), mask[:, :5].ravel(), mask[:, -5:].ravel()])
    if np.mean(border) > 0.55 and np.sum(~mask) > 25:
        mask = ~mask

    mask = ndimage.binary_opening(mask, structure=np.ones((3, 3), dtype=bool))
    return ndimage.binary_closing(mask, structure=np.ones((5, 5), dtype=bool))


def extract_shape_features(gray_std: np.ndarray) -> dict:
    """
    4. Fitur Bentuk & Geometri Terbaik (6 fitur):
       - Area Ratio, Solidity (Convex Hull), Circularity, Eccentricity, Num Objects
       - Hu Moment ke-1 (momen inersia bentuk invarian translasi/skala/rotasi)
    """
    mask = _otsu_foreground_mask(gray_std)
    area = float(np.sum(mask))
    if area < 16:
        return {
            "shape_area_ratio": 0.0,
            "shape_solidity": 0.0,
            "shape_circularity": 0.0,
            "shape_eccentricity": 0.0,
            "shape_num_objects": 0,
            "hu_moment_1": 0.0,
        }

    perim_mask = mask ^ ndimage.binary_erosion(mask, structure=np.ones((3, 3), dtype=bool))
    perimeter = float(np.sum(perim_mask))
    ys, xs = np.where(mask)

    try:
        hull_area = float(ConvexHull(np.column_stack([xs, ys])).volume)
    except Exception:
        hull_area = area

    pts = np.column_stack([ys - ys.mean(), xs - xs.mean()])
    if pts.shape[0] > 2:
        eigvals = np.sort(np.linalg.eigvalsh(np.cov(pts, rowvar=False)))[::-1]
        ecc = float(np.sqrt(max(0.0, 1.0 - (eigvals[1] / (eigvals[0] + 1e-12))))) if eigvals[0] > 1e-12 else 0.0
    else:
        ecc = 0.0

    labeled, _ = ndimage.label(mask, structure=np.ones((3, 3), dtype=int))
    comp_sizes = np.bincount(labeled.ravel())[1:]
    num_objects = int(np.sum(comp_sizes >= 0.005 * mask.size))

    dx_c = xs.astype(np.float64) - float(np.mean(xs))
    dy_c = ys.astype(np.float64) - float(np.mean(ys))
    hu1 = (float(np.sum(dx_c ** 2)) + float(np.sum(dy_c ** 2))) / (area ** 2)
    hu1_log = -np.sign(hu1) * np.log10(np.abs(hu1) + 1e-12)

    return {
        "shape_area_ratio": round(area / float(mask.size), 6),
        "shape_solidity": round(area / (hull_area + 1e-12), 6),
        "shape_circularity": round((4.0 * np.pi * area) / ((perimeter ** 2) + 1e-12), 6),
        "shape_eccentricity": round(ecc, 6),
        "shape_num_objects": num_objects,
        "hu_moment_1": round(float(hu1_log), 6),
    }


def extract_keypoint_features(gray_std: np.ndarray) -> dict:
    """
    5. Fitur Titik Kunci Ringkas (3 fitur):
       - Harris Corner Density, Spatial Spread, & Descriptor Diversity (Top-50 keypoints)
    """
    g = gray_std / 255.0
    H, W = g.shape
    ix = ndimage.sobel(g, axis=1)
    iy = ndimage.sobel(g, axis=0)
    sxx = ndimage.gaussian_filter(ix * ix, sigma=1.5)
    syy = ndimage.gaussian_filter(iy * iy, sigma=1.5)
    sxy = ndimage.gaussian_filter(ix * iy, sigma=1.5)

    R = (sxx * syy - sxy * sxy) - 0.04 * ((sxx + syy) ** 2)
    r_max = float(np.max(R))
    if r_max <= 1e-12:
        return {"kp_density": 0.0, "kp_spatial_spread": 0.0, "kp_desc_diversity": 0.0}

    kp_mask = (R == ndimage.maximum_filter(R, size=5)) & (R > 0.01 * r_max)
    ky, kx = np.where(kp_mask)
    kp_vals = R[kp_mask]
    kp_count = int(len(kx))

    if kp_count == 0:
        return {"kp_density": 0.0, "kp_spatial_spread": 0.0, "kp_desc_diversity": 0.0}

    kp_density = float(kp_count) / float(g.size) * 1000.0
    kp_spread = float(np.hypot(np.std(kx) / W, np.std(ky) / H)) if kp_count > 1 else 0.0

    mag = np.hypot(ix, iy)
    ang_360 = np.rad2deg(np.arctan2(iy, ix)) % 360.0
    descs = []
    for idx in np.argsort(kp_vals)[::-1][:50]:
        yc, xc = int(ky[idx]), int(kx[idx])
        y0, y1 = max(0, yc - 8), min(H, yc + 8)
        x0, x1 = max(0, xc - 8), min(W, xc + 8)
        p_bins = np.clip((ang_360[y0:y1, x0:x1] // 45.0).astype(np.int32), 0, 7)
        d = np.bincount(p_bins.ravel(), weights=mag[y0:y1, x0:x1].ravel(), minlength=8).astype(np.float64)
        descs.append(d / (np.linalg.norm(d) + 1e-12))

    diversity = float(np.mean(np.std(np.vstack(descs), axis=0))) if len(descs) > 1 else 0.0
    return {
        "kp_density": round(kp_density, 6),
        "kp_spatial_spread": round(kp_spread, 6),
        "kp_desc_diversity": round(diversity, 6),
    }


def _haar_dwt2(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Satu tingkat transformasi Haar Wavelet 2D murni NumPy."""
    l_row = (arr[:, 0::2] + arr[:, 1::2]) * 0.5
    h_row = (arr[:, 0::2] - arr[:, 1::2]) * 0.5
    ll = (l_row[0::2, :] + l_row[1::2, :]) * 0.5
    lh = (l_row[0::2, :] - l_row[1::2, :]) * 0.5
    hl = (h_row[0::2, :] + h_row[1::2, :]) * 0.5
    hh = (h_row[0::2, :] - h_row[1::2, :]) * 0.5
    return ll, lh, hl, hh


def extract_frequency_features(gray_std: np.ndarray) -> dict:
    """
    6. Fitur Domain Frekuensi Ringkas (3 fitur):
       - FFT High Frequency Energy Ratio & Spectral Entropy
       - Haar DWT Level-1 Detail Energy Ratio (LH1 + HL1 + HH1)
    """
    g_centered = (gray_std / 255.0) - float(np.mean(gray_std / 255.0))
    H, W = g_centered.shape

    P = np.abs(np.fft.fftshift(np.fft.fft2(g_centered))) ** 2
    total_power = float(np.sum(P)) + 1e-12
    cy, cx = H // 2, W // 2
    yy, xx = np.ogrid[:H, :W]
    radius = np.hypot(yy - cy, xx - cx)
    r_max = float(min(cy, cx))

    p_norm = P / total_power
    fft_high = float(np.sum(P[radius > 0.45 * r_max]) / total_power)
    fft_ent = -float(np.sum(p_norm * np.log2(p_norm + 1e-12)))

    ll1, lh1, hl1, hh1 = _haar_dwt2(g_centered)
    detail_e = float(np.sum(lh1 ** 2) + np.sum(hl1 ** 2) + np.sum(hh1 ** 2))
    total_dwt = detail_e + float(np.sum(ll1 ** 2)) + 1e-12

    return {
        "fft_high_energy_ratio": round(fft_high, 6),
        "fft_spectral_entropy": round(fft_ent, 6),
        "dwt_detail_energy_ratio": round(detail_e / total_dwt, 6),
    }


def ekstrak_fitur_gambar(image_path: str | Path) -> dict:
    """Mengekstrak 34 fitur klasik terbaik (6 kategori) dari 1 gambar."""
    with Image.open(image_path) as img:
        img_orig_rgb = img.convert("RGB")
        arr_orig = np.array(img_orig_rgb, dtype=np.float64)

        img_std_rgb = img_orig_rgb.resize(STANDARD_SIZE, resample=RESAMPLE_BILINEAR)
        arr_std = np.array(img_std_rgb, dtype=np.float64)
        gray_std = 0.299 * arr_std[:, :, 0] + 0.587 * arr_std[:, :, 1] + 0.114 * arr_std[:, :, 2]

        features = {}
        features.update(extract_color_features(img_orig_rgb, arr_orig, gray_std))
        features.update(extract_texture_features(gray_std))
        features.update(extract_edge_gradient_features(gray_std))
        features.update(extract_shape_features(gray_std))
        features.update(extract_keypoint_features(gray_std))
        features.update(extract_frequency_features(gray_std))
        return features


def proses_dataset(dataset_dir: str | Path | None = None, output_dir: str | Path | None = None) -> tuple[pd.DataFrame, Path]:
    """Memverifikasi 2 kelas ('Rapi' & 'Berantakan'), mengekstrak 34 fitur terbaik, dan menyimpan ke CSV."""
    dataset_path = find_dataset_dir(dataset_dir)
    print(f"Direktori Dataset: {dataset_path.resolve()}")

    expected_classes = ["Rapi", "Berantakan"]
    actual_subdirs = {d.name.lower(): d.name for d in dataset_path.iterdir() if d.is_dir()}
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    print("\n--- Verifikasi Direktori Kelas ---")
    class_map = {}
    for expected in expected_classes:
        if expected.lower() not in actual_subdirs:
            raise ValueError(f"Kelas '{expected}' tidak ditemukan di {dataset_path}")
        actual_name = actual_subdirs[expected.lower()]
        class_map[expected] = actual_name
        files = [f for f in (dataset_path / actual_name).iterdir() if f.is_file() and f.suffix.lower() in valid_exts]
        print(f"[OK] Kelas '{expected}' ditemukan: folder '{actual_name}' ({len(files)} gambar)")

    print("\n--- Memulai Ekstraksi 34 Fitur Klasik Terbaik (6 Kategori) ---")
    records = []
    for class_label in expected_classes:
        class_path = dataset_path / class_map[class_label]
        image_files = sorted(
            [f for f in class_path.iterdir() if f.is_file() and f.suffix.lower() in valid_exts],
            key=lambda p: p.name,
        )
        print(f"Memproses {len(image_files)} gambar dari kelas '{class_label}'...")
        for idx, fpath in enumerate(image_files, start=1):
            feats = ekstrak_fitur_gambar(fpath)
            records.append({"filename": fpath.name, "label": class_label, **feats})
            if idx % 20 == 0 or idx == len(image_files):
                print(f"  -> {class_label}: {idx}/{len(image_files)} selesai")

    df = pd.DataFrame(records)

    if output_dir is None:
        base_dir = dataset_path.parent if dataset_path.name.lower() == "data" else dataset_path
        out_path = base_dir / "hasil_ekstraksi"
    else:
        out_path = Path(output_dir)

    out_path.mkdir(parents=True, exist_ok=True)
    csv_path = out_path / "fitur_ekstraksi_sandal.csv"
    df.to_csv(csv_path, index=False)

    print(f"\n[SUKSES] Total {df.shape[0]} gambar & {df.shape[1]} kolom disimpan ke:")
    print(f" -> {csv_path.resolve()}")
    return df, csv_path


if __name__ == "__main__":
    proses_dataset()
