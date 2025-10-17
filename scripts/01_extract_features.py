
import argparse
import math
import re
from pathlib import Path
import numpy as np
import pandas as pd
import warnings

from skimage.io import imread
from skimage.color import rgb2gray
from skimage.util import img_as_ubyte
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
import cv2

warnings.filterwarnings("ignore", category=FutureWarning)

IMD_RE = re.compile(r"\bIMD\d{3}\b", re.IGNORECASE)
IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

def find_id_from_name(name: str) -> str | None:
    m = IMD_RE.search(name)
    return m.group(0).upper() if m else None

def load_gray(path: Path) -> np.ndarray:
    img = imread(str(path))
    if img.ndim == 3 and img.shape[-1] == 4:
        img = img[..., :3]
    if img.ndim == 3:
        img = rgb2gray(img)
    img = img.astype(np.float32)
    if img.max() > 1.0:
        img /= 255.0
    return img

def resize_if_needed(img: np.ndarray, size: int | None) -> np.ndarray:
    if not size:
        return img
    h, w = img.shape[:2]
    if max(h, w) == size and min(h, w) == size:
        return img
    if h >= w:
        new_h, new_w = size, int(w * (size / h))
    else:
        new_w, new_h = size, int(h * (size / w))
    img_r = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return img_r

def apply_clahe(img_gray: np.ndarray) -> np.ndarray:
    g = img_as_ubyte(np.clip(img_gray, 0, 1))
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    eq = clahe.apply(g) / 255.0
    return eq.astype(np.float32)

def find_mask_for_id(mask_dir: Path, iid: str) -> Path | None:
    pats = [
        f"{iid}*", f"{iid.lower()}*",
        f"*{iid}*.png", f"*{iid}*.jpg", f"*{iid}*.tif", f"*{iid}*.bmp"
    ]
    for pat in pats:
        cands = sorted(mask_dir.glob(pat))
        for p in cands:
            if p.is_file():
                return p
    for p in mask_dir.rglob("*"):
        if p.is_file() and IMD_RE.search(p.name):
            if find_id_from_name(p.name) == iid:
                return p
    return None

def binarize_mask(mask_img: np.ndarray, thresh: float = 0.5, invert: bool = False,
                  target_shape: tuple[int, int] | None = None) -> np.ndarray:
    m = mask_img
    if m.ndim == 3:
        m = rgb2gray(m)
    m = m.astype(np.float32)
    if m.max() > 1.0:
        m /= 255.0
    m = (m > thresh).astype(np.float32)

    if target_shape is not None and m.shape != target_shape:
        H, W = target_shape
        m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST).astype(np.float32)

    if invert:
        m = 1.0 - m
    return m

def apply_mask(img_gray: np.ndarray, mask_img: np.ndarray | None,
               thresh=0.5, invert=False) -> np.ndarray:
    if mask_img is None:
        return img_gray
    m = binarize_mask(mask_img, thresh=thresh, invert=invert, target_shape=img_gray.shape)
    return img_gray * m


def feat_lbp(img_gray: np.ndarray, P=8, R=1, method="uniform") -> np.ndarray:
    lbp = local_binary_pattern(img_gray, P, R, method=method)
    n_bins = P + 2
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
    return hist.astype(np.float32)

def feat_hu(img_gray: np.ndarray) -> np.ndarray:
    g = img_as_ubyte(np.clip(img_gray, 0, 1))
    _, thr = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    m = cv2.moments(thr)
    hu = cv2.HuMoments(m).flatten()
    hu = -np.sign(hu) * np.log10(np.abs(hu) + 1e-12)
    return hu.astype(np.float32)

def _angles_to_radians(angles_deg: list[int]) -> list[float]:
    return [a * math.pi / 180.0 for a in angles_deg]

def feat_glcm(img_gray: np.ndarray,
              distances=(1, 2, 3),
              angles_deg=(0, 45, 90, 135),
              levels=256) -> np.ndarray:
    g = img_as_ubyte(np.clip(img_gray, 0, 1))
    angles = _angles_to_radians(list(angles_deg))
    glcm = graycomatrix(g, distances=distances, angles=angles,
                        levels=levels, symmetric=True, normed=True)
    props = ["contrast", "dissimilarity", "homogeneity", "ASM", "energy", "correlation"]
    feats = []
    for p in props:
        v = graycoprops(glcm, p).ravel()
        feats.append(v.mean()); feats.append(v.std())
    return np.array(feats, dtype=np.float32)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--orig_dir", required=True, help="Pasta com imagens originais (ex.: data/orig)")
    ap.add_argument("--mask_dir", required=True, help="Pasta com máscaras binárias (ex.: data/mask)")
    ap.add_argument("--labels_csv", required=True, help="CSV id,label (Passo 2)")
    ap.add_argument("--out_dir", default="features", help="Pasta de saída dos CSVs")

    ap.add_argument("--resize", type=int, default=None, help="Redimensiona mantendo aspecto (lado maior = N px)")
    ap.add_argument("--clahe", action="store_true", help="Aplicar CLAHE (equalização adaptativa) nas imagens")
    ap.add_argument("--mask-thresh", type=float, default=0.5, help="Limiar p/ binarizar a máscara (0..1)")
    ap.add_argument("--invert-mask", action="store_true", help="Inverter máscara (se fundo/vazios estiverem brancos)")

    ap.add_argument("--lbp-P", type=int, default=8, help="LBP vizinhos")
    ap.add_argument("--lbp-R", type=float, default=1.0, help="LBP raio")
    ap.add_argument("--lbp-method", type=str, default="uniform", choices=["default","ror","uniform","var"])

    ap.add_argument("--glcm-levels", type=int, default=256, help="Níveis de cinza para GLCM")
    ap.add_argument("--glcm-dists", type=int, nargs="+", default=[1,2,3], help="Distâncias (pixels) p/ GLCM")
    ap.add_argument("--glcm-angles", type=int, nargs="+", default=[0,45,90,135], help="Ângulos (graus) p/ GLCM")

    args = ap.parse_args()

    orig_dir = Path(args.orig_dir); mask_dir = Path(args.mask_dir)
    out_dir  = Path(args.out_dir);  out_dir.mkdir(parents=True, exist_ok=True)

    labels = pd.read_csv(args.labels_csv, dtype={"id": str, "label": int})
    labels["id"] = labels["id"].str.upper().str.strip()
    label_map = dict(zip(labels["id"], labels["label"]))

    files = [p for p in orig_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS]
    files = sorted(files, key=lambda p: (find_id_from_name(p.name) or p.stem).upper())
    if not files:
        print("[erro] Nenhuma imagem encontrada em", orig_dir); return

    recs = {k: [] for k in ["lbp_orig","hu_orig","glcm_orig","lbp_mask","hu_mask","glcm_mask"]}

    total=sem_label=sem_mask=0
    for img_path in files:
        try:
            iid = find_id_from_name(img_path.name) or img_path.stem.upper()
            if iid not in label_map:
                sem_label += 1
                print(f"[aviso] sem label para {img_path.name} (id detectado: {iid}) — pulando.")
                continue
            y = int(label_map[iid])

            img = load_gray(img_path)
            if args.resize: img = resize_if_needed(img, args.resize)
            if args.clahe:  img = apply_clahe(img)

            mask_path = find_mask_for_id(mask_dir, iid)
            mask_img = imread(mask_path) if mask_path and mask_path.exists() else None
            if mask_img is None:
                sem_mask += 1
            img_m = apply_mask(img, mask_img, thresh=args.mask_thresh, invert=args.invert_mask)

            lbp_o = feat_lbp(img, P=args.lbp_P, R=args.lbp_R, method=args.lbp_method)
            hu_o  = feat_hu(img)
            glcm_o= feat_glcm(img, distances=tuple(args.glcm_dists),
                              angles_deg=tuple(args.glcm_angles), levels=args.glcm_levels)

            lbp_m = feat_lbp(img_m, P=args.lbp_P, R=args.lbp_R, method=args.lbp_method)
            hu_m  = feat_hu(img_m)
            glcm_m= feat_glcm(img_m, distances=tuple(args.glcm_dists),
                              angles_deg=tuple(args.glcm_angles), levels=args.glcm_levels)

            def pack(vec): 
                return {f"f{i}": float(v) for i, v in enumerate(vec)}

            recs["lbp_orig"].append({"id": iid, "label": y, **pack(lbp_o)})
            recs["hu_orig"].append({"id": iid, "label": y, **pack(hu_o)})
            recs["glcm_orig"].append({"id": iid, "label": y, **pack(glcm_o)})

            recs["lbp_mask"].append({"id": iid, "label": y, **pack(lbp_m)})
            recs["hu_mask"].append({"id": iid, "label": y, **pack(hu_m)})
            recs["glcm_mask"].append({"id": iid, "label": y, **pack(glcm_m)})

            total += 1
            if total % 20 == 0:
                print(f"[prog] processadas {total} imagens...")

        except Exception as e:
            print(f"[erro] falha ao processar {img_path.name}: {e}")

    for name, rows in recs.items():
        df = pd.DataFrame(rows)
        if df.empty:
            print(f"[aviso] nada para salvar em {name}.csv")
            continue
        df = df.assign(_k=df["id"].str.extract(r"(\d+)", expand=False).astype(int)) \
               .sort_values(["_k"]).drop(columns="_k")
        df.to_csv(out_dir / f"{name}.csv", index=False)
        print(f"[ok] salvo {out_dir/(name+'.csv')} ({len(df)} linhas)")

    print(f"[resumo] processadas: {total} | sem label: {sem_label} | sem máscara (usou original): {sem_mask}")

if __name__ == "__main__":
    main()
