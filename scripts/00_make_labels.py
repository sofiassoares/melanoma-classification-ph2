
import argparse
import re
from pathlib import Path
import pandas as pd

IMD_RE = re.compile(r"\bIMD\d{3}\b", re.IGNORECASE)
IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def parse_pipe_table(txt_path: Path) -> pd.DataFrame:
    rows = []
    with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if "||" not in line:
                continue

            parts = [c.strip() for c in line.strip("|").split("||")]
            if len(parts) < 3:
                continue

            name_col = parts[0]
            clin_col = parts[2]

            m = IMD_RE.search(name_col)
            if not m:
                continue

            iid = m.group(0).upper()

            mnum = re.search(r"\b([012])\b", clin_col)
            if mnum:
                label = int(mnum.group(1))
            else:
                clin_num = clin_col.strip()
                label = int(clin_num) if clin_num in {"0", "1", "2"} else 0  # fallback conservador

            rows.append({"id": iid, "label": label})

    df = pd.DataFrame(rows).drop_duplicates(subset=["id"])
    if not df.empty:
        df = df.assign(_k=df["id"].str.extract(r"(\d+)", expand=False).astype(int)).sort_values("_k").drop(columns="_k")
    return df


def parse_excel(meta_path: Path) -> pd.DataFrame:
    df = pd.read_excel(meta_path)
    id_col = next((c for c in df.columns if "name" in c.lower()
                   or "image" in c.lower() or "lesion" in c.lower() or c.lower() == "id"), df.columns[0])
    clin_col = next((c for c in df.columns if ("clinical" in c.lower() and "diagn" in c.lower())
                     or c.lower().startswith("clinical")), df.columns[-1])
    clin = df[clin_col].astype(str).str.extract(r"\b([012])\b", expand=False).fillna("0").astype(int)
    ids = df[id_col].astype(str).str.extract(IMD_RE, expand=False).str.upper()
    out = pd.DataFrame({"id": ids, "label": clin}).dropna(subset=["id"]).drop_duplicates(subset=["id"])
    return out


def read_meta(meta_path: Path) -> pd.DataFrame:
    if meta_path.suffix.lower() in [".xlsx", ".xls"]:
        return parse_excel(meta_path)
    return parse_pipe_table(meta_path)


def collect_ids_from_dir(images_dir: Path) -> set[str]:
    ids = set()
    if not images_dir.exists():
        return ids
    for p in images_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            m = IMD_RE.search(p.name)
            if m:
                ids.add(m.group(0).upper())
    return ids


def main():
    ap = argparse.ArgumentParser(description="Gerar data/labels.csv a partir do PH2_dataset.txt/.xlsx")
    ap.add_argument("--meta", required=True, help="Caminho para PH2_dataset.txt (formato '||') ou .xlsx")
    ap.add_argument("--out", default="data/labels.csv", help="Arquivo de saída (padrão: data/labels.csv)")
    ap.add_argument("--filter_ids_dir", default=None,
                    help="(Opcional) Pasta com imagens (ex.: data/orig) para filtrar somente IDs existentes")
    args = ap.parse_args()

    meta = Path(args.meta)
    df = read_meta(meta)

    if df.empty:
        print("[erro] Não encontrei linhas válidas com IMD### no arquivo informado.")
        return

    if args.filter_ids_dir:
        present = collect_ids_from_dir(Path(args.filter_ids_dir))
        before = len(df)
        df = df[df["id"].isin(present)]
        print(f"[info] Filtro por {args.filter_ids_dir}: {before} -> {len(df)} linhas")

    if not set(df["label"].unique()).issubset({0, 1, 2}):
        print("[aviso] Encontrei rótulos fora de {0,1,2}; normalizando.")
        df["label"] = df["label"].clip(lower=0, upper=2).astype(int)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    print(f"[ok] Gerado {out_path} com {len(df)} linhas")
    print("Contagem por label (0=nevus,1=atypical,2=melanoma):")
    print(df["label"].value_counts().sort_index())
    print("Amostra:")
    print(df.head(12).to_string(index=False))


if __name__ == "__main__":
    main()
