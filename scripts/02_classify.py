
import argparse, json, warnings, time
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (confusion_matrix, ConfusionMatrixDisplay)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

DATASETS = [
    "lbp_orig.csv", "hu_orig.csv", "glcm_orig.csv",
    "lbp_mask.csv", "hu_mask.csv", "glcm_mask.csv",
]

CLASSES = [0, 1, 2]

def load_xy(csv_path: Path):
    df = pd.read_csv(csv_path)
    X = df.filter(like="f").values
    y = df["label"].values.astype(int)
    return X, y

def make_models():
    return {
        "KNN": Pipeline([("scaler", StandardScaler()), ("clf", KNeighborsClassifier(n_neighbors=5))]),
        "SVM": Pipeline([("scaler", StandardScaler()), ("clf", SVC(kernel="rbf", C=2.0, gamma="scale"))]),
        "DecisionTree": DecisionTreeClassifier(random_state=42),
        "RandomForest": RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1),
        "MLP": Pipeline([("scaler", StandardScaler()), ("clf", MLPClassifier(hidden_layer_sizes=(64,), max_iter=800, random_state=42))]),
        "GradientBoosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
    }

def evaluate_cv(X, y, model, cv_splits=5, folds_out: Path | None = None, tag: str = ""):
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)
    scoring = {
        "accuracy": "accuracy",
        "balanced_accuracy": "balanced_accuracy",
        "precision": "precision_macro",
        "recall": "recall_macro",
        "f1": "f1_macro",
    }
    res = cross_validate(model, X, y, cv=cv, scoring=scoring, n_jobs=-1, return_estimator=False)
    if folds_out:
        folds_out.parent.mkdir(parents=True, exist_ok=True)
        per_fold = pd.DataFrame({
            "fold": np.arange(cv_splits) + 1,
            "accuracy": res["test_accuracy"],
            "balanced_accuracy": res["test_balanced_accuracy"],
            "precision_macro": res["test_precision"],
            "recall_macro": res["test_recall"],
            "f1_macro": res["test_f1"],
        })
        per_fold.to_csv(folds_out, index=False)

    out = {k.replace("test_", ""): (v.mean(), v.std()) for k, v in res.items() if k.startswith("test_")}
    return out

def save_confusion(X, y, model, title, outpath: Path, normalize=False):
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    model.fit(Xtr, ytr)
    yhat = model.predict(Xte)
    cm = confusion_matrix(yte, yhat, labels=CLASSES, normalize="true" if normalize else None)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASSES)
    disp.plot(values_format=".2f" if normalize else "d", cmap="Blues")
    plt.title(title + (" (normalizada)" if normalize else ""))
    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outpath, bbox_inches="tight", dpi=180)
    plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features_dir", default="features")
    ap.add_argument("--out_dir", default="models")
    ap.add_argument("--cv", type=int, default=5)
    ap.add_argument("--normalize-confusion", action="store_true", help="Normaliza matrizes de confusão por linha")
    args = ap.parse_args()

    feats_dir = Path(args.features_dir)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    models = make_models()
    rows = []

    print(f"Iniciando avaliação com {len(models)} classificadores...")
    t0_all = time.time()
    for ds in DATASETS:
        print(f"\nProcessando dataset: {ds}")
        X, y = load_xy(feats_dir / ds)
        for name, mdl in models.items():
            t0 = time.time()
            print(f"  Avaliando: {name}...")
            scores = evaluate_cv(
                X, y, mdl, cv_splits=args.cv,
                folds_out=out_dir / "folds" / f"{ds.replace('.csv','')}__{name}.csv",
                tag=f"{ds}__{name}"
            )
            rows.append({
                "dataset": ds.replace(".csv",""),
                "classifier": name,
                "accuracy_mean": round(scores["accuracy"][0], 4),
                "balanced_accuracy_mean": round(scores["balanced_accuracy"][0], 4),
                "precision_macro_mean": round(scores["precision"][0], 4),
                "recall_macro_mean": round(scores["recall"][0], 4),
                "f1_macro_mean": round(scores["f1"][0], 4),
            })
            save_confusion(X, y, mdl,
                           f"{ds.replace('.csv','')} - {name}",
                           out_dir / "confusions" / f"{ds.replace('.csv','')}__{name}.png",
                           normalize=args.normalize_confusion)
            print(f"    OK ({time.time()-t0:.1f}s)")

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "results.csv", index=False)

    with open(out_dir / "settings.json", "w", encoding="utf-8") as f:
        json.dump({"cv": args.cv, "models": list(models.keys()), "datasets": DATASETS}, f, ensure_ascii=False, indent=2)
    
    print("\n" + "="*34)
    print(f"[ok] Avaliação concluída em {time.time()-t0_all:.1f}s")
    print(f"[ok] Resultados: {out_dir/'results.csv'}")
    print(f"[ok] Matrizes:   {out_dir/'confusions'}")
    print(f"[ok] Folds CSV:  {out_dir/'folds'}")
    print("="*34)

if __name__ == "__main__":
    main()
