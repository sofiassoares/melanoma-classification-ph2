
import re
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Análise PH2", layout="wide")

RESULTS_CSV = Path("models/results.csv")
LABELS_CSV  = Path("data/labels.csv")
ORIG_DIR    = Path("data/orig")
MASK_DIR    = Path("data/mask")
CONF_DIR    = Path("models/confusions")
FIGS_DIR    = Path("figs")
FIGS_DIR.mkdir(exist_ok=True)

@st.cache_data
def load_results(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    order_ds = ["lbp_orig","hu_orig","glcm_orig","lbp_mask","hu_mask","glcm_mask"]
    if "dataset" in df.columns:
        df["dataset"] = pd.Categorical(df["dataset"], categories=order_ds, ordered=True)
    return df

@st.cache_data
def load_labels(csv_path: Path) -> pd.DataFrame | None:
    return pd.read_csv(csv_path) if csv_path.exists() else None

def export_plotly(fig, filename: str):
    path = FIGS_DIR / filename
    try:
        fig.write_image(path, scale=2)
        st.success(f"Gráfico salvo em: {path}")
    except Exception as e:
        st.error(f"Erro ao salvar o gráfico: {e}. Dica: instale 'kaleido' →  pip install kaleido")

def find_img_for_id(folder: Path, iid: str):
    hits = sorted(folder.glob(f"{iid}*"))
    return hits[0] if hits else None

def extract_imd_ids(folder: Path) -> list[str]:
    ids = set()
    imdre = re.compile(r"\bIMD\d{3}\b", re.IGNORECASE)
    for p in folder.glob("*"):
        m = imdre.search(p.name)
        if m:
            ids.add(m.group(0).upper())
    return sorted(ids)

if not RESULTS_CSV.exists():
    st.error("Arquivo de resultados não encontrado. Execute os scripts primeiro (02_classify.py).")
    st.stop()

df = load_results(RESULTS_CSV)
if df.empty:
    st.error("models/results.csv está vazio.")
    st.stop()

st.sidebar.title("🔍 Explore os Resultados")
datasets = st.sidebar.multiselect(
    "Filtrar por Dataset",
    options=df["dataset"].cat.categories.tolist(),
    default=list(df["dataset"].cat.categories),
)
classifiers = st.sidebar.multiselect(
    "Filtrar por Classificador",
    options=sorted(df["classifier"].unique()),
    default=sorted(df["classifier"].unique()),
)

fdf = df[df["dataset"].isin(datasets) & df["classifier"].isin(classifiers)].copy()
if fdf.empty:
    st.warning("Nenhuma combinação encontrada para os filtros selecionados.")
    st.stop()

st.markdown("""
<div style="text-align: center;">
  <h1>Análise de Classificação de Lesões de Pele</h1>
  <p><strong>Trabalho Prático - Dataset PH2</strong></p>
  <p>Sofia [Sobrenome] | Outubro de 2025</p>
</div>
""", unsafe_allow_html=True)

with st.expander("Clique para ver o objetivo do projeto"):
    st.markdown("""
**Objetivo:** aplicar e comparar três técnicas de extração de características (LBP, Momentos de Hu e GLCM) no dataset PH2,
construindo seis conjuntos de features (originais × máscaras) e avaliando múltiplos classificadores com métricas padrão.
""")
st.markdown("---")

st.header("Conclusão Principal: O Melhor Modelo")
best_overall = fdf.sort_values("f1_macro_mean", ascending=False).iloc[0]
origem_txt = "máscara da lesão" if "mask" in best_overall["dataset"] else "imagem original"
st.markdown(
    f"A melhor combinação encontrada foi **{best_overall['classifier']}** com **{best_overall['dataset'].split('_')[0].upper()}** "
    f"na **{origem_txt}**."
)

col1, col2, col3, col4 = st.columns(4)
col1.metric("F1-Score", f"{best_overall['f1_macro_mean']:.3f}")
col2.metric("Acurácia", f"{best_overall['accuracy_mean']:.3f}")
col3.metric("Precisão (macro)", f"{best_overall['precision_macro_mean']:.3f}")
if "balanced_accuracy_mean" in fdf.columns:
    col4.metric("Balanced Acc.", f"{best_overall['balanced_accuracy_mean']:.3f}")
else:
    col4.metric("Recall (macro)", f"{best_overall['recall_macro_mean']:.3f}")
st.caption("F1 combina Precisão e Recall; Balanced Accuracy é útil quando há desbalanceamento entre classes.")
st.markdown("---")

st.header("Análise do Dataset PH2")
col1_data, col2_data = st.columns([1, 2])
with col1_data:
    st.markdown("""
O dataset PH2 contém **200 imagens** com 3 classes clínicas:
- **0:** Nevus Comum
- **1:** Nevus Atípico
- **2:** Melanoma

Usamos **F1-macro** como métrica principal devido ao desbalanceamento (40 melanomas vs 160 nevos).
""")
with col2_data:
    labels_df = load_labels(LABELS_CSV)
    if labels_df is not None:
        class_counts = labels_df['label'].value_counts().sort_index().reset_index()
        class_counts.columns = ['label', 'count']
        class_counts['label'] = class_counts['label'].map({0: 'Nevus Comum', 1: 'Nevus Atípico', 2: 'Melanoma'})
        fig_dist = px.bar(class_counts, x='label', y='count', text_auto=True,
                          title="Distribuição de Classes (PH2)",
                          labels={'label': 'Diagnóstico Clínico', 'count': 'Amostras'})
        st.plotly_chart(fig_dist, use_container_width=True)
        if st.button("Exportar Distribuição (PNG)"):
            export_plotly(fig_dist, "distribuicao_classes.png")
    else:
        st.warning(f"Arquivo {LABELS_CSV} não encontrado.")
st.markdown("---")

st.header("Análise Detalhada dos Resultados")
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Visão Geral", "Classificadores", "Impacto da Máscara", "Melhores por Técnica", "Matrizes de Confusão"
])

with tab1:
    st.subheader("Comparativo Geral (F1-macro)")
    st.info("**Como interpretar:** O **mapa de calor** oferece uma visão rápida dos padrões de desempenho (tons claros = melhor). O **gráfico de barras** é ideal para comparar os valores exatos de cada combinação.")
    view_type = st.selectbox("Visualização", ["Mapa de Calor", "Barras Agrupadas"], label_visibility="collapsed")
    if view_type == "Mapa de Calor":
        heat_pivot = fdf.pivot_table(index="dataset", columns="classifier", values="f1_macro_mean")
        fig_heat = px.imshow(
            heat_pivot, text_auto=".3f", aspect="auto",
            color_continuous_scale=px.colors.sequential.Viridis,
            labels=dict(x="Classificador", y="Dataset", color="F1")
        )
        fig_heat.update_layout(title="F1-macro por (Dataset × Classificador)")
        st.plotly_chart(fig_heat, use_container_width=True)
        if st.button("Exportar Heatmap (PNG)"):
            export_plotly(fig_heat, "heatmap_f1.png")
    else:
        fig_bar = px.bar(
            fdf, x="dataset", y="f1_macro_mean", color="classifier", barmode="group", text_auto=".2f",
            labels={"dataset": "Dataset", "f1_macro_mean": "F1-macro", "classifier": "Classificador"},
            title="F1-macro por Dataset (Barras Agrupadas)"
        )
        st.plotly_chart(fig_bar, use_container_width=True)
        if st.button("Exportar Barras (PNG)"):
            export_plotly(fig_bar, "barras_f1.png")
    st.info("Padrão claro: **GLCM** e **LBP** em **máscara** tendem a render melhor.")

with tab2:
    st.subheader("Consistência dos Classificadores")
    st.info(""" **Como interpretar:** Cada ponto representa o desempenho de um classificador em um dos seis testes. Um bom classificador terá seus pontos **agrupados e em uma posição alta**, indicando um desempenho consistentemente bom. Pontos muito espalhados indicam instabilidade. """)
    fig_strip = px.strip(
        fdf, x="classifier", y="f1_macro_mean", color="classifier",
        labels={"classifier": "Classificador", "f1_macro_mean": "F1 por teste"},
        title="Dispersão do F1 por Classificador (6 pontos por classificador)"
    )
    st.plotly_chart(fig_strip, use_container_width=True)
    if st.button("Exportar Consistência (PNG)"):
        export_plotly(fig_strip, "consistencia_classificadores.png")

with tab3:
    st.subheader("Qual o Impacto de Usar a Máscara de Segmentação?") 
    st.info("**Contexto:** A hipótese deste teste é que focar a análise apenas na área da lesão (a máscara) pode eliminar ruídos (pele saudável, pelos) e gerar resultados mais precisos do que usar a imagem inteira.")
    tmp = fdf.copy()
    tmp["extrator"] = tmp["dataset"].str.extract(r"^(lbp|hu|glcm)", expand=False)
    tmp["origem"]   = tmp["dataset"].str.extract(r"(orig|mask)$", expand=False)
    agg = tmp.groupby(["extrator","origem"], observed=True)["f1_macro_mean"].mean().reset_index()
    fig_comp = px.bar(
        agg, x="extrator", y="f1_macro_mean", color="origem", barmode="group", text_auto=".3f",
        labels={"extrator": "Extrator", "origem": "Fonte", "f1_macro_mean": "F1-médio"},
        title="F1 médio por Extrator (Imagem Original vs Máscara)"
    )
    st.plotly_chart(fig_comp, use_container_width=True)
    if st.button("Exportar Orig×Mask (PNG)"):
        export_plotly(fig_comp, "comparativo_orig_mask.png")

with tab4:
    st.subheader("Melhor Classificador por Dataset")
    st.info("**Como interpretar:** A tabela responde à pergunta: 'Se eu fosse obrigado a usar uma técnica específica (como LBP), qual classificador seria a minha melhor escolha?'. Tons de verde mais escuros indicam melhor desempenho.")
    best_per_ds = fdf.loc[fdf.groupby("dataset")["f1_macro_mean"].idxmax()].sort_values("dataset")
    numeric_cols = [c for c in ["accuracy_mean","balanced_accuracy_mean","precision_macro_mean","recall_macro_mean","f1_macro_mean"] if c in best_per_ds.columns]
    st.dataframe(
        best_per_ds.style.background_gradient(cmap='Greens', subset=numeric_cols).format("{:.4f}", subset=numeric_cols),
        use_container_width=True, hide_index=True
    )
    st.download_button("⬇️ Baixar CSV (Melhores por Dataset)", best_per_ds.to_csv(index=False).encode("utf-8"), file_name="best_per_dataset.csv")

with tab5:
    st.subheader("Matrizes de Confusão (80/20 hold-out)")

    st.info("""
**Como interpretar a matriz:**
- As linhas representam as **classes reais** e as colunas, as **classes preditas** pelo modelo.
- A **diagonal principal** mostra os acertos (quanto mais claro ou intenso, melhor).
- Valores fora da diagonal são **erros de confusão** — por exemplo, uma lesão **Melanoma (2)** classificada como **Nevus Atípico (1)**.
- Quando normalizada, cada linha soma 1.0, o que permite comparar a taxa de acerto por classe, mesmo em datasets desbalanceados.
    """)

    ds_sel  = st.selectbox("Dataset", options=df["dataset"].cat.categories.tolist(), index=5)
    clf_sel = st.selectbox("Classificador", options=sorted(df["classifier"].unique()))

    img_path = CONF_DIR / f"{ds_sel}__{clf_sel}.png"
    if img_path.exists():
        st.image(str(img_path), caption=f"Matriz de Confusão — {ds_sel} × {clf_sel}", use_container_width=True)
    else:
        st.info(f"Não encontrei {img_path}. Rode 02_classify.py para gerar.")

st.markdown("---")

st.header("Explorador de Imagens (Original × Máscara)")
ids = extract_imd_ids(ORIG_DIR)
if ids:
    iid = st.selectbox("Selecione um ID (IMD###)", ids)
    orig_p = find_img_for_id(ORIG_DIR, iid)
    mask_p = find_img_for_id(MASK_DIR, iid)
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Imagem Original")
        if orig_p: st.image(str(orig_p), caption=orig_p.name, use_container_width=True)
        else: st.info("Original não encontrada.")
    with c2:
        st.subheader("Máscara da Lesão")
        if mask_p: st.image(str(mask_p), caption=mask_p.name, use_container_width=True)
        else: st.info("Máscara não encontrada.")

st.markdown("---")
st.download_button("⬇Baixar CSV Filtrado", fdf.to_csv(index=False).encode("utf-8"), file_name="results_filtrado.csv")
st.caption("Dica: esses arquivos PNG exportados em FIGS/ já estão prontos para inserir no LaTeX.")
