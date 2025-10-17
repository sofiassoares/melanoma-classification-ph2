# 🔬 Classificação Automática de Melanomas e Nevos

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange.svg)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.0+-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **Estudo comparativo de técnicas clássicas de extração de características (LBP, GLCM, Momentos de Hu) e algoritmos de Machine Learning para classificação de lesões dermatoscópicas no dataset PH2.**

Trabalho desenvolvido para a disciplina de **Inteligência Artificial** do curso de **Análise e Desenvolvimento de Sistemas** da **Universidade Federal do Ceará - Campus Itapajé**.

## 📊 Sobre o Projeto

Este trabalho investiga a aplicação de **técnicas clássicas de visão computacional** combinadas com **algoritmos de aprendizado de máquina** para classificação automática de lesões de pele em três categorias:

- 🟢 **Nevos Comuns** (benigno - 80 amostras)
- 🟡 **Nevos Atípicos** (benigno com padrões incomuns - 80 amostras)
- 🔴 **Melanomas** (maligno - 40 amostras)

### 🎯 Principais Resultados

| Rank | Técnica | Variante | Classificador | F1-Score |
|------|---------|----------|---------------|----------|
| 🥇 | **LBP** | Máscara | MLP | **0.6218** |
| 🥈 | LBP | Máscara | SVM | 0.6100 |
| 🥉 | GLCM | Máscara | Gradient Boosting | 0.5958 |
| 4 | GLCM | Máscara | SVM | 0.5936 |
| 5 | Momentos de Hu | Original | Random Forest | 0.5251 |

**Insight principal:** Descritores locais (LBP, GLCM) se beneficiam da segmentação com máscara (+15.6% e +8.6%), enquanto descritores globais (Momentos de Hu) preferem o contexto da imagem completa.

## 🚀 Como Usar

### **1. Clonar o Repositório**
git clone https://github.com/sofiassoares/melanoma-classification-ph2.git
cd melanoma-classification-ph2

### **2. Baixar o Dataset PH2**

⚠️ **IMPORTANTE:** As imagens não estão incluídas neste repositório devido ao tamanho (200 imagens + 200 máscaras = ~50MB).

#### **Download via Kaggle**
https://www.kaggle.com/datasets/spacesurfer/ph2-dataset/data

Extraia o arquivo ZIP
Organize assim:
data/
├── orig/ # Cole aqui as 200 imagens _orig.png
├── mask/ # Cole aqui as 200 máscaras _lesion.png
└── labels.csv # Será gerado pelo script


---

### **4. Executar Pipeline Completo**

O projeto é dividido em **3 scripts sequenciais**. Execute na ordem:

#### **📋 Script 1: Gerar Rótulos (`00_make_labels.py`)**

---

#### **🔬 Script 2: Extrair Características (`01_extract_features.py`)**

---

#### **🤖 Script 3: Treinar e Avaliar Modelos (`02_classify.py`)**


---

### **5. Visualizar Resultados no Dashboard**

streamlit run app.py

Acesse: [**http://localhost:**](http://localhost:)

**Funcionalidades do dashboard:**
- 📊 Tabela interativa com os 36 resultados
- 🔥 Heatmap de F1-Score (6 datasets × 6 modelos)
- 📈 Gráficos comparativos (Original vs Máscara)
- 🖼️ Visualização de matrizes de confusão
- 🔍 Filtros por técnica ou classificador










