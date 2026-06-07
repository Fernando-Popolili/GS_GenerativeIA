# 🛰️ EpiSat — Previsão de Risco Epidemiológico via Dados Satelitais

> **FIAP · Global Solution 2026 · Engenharia de Software · 4º Ano**  
> Disciplina: GAIE — Generative AI For Engineering

🔗 **Aplicação em funcionamento:** [EpiSat — Hugging Face Spaces](https://huggingface.co/spaces/FernandoPopolili/GS_MODEL_FIAP)  
🔗 **Repositório GitHub:** [GS_GenerativeIA](https://github.com/Fernando-Popolili/GS_GenerativeIA.git)

---

## Contexto do Problema

Doenças como **dengue, malária e leishmaniose** são responsáveis por milhões de casos anuais no Brasil, com impacto direto na saúde pública e na economia. Sua proliferação está diretamente ligada a condições ambientais monitoráveis por satélite — temperatura, umidade, cobertura vegetal e corpos d'água.

O Brasil gasta bilhões anualmente em saúde pública sem sistemas preditivos capazes de antecipar surtos com semanas de antecedência. O **EpiSat** propõe preencher essa lacuna cruzando dados satelitais com histórico epidemiológico para classificar municípios brasileiros em níveis de risco — **Baixo, Médio ou Alto** — antecipando onde e quando agir com medidas preventivas.

**ODS relacionados:** 3 (Saúde e Bem-Estar) · 10 (Redução das Desigualdades) · 11 (Cidades Sustentáveis) · 13 (Ação Climática)

---

## Fonte dos Dados

O dataset foi gerado sinteticamente com base em correlações epidemiológicas reais, simulando dados das seguintes fontes:

| Fonte | Dados simulados |
|-------|----------------|
| Copernicus Sentinel-2 | NDVI, cobertura de água |
| OpenWeather / ERA5 | Temperatura, umidade, precipitação |
| NASA SRTM | Altitude média por município |
| IBGE | População, densidade demográfica |
| SINAN / DATASUS | Casos notificados por semana |

**Características do dataset:**

| Propriedade | Valor |
|-------------|-------|
| Linhas | 7.000 |
| Colunas | 18 |
| Municípios | 140 cidades brasileiras reais |
| Período | 50 semanas epidemiológicas |
| Target | `nivel_risco` — Baixo / Médio / Alto |

> ⚠️ **Nota sobre o dataset sintético:** Datasets artificiais tendem a ter correlações muito diretas entre features e target. Após 5 iterações ajustando níveis de ruído, anomalias por município e remoção de data leakage, o dataset final atingiu um equilíbrio realista.

---

## Metodologia

### 1. Tratamento dos Dados

O dataset passou pelas seguintes etapas de pré-processamento:

- **Verificação de nulos** — nenhum valor ausente encontrado, dataset íntegro
- **Encoding** com `LabelEncoder` nas colunas categóricas (`municipio_codigo`, `nome_municipio`, `uf`) e mapeamento manual ordenado para o target (`Baixo=0`, `Medio=1`, `Alto=2`)
- **Normalização** com `MinMaxScaler` nas 13 colunas numéricas, transformando os valores para o intervalo [0, 1]

---

### 2. Seleção das Melhores Features

Utilizamos três técnicas para identificar as colunas com maior poder preditivo:

| Técnica | O que analisa |
|---------|--------------|
| Random Forest | Importância por redução de impureza (Gini) |
| XGBoost | Importância por ganho de informação |
| PCA | Variância explicada — 2 componentes capturam 99% da informação |

Os dois modelos concordaram no top 3:

| Posição | Feature | Interpretação |
|---------|---------|--------------|
| 1º | `altitude_media_m` | Baixas altitudes favorecem o *Aedes aegypti* |
| 2º | `casos_semana_anterior` | Surtos têm forte autocorrelação temporal |
| 3º | `densidade_demografica` | Cidades densas aceleram a transmissão |

Ao final foram selecionadas as **10 features** com maior importância consistente nos dois modelos para o treinamento final.

---

### 3. Modelos Testados

Foram testados três algoritmos com divisão **70% treino / 30% teste** estratificada:

Acurácia e F1-macro Média por volta de 89%/90%

O **F1-macro** foi adotado como métrica principal por considerar o desempenho nas três classes igualmente, incluindo a mais crítica — **Alto** — onde uma predição errada representa uma falha grave no sistema de alerta epidemiológico.

**Modelo escolhido: XGBoost** — Os três modelos tiveram desempenho praticamente igual (diferença <0,4%). Optei pelo XGBoost por maior familiaridade, velocidade de inferência no Gradio e bom desempenho consistente na classe crítica 'Alto'.

---

## Interpretabilidade com SHAP

A análise SHAP identificou as features mais determinantes nas predições do modelo:

- **`altitude_media_m`** — maior impacto de todas as features. Baixas altitudes aumentam drasticamente o risco, coerente com o comportamento do *Aedes aegypti*, que não sobrevive bem acima de 1.000m
- **`densidade_demografica`** — contexto urbano denso acelera a transmissão vetorial
- **`casos_semana_anterior`** — histórico recente de casos é forte preditor de continuidade do surto
- **`indice_saneamento_pct`** — fator protetor: valores altos de saneamento reduzem consistentemente o risco predito
- **`precipitacao_acumulada_mm`** e **`precip_tendencia`** — chuva acumulada favorece a criação de criadouros do mosquito

---

### 4. Deploy

A aplicação foi desenvolvida com **Gradio** e disponibilizada publicamente. O usuário insere os dados ambientais e socioeconômicos do município e recebe instantaneamente a classificação de risco epidemiológico.

| Arquivo | Conteúdo |
|---------|----------|
| `modelo_episat.pkl` | Modelo XGBoost treinado |
| `scaler.pkl` | MinMaxScaler ajustado nas 13 colunas |
| `mapa_risco.json` | Mapeamento das classes (Baixo / Médio / Alto) |
| `app.py` | Interface Gradio |

---

## Estrutura do Repositório

```
├── modelo.ipynb            # Pipeline completo de ML
├── app.py                  # Aplicação Gradio para deploy
├── episat_dataset.csv      # Dataset (7.000 linhas × 18 colunas)
├── modelo_episat.pkl       # Modelo XGBoost treinado
├── scaler.pkl              # Scaler para normalização
├── mapa_risco.json         # Mapeamento das classes
└── README.md
```

---

## Instruções para Execução

**Pré-requisitos**

```bash
python -m pip install pandas scikit-learn xgboost lightgbm shap matplotlib seaborn joblib gradio
```

**Executar o notebook**
```bash
git clone https://github.com/Fernando-Popolili/GS_GenerativeIA.git
Cd pastaClonada
jupyter notebook modelo.ipynb
```

**Executar a aplicação localmente**
```bash
python app.py
```
Acesse em `http://localhost:7860`

---

## Tecnologias Utilizadas

| Tecnologia | Uso |
|------------|-----|
| Python 3.10+ | Linguagem principal |
| pandas | Manipulação de dados |
| scikit-learn | Pré-processamento e Random Forest |
| XGBoost | Modelo principal de classificação |
| LightGBM | Modelo comparativo |
| SHAP | Interpretabilidade |
| Gradio | Interface de deploy |
| joblib | Serialização do modelo |

---

## Equipe

| Nome | RM |
|------|----|
| Augusto Milreu | RM98245 |
| David Guilherme B. Denunci | RM98603 |
| Fernando Popolili | RM99919 |
| Lucas Palamartschuk de Toledo | RM97913 |
| Matheus Zanardi | RM98832 |

---

*Global Solution 2026 · FIAP · Engenharia de Software*