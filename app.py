import gradio as gr
import pandas as pd
import joblib

modelo = joblib.load("modelo_episat.pkl")
scaler = joblib.load("scaler.pkl")   # scaler treinado nas 13 colunas

# A ordem exata das 13 colunas (deve ser a mesma usada no treinamento)
COLUNAS_13 = [
    "temperatura_media_c", "temperatura_max_c", "umidade_relativa_pct",
    "precipitacao_acumulada_mm", "ndvi_medio", "cobertura_agua_pct",
    "altitude_media_m", "populacao_exposta", "densidade_demografica",
    "indice_saneamento_pct", "casos_semana_anterior",
    "precip_tendencia", "temp_tendencia"
]

# As 10 features que o modelo realmente usa (subconjunto)
FEATURES_MODELO = [
    "altitude_media_m",
    "casos_semana_anterior",
    "densidade_demografica",
    "indice_saneamento_pct",
    "precipitacao_acumulada_mm",
    "precip_tendencia",
    "umidade_relativa_pct",
    "cobertura_agua_pct",
    "ndvi_medio",
    "temperatura_media_c",
]

def prever(temp_media, temp_max, umidade, precipitacao, ndvi, cobertura_agua,
           altitude, populacao_exposta, densidade, saneamento,
           casos_anterior, precip_tendencia, temp_tendencia):
    # Monta DataFrame com as 13 colunas (na ordem correta)
    dados = pd.DataFrame([{
        "temperatura_media_c": temp_media,
        "temperatura_max_c": temp_max,
        "umidade_relativa_pct": umidade,
        "precipitacao_acumulada_mm": precipitacao,
        "ndvi_medio": ndvi,
        "cobertura_agua_pct": cobertura_agua,
        "altitude_media_m": altitude,
        "populacao_exposta": populacao_exposta,
        "densidade_demografica": densidade,
        "indice_saneamento_pct": saneamento,
        "casos_semana_anterior": casos_anterior,
        "precip_tendencia": precip_tendencia,
        "temp_tendencia": temp_tendencia,
    }])
    
    # Normaliza as 13 colunas com o scaler salvo
    dados[COLUNAS_13] = scaler.transform(dados[COLUNAS_13])
    
    # Seleciona apenas as 10 features que o modelo usa
    dados_modelo = dados[FEATURES_MODELO]
    
    # Predição
    pred = modelo.predict(dados_modelo)[0]
    proba = modelo.predict_proba(dados_modelo)[0]
    
    resultado = {0: "Baixo", 1: "Medio", 2: "Alto"}[pred]
    icons = {"Alto": "🔴 Alto", "Medio": "🟡 Médio", "Baixo": "🟢 Baixo"}
    
    # (Opcional) prints para debug
    print(f"Predição: {pred} -> {resultado}")
    print(f"Probabilidades: Baixo={proba[0]:.2f}, Médio={proba[1]:.2f}, Alto={proba[2]:.2f}")
    
    return f"Nível de Risco: {icons[resultado]}"

# Interface com 13 inputs
interface = gr.Interface(
    fn=prever,
    inputs=[
        gr.Number(label="Temperatura Média (°C)"),
        gr.Number(label="Temperatura Máxima (°C)"),
        gr.Number(label="Umidade Relativa (%)"),
        gr.Number(label="Precipitação Acumulada (mm)"),
        gr.Number(label="NDVI Médio"),
        gr.Number(label="Cobertura de Água (%)"),
        gr.Number(label="Altitude Média (m)"),
        gr.Number(label="População Exposta"),
        gr.Number(label="Densidade Demográfica (hab/km²)"),
        gr.Number(label="Índice de Saneamento (%)"),
        gr.Number(label="Casos Semana Anterior (taxa/100k)"),
        gr.Number(label="Tendência de Precipitação (média 3 sem.)"),
        gr.Number(label="Tendência de Temperatura"),
    ],
    outputs="text",
    title="🛰️ EpiSat",
    description="Previsão de risco epidemiológico por município · FIAP Global Solution 2026",
    examples=[
        # Exemplo 1 (Baixo) – valores brutos
        [18, 25, 40, 10, 0.15, 1, 1200, 5000, 40, 95, 8, 8, 0.5],
        # Exemplo 2 (Médio)
        [24.5, 29.0, 65, 45, 0.42, 8, 550, 15000, 180, 78, 45, 35, 0.9],
        # Exemplo 3 (Alto)
        [32, 38, 85, 140, 0.75, 25, 50, 50000, 900, 35, 180, 130, 2.5],
    ]
)

if __name__ == "__main__":
    interface.launch()