import gradio as gr
import pandas as pd
import joblib

modelo  = joblib.load("modelo_episat.pkl")
encoder = joblib.load("label_encoder.pkl")

# Exatamente as 10 features usadas no treino, na mesma ordem
FEATURES = [
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

def prever(altitude, casos_ant, densidade, saneamento,
           precipitacao, precip_tend, umidade, agua, ndvi, temp):

    dados = pd.DataFrame([{
        "altitude_media_m":          altitude,
        "casos_semana_anterior":     casos_ant,
        "densidade_demografica":     densidade,
        "indice_saneamento_pct":     saneamento,
        "precipitacao_acumulada_mm": precipitacao,
        "precip_tendencia":          precip_tend,
        "umidade_relativa_pct":      umidade,
        "cobertura_agua_pct":        agua,
        "ndvi_medio":                ndvi,
        "temperatura_media_c":       temp,
    }])

    pred      = modelo.predict(dados)
    resultado = encoder.inverse_transform(pred)[0]

    icons = {"Alto": "🔴 Alto", "Medio": "🟡 Médio", "Baixo": "🟢 Baixo"}
    return f"Nível de Risco: {icons.get(resultado, resultado)}"


interface = gr.Interface(
    fn=prever,
    inputs=[
        gr.Number(label="Altitude Média (m)"),
        gr.Number(label="Casos Semana Anterior (taxa/100k)"),
        gr.Number(label="Densidade Demográfica (hab/km²)"),
        gr.Number(label="Índice de Saneamento (%)"),
        gr.Number(label="Precipitação Acumulada (mm)"),
        gr.Number(label="Tendência de Precipitação (média 3 sem.)"),
        gr.Number(label="Umidade Relativa (%)"),
        gr.Number(label="Cobertura de Água (%)"),
        gr.Number(label="NDVI Médio"),
        gr.Number(label="Temperatura Média (°C)"),
    ],
    outputs="text",
    title="🛰️ EpiSat",
    description="Previsão de risco epidemiológico por município · FIAP Global Solution 2026",
    examples=[
        [50,  20, 80,  90, 10,  8,  65, 3, 0.25, 22],   # Baixo
        [200, 80, 300, 60, 60, 55,  75, 10, 0.45, 27],   # Médio
        [30, 200, 800, 30, 150, 130, 85, 25, 0.70, 32],  # Alto
    ]
)

interface.launch()