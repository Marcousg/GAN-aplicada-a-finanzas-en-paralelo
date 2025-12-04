import pandas as pd
import numpy as np
import yfinance as yf

# -----------------------------------------------
# 1. Activos de tu portafolio
# -----------------------------------------------
tickers = ["^GSPC", "GC=F", "NVDA", "JPM", "MELI", "GOOG", "JNJ", "RTX", "WMT"]

# Descarga precios ajustados (diarios, 10 años)
data_assets = yf.download(
    tickers,
    start="2013-01-01",
    end="2025-01-01",
    interval="1d"
)["Close"]

# Renombra columnas para evitar caracteres raros
data_assets.columns = [
    "SP500", "GOLD", "NVDA", "JPM", "MELI", "GOOG", "JNJ", "RTX", "WMT"
]

# -----------------------------------------------
# 2. Calcular retornos logarítmicos
# -----------------------------------------------
returns = np.log(data_assets).diff().dropna()
returns.columns = [c + "_ret" for c in returns.columns]

# Guardar el dataset final en un archivo CSV
output_path = "dataset.csv"
returns.to_csv(output_path, index=True)
