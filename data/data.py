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

# -----------------------------------------------
# 3. Descarga de Features Macro y de Mercado
# (vía yfinance: “^TNX” = 10Y, etc.)
# -----------------------------------------------


feature_tickers = ["^IRX", "^TNX", "^TYX", "^VIX"]

data_tickers = yf.download(
    feature_tickers,
    start="2013-01-01",
    end="2025-01-01",
    interval="1d"
)["Close"]

data_tickers.columns = ["YC_3M", "YC_5Y", "YC_10Y", "VIX"]

data_tickers.reindex(returns.index).ffill().bfill()  

features = data_tickers / 100.0 # Convertir a decimal (tasas están en %)

# -----------------------------------------------
# 4. Unificación del dataset: R_h + F_h
# -----------------------------------------------
Dh = pd.concat([returns, features], axis=1).dropna()

print("Dataset listo para CTGAN")
print(Dh.head())
print("\nDimensiones:", Dh.shape)

# Guardar el dataset final en un archivo CSV
output_path = "dataset_ctgan.csv"
Dh.to_csv(output_path, index=True)
