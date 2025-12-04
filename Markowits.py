import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#from numba import njit, prange


data = pd.read_csv(r"GAN-aplicada-a-finanzas-en-paralelo\data\dataset.csv")


# Parámetros de Markowitz
mu = data.mean()
   
cov = data.cov()



# ----------------------------------------------------
# 2. Función con Numba para generar portafolios aleatorios
# ----------------------------------------------------
@njit(parallel=True)
def simulate_portfolios(mu, cov, n_portfolios):
    n = len(mu)

    # Matrices para guardar resultados
    all_weights = np.zeros((n_portfolios, n))
    returns_p = np.zeros(n_portfolios)
    vol_p = np.zeros(n_portfolios)
    sharpe_p = np.zeros(n_portfolios)

    for i in range(n_portfolios):
        # pesos aleatorios normalizados
        w = np.random.random(n)
        w = w / np.sum(w)

        # métricas
        ret = np.dot(w, mu)
        vol = np.sqrt(np.dot(w.T, np.dot(cov, w)))
        sharpe = ret / vol

        # guardar
        all_weights[i] = w
        returns_p[i] = ret
        vol_p[i] = vol
        sharpe_p[i] = sharpe

    return all_weights, returns_p, vol_p, sharpe_p

# ----------------------------------------------------
# 3. Ejecutar simulación paralelizada
# ----------------------------------------------------
N = 200000  # puedes subirlo a 1,000,000 si tu CPU es potente
weights, rets, vols, sharpes = simulate_portfolios(mu, cov, N)

# ----------------------------------------------------
# 4. Obtener los portafolios óptimos
# ----------------------------------------------------
idx_min_var = np.argmin(vols)
idx_max_sharpe = np.argmax(sharpes)

w_min_var = weights[idx_min_var]
w_max_sharpe = weights[idx_max_sharpe]

tickers = data.columns.tolist()

print("\n--- Portafolio de Mínima Varianza ---")
for t, w in zip(tickers, w_min_var):
    print(f"{t}: {w:.4f}")

print("\n--- Portafolio de Máximo Sharpe ---")
for t, w in zip(tickers, w_max_sharpe):
  print(f"{t}: {w:.4f}")

# ----------------------------------------------------
# 5. Graficar la Frontera Eficiente Aproximada
# ----------------------------------------------------
plt.figure(figsize=(10,6))
plt.scatter(vols, rets, s=3)
plt.scatter(vols[idx_max_sharpe], rets[idx_max_sharpe], color="red", s=50, label="Máximo Sharpe")
plt.scatter(vols[idx_min_var], rets[idx_min_var], color="green", s=50, label="Mínima Varianza")
plt.xlabel("Volatilidad")
plt.ylabel("Rendimiento Esperado")
plt.title("Frontera Eficiente (Simulación Monte Carlo + Numba)")
plt.legend()
plt.show()
