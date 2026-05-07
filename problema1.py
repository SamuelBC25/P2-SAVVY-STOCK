"""
problema1.py
Practica 2 — PROBLEMA 1: Maximizar retorno esperado (sin considerar riesgo)

======================================================================
Formulacion matematica
-----------------------
Variables de decision:
    x = [x_BB, x_LOP, x_ILI, x_HEAL, x_QUI, x_AUA]
    Cada x_i en [0, 0.4]

Funcion objetivo:
    max  f(x) = sum(r_i * x_i)
    r = [BB=0.20, LOP=0.42, ILI=1.00, HEAL=0.50, QUI=0.46, AUA=0.30]

Restricciones:
    Igualdad  -> h1(x) = sum(x_i) - 1 = 0
    Desigualdad (manejadas por lb/ub en main.py):
                 x_i <= 0.4   y   x_i >= 0

Manejador de restricciones: PENALIZACION EXTERIOR (Dr. Molina, diap. 28)
    P(x)       = sum[max(g_i, 0)]^2  +  sum[h_j]^2
    aptitud(x) = f(x) - lambda_P * P(x)    [maximizacion]
======================================================================
"""

import numpy as np

# == Nombre del problema =============================================
NOMBRE = "Problema 1 - Maximizar retorno (sin riesgo)"

# == Datos ============================================================
ACCIONES = ["BB",  "LOP",  "ILI",  "HEAL", "QUI",  "AUA"]
RETORNOS = np.array([0.20,  0.42,   1.00,   0.50,   0.46,  0.30])

# == Limites de las variables de decision ============================
Nvar = 6
lb   = np.zeros(Nvar)       # x_i >= 0
ub   = np.full(Nvar, 0.4)   # x_i <= 0.40

# == Coeficiente de penalizacion exterior ============================
lambda_P = 10.0


# ====================================================================
# FUNCION DE PENALIZACION  P(x)
#
#   Rd = []         sin g_i extras (lb/ub gestionados en main.py)
#   Ri = [h1(x)]   h1(x) = sum(x_i) - 1 = 0
#
#   P(x) = sum[max(Rd, 0)]^2  +  sum[Ri]^2
# ====================================================================
def calcular_P(x: np.ndarray) -> float:
    # Restricciones de desigualdad g_i <= 0
    Rd    = np.array([])                     # sin restricciones extra
    pen_g = np.sum(np.maximum(Rd, 0.0) ** 2)

    # Restricciones de igualdad h_j = 0
    h1    = np.sum(x) - 1.0                  # h1(x) = sum(x_i) - 1 = 0
    Ri    = np.array([h1])
    pen_h = np.sum(Ri ** 2)

    return pen_g + pen_h


# ====================================================================
# FUNCION DE APTITUD PENALIZADA
#   aptitud(x) = f(x) - lambda_P * P(x)
# ====================================================================
def calcular_aptitud(x: np.ndarray) -> float:
    f_x = float(np.dot(RETORNOS, x))        # retorno esperado
    P_x = calcular_P(x)
    return f_x - lambda_P * P_x


# ====================================================================
# METRICAS FINALES DE LA SOLUCION
# Devuelve un dict con los valores relevantes para reportar
# ====================================================================
def calcular_metricas(pesos: np.ndarray) -> dict:
    return {
        "retorno": float(np.dot(RETORNOS, pesos)),
        "riesgo":  None,                     # no aplica en P1
        "P_x":     calcular_P(pesos),
    }
