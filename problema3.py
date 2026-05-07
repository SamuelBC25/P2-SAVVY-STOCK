"""
problema3.py
Practica 2 — PROBLEMA 3: Maximizar retorno con riesgo (varianza) <= 20%

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
    Desigualdad:
        g1(x) = x^T*COV*x - 0.20 <= 0   (riesgo <= 20%)
    Limites (manejados por lb/ub en main.py):
        x_i <= 0.4   y   x_i >= 0

Manejador de restricciones: PENALIZACION EXTERIOR
    P(x)       = sum[max(g_i, 0)]^2  +  sum[h_j]^2
    aptitud(x) = f(x) - lambda_P * P(x)    [maximizacion]
======================================================================
"""

import numpy as np

# == Nombre del problema =============================================
NOMBRE = "Problema 3 - Maximizar retorno (riesgo <= 20%)"

# == Datos ============================================================
ACCIONES = ["BB",  "LOP",  "ILI",  "HEAL", "QUI",  "AUA"]
RETORNOS = np.array([0.20,  0.42,   1.00,   0.50,   0.46,  0.30])

# Matriz de varianzas-covarianzas (6x6 simetrica)
COV = np.array([
    #  BB      LOP     ILI     HEAL     QUI     AUA
    [ 0.032,  0.005,  0.030, -0.031, -0.027,  0.010],  # BB
    [ 0.005,  0.100,  0.085, -0.070, -0.050,  0.020],  # LOP
    [ 0.030,  0.085,  0.333, -0.110, -0.020,  0.042],  # ILI
    [-0.031, -0.070, -0.110,  0.125,  0.050, -0.060],  # HEAL
    [-0.027, -0.050, -0.020,  0.050,  0.065, -0.020],  # QUI
    [ 0.010,  0.020,  0.042, -0.060, -0.020,  0.080],  # AUA
])

RIESGO_MAX = 0.20       # restriccion: varianza <= 20%

# == Limites de las variables de decision ============================
Nvar = 6
lb   = np.zeros(Nvar)
ub   = np.full(Nvar, 0.4)

# == Coeficiente de penalizacion exterior ============================
lambda_P = 50.0


# ====================================================================
# FUNCION DE PENALIZACION  P(x)
#
#   Rd = [g1(x)]    g1(x) = x^T*COV*x - 0.20  <= 0
#   Ri = [h1(x)]    h1(x) = sum(x_i) - 1        = 0
#
#   P(x) = sum[max(Rd, 0)]^2  +  sum[Ri]^2
# ====================================================================
def calcular_P(x: np.ndarray) -> float:
    # Restricciones de desigualdad g_i <= 0
    g1    = float(x @ COV @ x) - RIESGO_MAX            # g1 <= 0
    Rd    = np.array([g1])
    pen_g = np.sum(np.maximum(Rd, 0.0) ** 2)

    # Restricciones de igualdad h_j = 0
    h1    = np.sum(x) - 1.0                             # h1 = 0
    Ri    = np.array([h1])
    pen_h = np.sum(Ri ** 2)

    return pen_g + pen_h


# ====================================================================
# FUNCION DE APTITUD PENALIZADA
#   aptitud(x) = f(x) - lambda_P * P(x)
# ====================================================================
def calcular_aptitud(x: np.ndarray) -> float:
    f_x = float(np.dot(RETORNOS, x))
    P_x = calcular_P(x)
    return f_x - lambda_P * P_x


# ====================================================================
# METRICAS FINALES DE LA SOLUCION
# ====================================================================
def calcular_metricas(pesos: np.ndarray) -> dict:
    return {
        "retorno": float(np.dot(RETORNOS, pesos)),
        "riesgo":  float(pesos @ COV @ pesos),
        "P_x":     calcular_P(pesos),
    }
