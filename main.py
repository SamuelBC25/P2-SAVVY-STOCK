"""
main.py
Practica 2 — SAVVY STOCK SELECTIONS
AG con Penalizacion Exterior

Cada problema (problema1.py, problema2.py, problema3.py) solo define:
    - lb, ub           limites de las variables
    - lambda_P         coeficiente de penalizacion
    - calcular_aptitud(x)  funcion objetivo penalizada
    - calcular_P(x)        funcion de penalizacion
    - calcular_metricas(x) metricas finales de la solucion

Estructura del proyecto:
    main.py       <- AG completo + orquestador (este archivo)
    problema1.py  <- datos, F_P y aptitud del Problema 1
    problema2.py  <- datos, F_P y aptitud del Problema 2
    problema3.py  <- datos, F_P y aptitud del Problema 3
"""

import numpy as np
import csv
import problema1
import problema2
import problema3

# == Parametros globales del AG ========================================
Np      = 50       # Tamano de poblacion
GenMax  = 200      # Numero maximo de generaciones
Pc      = 0.9      # Probabilidad de cruzamiento
Pm      = 1 / 6    # Probabilidad de mutacion por gen (aprox 1/Nvar)
Nc      = 5        # Indice de distribucion SBX
Nm      = 40       # Indice de distribucion mutacion polinomial

NUM_EJECUCIONES = 10


# ====================================================================
# PASO 1 — GENERACION DE POBLACION INICIAL
# ====================================================================
def generar_poblacion_inicial(lb: np.ndarray, ub: np.ndarray) -> np.ndarray:
    """
    Genera Np individuos con distribucion uniforme en [lb, ub].
    Normaliza cada individuo para que parta con sum(x) ~ 1,
    acelerando la convergencia hacia la region factible.
    """
    Nvar      = len(lb)
    poblacion = np.zeros((Np, Nvar))
    for i in range(Np):
        x = lb + np.random.rand(Nvar) * (ub - lb)
        s = x.sum()
        if s > 0:
            x = x / s
        x = np.clip(x, lb, ub)
        poblacion[i] = x
    return poblacion


# ====================================================================
# PASO 2 / 7 — EVALUACION EN F_P (aptitud penalizada)
# ====================================================================
def evaluar_poblacion(poblacion: np.ndarray, problema) -> np.ndarray:
    """Evalua todos los individuos usando la aptitud del problema activo."""
    return np.array([problema.calcular_aptitud(poblacion[i])
                     for i in range(len(poblacion))])


# ====================================================================
# PASO 4 — SELECCION: torneo determinista binario
# ====================================================================
def seleccion_torneo(poblacion: np.ndarray,
                     aptitudes: np.ndarray) -> np.ndarray:
    """
    Torneo determinista binario:
    Compara dos individuos aleatorios y retorna al ganador.
    """
    perm1     = np.random.permutation(Np)
    perm2     = np.random.permutation(Np)
    ganadores = np.where(aptitudes[perm1] >= aptitudes[perm2],
                         perm1, perm2)
    return poblacion[ganadores]


# ====================================================================
# PASO 5 — CRUZAMIENTO SBX CON LIMITES
# ====================================================================
def cruzamiento_sbx(parents: np.ndarray,
                    lb: np.ndarray,
                    ub: np.ndarray) -> np.ndarray:
    """
    Simulated Binary Crossover (SBX) con control de limites lb/ub.
    Parametro de distribucion: Nc
    """
    Nvar  = len(lb)
    Hijos = np.zeros((Np, Nvar))

    for i in range(0, Np - 1, 2):
        if np.random.rand() <= Pc:
            U = np.random.rand()
            for j in range(Nvar):
                P1, P2 = parents[i, j], parents[i + 1, j]

                if abs(P2 - P1) < 1e-10:
                    Hijos[i, j] = P1
                    Hijos[i + 1, j] = P2
                    continue

                if P1 > P2:
                    P1, P2 = P2, P1

                beta  = 1.0 + (2.0 / (P2 - P1)) * min(P1 - lb[j], ub[j] - P2)
                alpha = 2.0 - abs(beta) ** (-(Nc + 1))

                if U <= 1.0 / alpha:
                    beta_c = (U * alpha) ** (1.0 / (Nc + 1))
                else:
                    beta_c = (1.0 / (2.0 - U * alpha)) ** (1.0 / (Nc + 1))

                Hijos[i,     j] = np.clip(0.5 * ((P1 + P2) - beta_c * abs(P2 - P1)), lb[j], ub[j])
                Hijos[i + 1, j] = np.clip(0.5 * ((P1 + P2) + beta_c * abs(P2 - P1)), lb[j], ub[j])
        else:
            Hijos[i]     = parents[i].copy()
            Hijos[i + 1] = parents[i + 1].copy()

    if Np % 2 == 1:
        Hijos[-1] = parents[-1].copy()

    return Hijos


# ====================================================================
# PASO 6 — MUTACION POLINOMIAL CON LIMITES
# ====================================================================
def mutacion_polinomial(Hijos: np.ndarray,
                        lb: np.ndarray,
                        ub: np.ndarray) -> np.ndarray:
    """
    Mutacion polinomial con control de limites lb/ub.
    Parametro de distribucion: Nm
    """
    Nvar = len(lb)
    for i in range(Np):
        for j in range(Nvar):
            if np.random.rand() <= Pm:
                r     = np.random.rand()
                delta = min(ub[j] - Hijos[i, j],
                            Hijos[i, j] - lb[j]) / (ub[j] - lb[j])

                if r <= 0.5:
                    deltaq = ((2.0 * r + (1.0 - 2.0 * r) *
                               (1.0 - delta) ** (Nm + 1))
                              ** (1.0 / (Nm + 1))) - 1.0
                else:
                    deltaq = 1.0 - ((2.0 * (1.0 - r) +
                                     2.0 * (r - 0.5) *
                                     (1.0 - delta) ** (Nm + 1))
                                    ** (1.0 / (Nm + 1)))

                Hijos[i, j] = np.clip(
                    Hijos[i, j] + deltaq * (ub[j] - lb[j]),
                    lb[j], ub[j])
    return Hijos


# ====================================================================
# PASO 8 — SUSTITUCION EXTINTIVA CON ELITISMO
# ====================================================================
def sustitucion_extintiva(poblacion:        np.ndarray,
                           aptitudes:       np.ndarray,
                           Hijos:           np.ndarray,
                           aptitudes_hijos: np.ndarray):
    """
    Paso 3 - Implicito en cada generacion, pero se realiza al final del ciclo de vida de los hijos.
    Preserva al mejor individuo de la generacion actual (elite)
    reemplazando al peor hijo. Garantiza que la solucion no empeore.
    """
    idx_elite = np.argmax(aptitudes)
    elite     = poblacion[idx_elite].copy()
    ap_elite  = aptitudes[idx_elite]

    idx_peor  = np.argmin(aptitudes_hijos)
    Hijos[idx_peor]           = elite
    aptitudes_hijos[idx_peor] = ap_elite

    return Hijos, aptitudes_hijos


# ====================================================================
# EJECUCION DEL AG
# ====================================================================
def ejecutar_AG(problema) -> tuple:
    """
    Ejecuta el AG para el problema dado.
    Retorna: (mejor_individuo, mejor_fitness, historia)
    """
    lb = problema.lb
    ub = problema.ub

    # == Paso 1 ==
    poblacion = generar_poblacion_inicial(lb, ub)

    # == Paso 2 ==
    aptitudes = evaluar_poblacion(poblacion, problema)

    historia = []
    
    for gen in range(GenMax):

        # == Paso 4 ==
        parents = seleccion_torneo(poblacion, aptitudes)

        # == Paso 5 ==
        Hijos = cruzamiento_sbx(parents, lb, ub)

        # == Paso 6 ==
        Hijos = mutacion_polinomial(Hijos, lb, ub)

        # == Paso 7 ==
        aptitudes_hijos = evaluar_poblacion(Hijos, problema)

        # == Paso 8 == 
        # == Paso 3 == (implicito en paso 8)
        poblacion, aptitudes = sustitucion_extintiva(
            poblacion, aptitudes, Hijos, aptitudes_hijos)

        historia.append({
            "generacion": gen + 1,
            "mejor":      float(np.max(aptitudes)),
            "promedio":   float(np.mean(aptitudes)),
            "peor":       float(np.min(aptitudes)),
        })

    idx = np.argmax(aptitudes)
    return poblacion[idx].copy(), float(aptitudes[idx]), historia


# ====================================================================
# CICLO DE EJECUCIONES PARA UN PROBLEMA
# ====================================================================
def resolver_problema(problema, num_ejecuciones: int,
                       ruta_metricas: str, ruta_resumen: str) -> dict:
    """
    Ejecuta el AG num_ejecuciones veces para el problema dado,
    guarda los CSVs y retorna la mejor solucion encontrada.
    """
    lb = problema.lb
    ub = problema.ub

    resumen_global         = []
    mejor_fitness_global   = -float('inf')
    mejor_individuo_global = None

    for exec_num in range(1, num_ejecuciones + 1):
        ind, fitness, historia = ejecutar_AG(problema)

        # Normalizar pesos respetando lb/ub para garantizar factibilidad
        pesos = np.clip(ind, lb, ub)
        s = pesos.sum()
        if s > 0:
            pesos = pesos / s
        pesos = np.clip(pesos, lb, ub)

        metricas = problema.calcular_metricas(pesos)

        resumen_global.append({
            "ejecucion":     exec_num,
            "mejor_fitness": fitness,
            "pesos":         pesos,
            "retorno":       metricas["retorno"],
            "riesgo":        metricas["riesgo"],
            "P_x":           metricas["P_x"],
            "historia":      historia,
        })

        if fitness > mejor_fitness_global:
            mejor_fitness_global   = fitness
            mejor_individuo_global = pesos.copy()

    guardar_metricas_csv(resumen_global, ruta_metricas)
    guardar_resumen_csv(resumen_global,  ruta_resumen, problema)

    return {
        "mejor_pesos":    mejor_individuo_global,
        "metricas":       problema.calcular_metricas(mejor_individuo_global),
        "resumen_global": resumen_global,
    }


# ====================================================================
# GUARDADO DE CSVs
# ====================================================================
def guardar_metricas_csv(resumen: list, ruta: str) -> None:
    """CSV con evolucion por generacion y ejecucion."""
    with open(ruta, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["ejecucion", "generacion", "mejor", "promedio", "peor"])
        for r in resumen:
            for h in r["historia"]:
                w.writerow([r["ejecucion"], h["generacion"],
                             f"{h['mejor']:.6f}",
                             f"{h['promedio']:.6f}",
                             f"{h['peor']:.6f}"])
    print(f"  CSV metricas  -> {ruta}")


def guardar_resumen_csv(resumen: list, ruta: str, problema) -> None:
    """CSV con pesos por ejecucion + tabla de indicadores."""
    ACCIONES = problema.ACCIONES
    Nvar     = problema.Nvar

    retornos = [r["retorno"] for r in resumen]
    riesgos  = [r["riesgo"]  for r in resumen if r["riesgo"] is not None]

    with open(ruta, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)

        # Pesos por ejecucion
        w.writerow(["Ejecucion", "Fitness_F_P(x)"] +
                   [f"x_{a}" for a in ACCIONES] +
                   ["Retorno (%)", "Riesgo (var)", "Suma_pesos", "P(x)"])
        for r in resumen:
            w.writerow(
                [r["ejecucion"], f"{r['mejor_fitness']:.6f}"] +
                [f"{r['pesos'][j]:.4f}" for j in range(Nvar)] +
                [f"{r['retorno']*100:.2f}",
                 f"{r['riesgo']:.6f}" if r["riesgo"] is not None else "N/A",
                 f"{sum(r['pesos']):.4f}",
                 f"{r['P_x']:.8f}"])

        w.writerow([])

        # Tabla de indicadores 
        nombre = getattr(problema, "NOMBRE", "Problema")
        w.writerow([f"Indicador {nombre}", "Retorno (%)", "Riesgo (var)"])
        w.writerow(["Mejor",
                    f"{max(retornos)*100:.4f}",
                    f"{min(riesgos):.6f}" if riesgos else "N/A"])
        w.writerow(["Media",
                    f"{np.mean(retornos)*100:.4f}",
                    f"{np.mean(riesgos):.6f}" if riesgos else "N/A"])
        w.writerow(["Peor",
                    f"{min(retornos)*100:.4f}",
                    f"{max(riesgos):.6f}" if riesgos else "N/A"])
        w.writerow(["Desv. Estandar",
                    f"{np.std(retornos)*100:.4f}",
                    f"{np.std(riesgos):.6f}" if riesgos else "N/A"])

    print(f"  CSV resumen   -> {ruta}")


# ====================================================================
# IMPRESION DE RESULTADOS
# ====================================================================
def imprimir_resultado(nombre: str, res: dict, problema) -> None:
    pesos    = res["mejor_pesos"]
    metricas = res["metricas"]
    ACCIONES = problema.ACCIONES

    todos_retornos = [r["retorno"] for r in res["resumen_global"]]
    todos_riesgos  = [r["riesgo"]  for r in res["resumen_global"]
                      if r["riesgo"] is not None]

    print(f"\n{'='*65}")
    print(f"  {nombre}")
    print(f"{'='*65}")
    print(f"  Retorno optimo : {metricas['retorno']*100:.4f}%")
    if metricas["riesgo"] is not None:
        print(f"  Riesgo optimo  : {metricas['riesgo']:.6f}  (varianza)")
    print(f"  Penalizacion   : {metricas['P_x']:.8f}  (factible si ~0)")
    print(f"  Suma pesos     : {pesos.sum():.6f}")
    print()
    print("  Cartera optima:")
    for a, v in zip(ACCIONES, pesos):
        bar = "#" * int(v * 100)
        print(f"    {a:5s}: {v*100:6.2f}%  {bar}")
    print()
    print("  +------------------------------------------+")
    print(f"  |  Tabla Indicadores — {nombre[:18]:<18}|")
    print("  +------------------------------------------+")
    print(f"  |  Mejor  retorno : {max(todos_retornos)*100:7.4f} %               |")
    print(f"  |  Media  retorno : {np.mean(todos_retornos)*100:7.4f} %               |")
    print(f"  |  Peor   retorno : {min(todos_retornos)*100:7.4f} %               |")
    print(f"  |  Desv. Estandar : {np.std(todos_retornos)*100:7.4f} %               |")
    if todos_riesgos:
        print(f"  |  Mejor  riesgo  : {min(todos_riesgos):10.6f}              |")
        print(f"  |  Media  riesgo  : {np.mean(todos_riesgos):10.6f}              |")
        print(f"  |  Peor   riesgo  : {max(todos_riesgos):10.6f}              |")
        print(f"  |  Desv.  riesgo  : {np.std(todos_riesgos):10.6f}              |")
    print("  +------------------------------------------+")


# ====================================================================
# RESUMEN COMPARATIVO
# ====================================================================
def imprimir_resumen_comparativo(resultados: list) -> None:
    print(f"\n{'='*65}")
    print("  RESUMEN COMPARATIVO — 3 PROBLEMAS")
    print(f"{'='*65}")
    print(f"  {'Problema':<20} {'Retorno':>10} {'Riesgo':>14} {'Suma xi':>9}")
    print(f"  {'-'*20} {'-'*10} {'-'*14} {'-'*9}")
    for nombre, res, prob in resultados:
        m  = res["metricas"]
        r  = m["retorno"]
        ri = m["riesgo"] if m["riesgo"] is not None else float('nan')
        s  = res["mejor_pesos"].sum()
        print(f"  {nombre:<20} {r*100:>9.2f}%  {ri:>14.6f}  {s:>9.4f}")
    print(f"{'='*65}")
    print()
    print("  Archivos generados:")
    print("    p1_metricas.csv   p1_resumen.csv")
    print("    p2_metricas.csv   p2_resumen.csv")
    print("    p3_metricas.csv   p3_resumen.csv")
    print(f"{'='*65}")


# ====================================================================
# ENTRY POINT
# ====================================================================
if __name__ == "__main__":

    print("=" * 65)
    print("  AG con Penalizacion Exterior para Seleccion de Acciones")
    print("  PRACTICA 2 — SAVVY STOCK SELECTIONS")
    print("=" * 65)
    print(f"  Np={Np} | GenMax={GenMax} | Pc={Pc} | Pm={Pm:.4f}")
    print(f"  Nc={Nc}  | Nm={Nm}  | Ejecuciones={NUM_EJECUCIONES}")
    print("=" * 65)

    resultados = []

    # == PROBLEMA 1 ===================================================
    print(f"\n>>> Resolviendo {problema1.NOMBRE} ({NUM_EJECUCIONES} ejecuciones)...")
    res1 = resolver_problema(
        problema1,
        num_ejecuciones=NUM_EJECUCIONES,
        ruta_metricas="p1_metricas.csv",
        ruta_resumen ="p1_resumen.csv",
    )
    imprimir_resultado(problema1.NOMBRE, res1, problema1)
    resultados.append(("P1 (max ret)", res1, problema1))

    # == PROBLEMA 2 ===================================================
    print(f"\n>>> Resolviendo {problema2.NOMBRE} ({NUM_EJECUCIONES} ejecuciones)...")
    res2 = resolver_problema(
        problema2,
        num_ejecuciones=NUM_EJECUCIONES,
        ruta_metricas="p2_metricas.csv",
        ruta_resumen ="p2_resumen.csv",
    )
    imprimir_resultado(problema2.NOMBRE, res2, problema2)
    resultados.append(("P2 (min rie)", res2, problema2))

    # == PROBLEMA 3 ===================================================
    print(f"\n>>> Resolviendo {problema3.NOMBRE} ({NUM_EJECUCIONES} ejecuciones)...")
    res3 = resolver_problema(
        problema3,
        num_ejecuciones=NUM_EJECUCIONES,
        ruta_metricas="p3_metricas.csv",
        ruta_resumen ="p3_resumen.csv",
    )
    imprimir_resultado(problema3.NOMBRE, res3, problema3)
    resultados.append(("P3 (max ret, rie<=20%)", res3, problema3))

    # == RESUMEN FINAL =================================================
    imprimir_resumen_comparativo(resultados)
