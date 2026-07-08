# ============================================================
# 15_ver_prediccion_ridge_cuartos.py
# Versión limpia en español.
#
# Objetivo:
# - Mostrar la lectura específica del modelo Ridge para cuartos.
# - Evitar salidas como "gana por Ridge calibrado".
# - Usar etiquetas claras en español.
# - Ajustar las lambdas a una escala realista de fútbol.
#
# Este script NO entrena.
# Ejecutar después de reentrenar o después de generar fixtures de cuartos:
#
# python 15_ver_prediccion_ridge_cuartos.py
# ============================================================

from pathlib import Path
import math
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUT_DIR = BASE_DIR / "outputs"

FIXTURES = DATA_DIR / "fixtures_cuartos_modelo.csv"

MODELO_LOCAL = OUT_DIR / "modelo_home_ridge.pkl"
MODELO_VISITANTE = OUT_DIR / "modelo_away_ridge.pkl"
COLUMNAS_MODELO = OUT_DIR / "columnas_modelo.pkl"

SALIDA = OUT_DIR / "prediccion_ridge_cuartos.csv"
PANEL = OUT_DIR / "panel_ridge_cuartos.png"

MAX_GOLES = 6

# Escala realista para fase eliminatoria.
# No es una certeza, es una calibración práctica para evitar lambdas excesivas.
MEDIA_GOLES_PARTIDO = 2.15
LAMBDA_MIN = 0.35
LAMBDA_MAX = 2.35


def sigmoid(x):
    x = max(min(float(x), 8), -8)
    return 1 / (1 + math.exp(-x))


def poisson_pmf(k, lam):
    if lam <= 0:
        return 0
    return math.exp(-lam) * (lam ** k) / math.factorial(k)


def leer_csv(ruta):
    if not ruta.exists():
        raise FileNotFoundError(f"No existe: {ruta}")
    df = pd.read_csv(ruta)
    df.columns = df.columns.str.strip()
    return df


def obtener_columnas_predictoras(df):
    if COLUMNAS_MODELO.exists():
        columnas = joblib.load(COLUMNAS_MODELO)
        columnas = [c for c in columnas if c in df.columns]
        if len(columnas) > 0:
            return columnas

    candidatas = [
        "home_gf12",
        "home_ga12",
        "home_pts12",
        "home_prev_matches",
        "away_gf12",
        "away_ga12",
        "away_pts12",
        "away_prev_matches",
        "diff_fifa",
        "diff_elo",
        "h2h",
        "neutral",
        "home_advantage",
        "home_fifa_rank",
        "away_fifa_rank",
        "home_elo",
        "away_elo",
        "home_form",
        "away_form",
        "home_attack_strength",
        "away_attack_strength",
        "home_defense_strength",
        "away_defense_strength"
    ]

    columnas = [c for c in candidatas if c in df.columns]

    if len(columnas) == 0:
        excluir = {
            "date", "home_team", "away_team", "home_score", "away_score",
            "tournament", "city", "country", "stage", "round", "match_id"
        }

        for col in df.columns:
            if col in excluir:
                continue

            serie = pd.to_numeric(df[col], errors="coerce")
            if serie.notna().sum() > 0:
                columnas.append(col)

    if len(columnas) == 0:
        raise ValueError("No se encontraron columnas predictoras.")

    return columnas


def preparar_matriz(df, columnas):
    X = df[columnas].copy()

    for col in columnas:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True))
    X = X.fillna(0)

    return X


def numero(fila, col, default=0):
    valor = pd.to_numeric(fila.get(col, default), errors="coerce")
    if pd.isna(valor):
        return default
    return float(valor)


def calcular_ventaja_contextual(fila):
    """
    Valor positivo favorece al equipo local.
    Valor negativo favorece al equipo visitante.
    """

    diff_elo = numero(fila, "diff_elo", 0)
    diff_fifa = numero(fila, "diff_fifa", 0)
    h2h = numero(fila, "h2h", 0)

    home_gf12 = numero(fila, "home_gf12", 0)
    away_gf12 = numero(fila, "away_gf12", 0)

    home_ga12 = numero(fila, "home_ga12", 0)
    away_ga12 = numero(fila, "away_ga12", 0)

    home_pts12 = numero(fila, "home_pts12", 0)
    away_pts12 = numero(fila, "away_pts12", 0)

    ventaja = 0.0

    # Elo: mayor Elo favorece.
    ventaja += diff_elo / 450.0

    # FIFA: menor ranking es mejor.
    # Si diff_fifa = ranking_local - ranking_visitante,
    # un valor negativo favorece al local.
    ventaja += -diff_fifa / 45.0

    # Historial directo.
    ventaja += h2h * 0.08

    # Forma reciente.
    ventaja += (home_gf12 - away_gf12) / 18.0
    ventaja += (away_ga12 - home_ga12) / 18.0
    ventaja += (home_pts12 - away_pts12) / 30.0

    return max(min(ventaja, 2.5), -2.5)


def ajustar_lambdas(pred_local_cruda, pred_visitante_cruda, fila):
    """
    Ridge puede producir valores muy altos.
    Este ajuste mantiene la relación entre ambos equipos, pero lleva
    el total esperado de goles a una escala más realista de fútbol.
    """

    pred_local_cruda = max(float(pred_local_cruda), 0.01)
    pred_visitante_cruda = max(float(pred_visitante_cruda), 0.01)

    total_crudo = pred_local_cruda + pred_visitante_cruda

    if total_crudo > 0:
        lambda_modelo_local = MEDIA_GOLES_PARTIDO * (pred_local_cruda / total_crudo)
        lambda_modelo_visitante = MEDIA_GOLES_PARTIDO * (pred_visitante_cruda / total_crudo)
    else:
        lambda_modelo_local = MEDIA_GOLES_PARTIDO / 2
        lambda_modelo_visitante = MEDIA_GOLES_PARTIDO / 2

    ventaja = calcular_ventaja_contextual(fila)

    proporcion_local = sigmoid(ventaja)

    lambda_contexto_local = MEDIA_GOLES_PARTIDO * proporcion_local
    lambda_contexto_visitante = MEDIA_GOLES_PARTIDO * (1 - proporcion_local)

    # Mezcla del modelo con contexto.
    # 60% salida del modelo Ridge y 40% contexto FIFA/Elo/forma.
    lambda_local = 0.60 * lambda_modelo_local + 0.40 * lambda_contexto_local
    lambda_visitante = 0.60 * lambda_modelo_visitante + 0.40 * lambda_contexto_visitante

    lambda_local = float(np.clip(lambda_local, LAMBDA_MIN, LAMBDA_MAX))
    lambda_visitante = float(np.clip(lambda_visitante, LAMBDA_MIN, LAMBDA_MAX))

    return lambda_local, lambda_visitante, ventaja


def calcular_probabilidades(lambda_local, lambda_visitante):
    matriz = []

    for goles_local in range(MAX_GOLES + 1):
        for goles_visitante in range(MAX_GOLES + 1):
            p = poisson_pmf(goles_local, lambda_local) * poisson_pmf(goles_visitante, lambda_visitante)
            matriz.append((goles_local, goles_visitante, p))

    total = sum(p for _, _, p in matriz)

    if total > 0:
        matriz = [(g_l, g_v, p / total) for g_l, g_v, p in matriz]

    prob_local = sum(p for g_l, g_v, p in matriz if g_l > g_v)
    prob_empate = sum(p for g_l, g_v, p in matriz if g_l == g_v)
    prob_visitante = sum(p for g_l, g_v, p in matriz if g_l < g_v)

    marcadores = sorted(matriz, key=lambda x: x[2], reverse=True)

    return prob_local, prob_empate, prob_visitante, marcadores


def decidir_clasificado(fila, goles_local, goles_visitante, prob_local, prob_visitante, ventaja):
    local = fila["home_team"]
    visitante = fila["away_team"]

    if goles_local > goles_visitante:
        return local, "Victoria en 90 minutos", "Victoria local"

    if goles_visitante > goles_local:
        return visitante, "Victoria en 90 minutos", "Victoria visitante"

    # Si el marcador más probable es empate, se define el clasificado por probabilidad/contexto.
    if prob_local > prob_visitante:
        return local, "Empate en 90 minutos; avanza por mayor probabilidad estimada", "Empate"

    if prob_visitante > prob_local:
        return visitante, "Empate en 90 minutos; avanza por mayor probabilidad estimada", "Empate"

    if ventaja > 0:
        return local, "Empate en 90 minutos; avanza por ventaja de contexto", "Empate"

    if ventaja < 0:
        return visitante, "Empate en 90 minutos; avanza por ventaja de contexto", "Empate"

    return local, "Empate en 90 minutos; avanza por criterio auxiliar", "Empate"


def crear_panel(df):
    tabla = df[[
        "match_id",
        "partido",
        "lambda_local_cruda",
        "lambda_visitante_cruda",
        "lambda_local_ajustada",
        "lambda_visitante_ajustada",
        "marcador_90_min",
        "resultado_90_min",
        "prob_local_pct",
        "prob_empate_pct",
        "prob_visitante_pct",
        "clasificado_estimado",
        "criterio"
    ]].copy()

    for col in tabla.columns:
        if col not in [
            "partido",
            "marcador_90_min",
            "resultado_90_min",
            "clasificado_estimado",
            "criterio"
        ]:
            tabla[col] = pd.to_numeric(tabla[col], errors="coerce").round(3)

    fig, ax = plt.subplots(figsize=(24, 6))
    ax.axis("off")

    fig.suptitle(
        "Lectura del modelo Ridge - Cuartos de final",
        fontsize=16,
        fontweight="bold"
    )

    t = ax.table(
        cellText=tabla.values,
        colLabels=tabla.columns,
        cellLoc="center",
        loc="center"
    )

    t.auto_set_font_size(False)
    t.set_fontsize(7.5)
    t.scale(1, 1.6)

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.savefig(PANEL, dpi=300, bbox_inches="tight")
    plt.close()


def main():
    if not MODELO_LOCAL.exists():
        raise FileNotFoundError(f"No existe {MODELO_LOCAL}. Reentrena o copia los modelos Ridge.")

    if not MODELO_VISITANTE.exists():
        raise FileNotFoundError(f"No existe {MODELO_VISITANTE}. Reentrena o copia los modelos Ridge.")

    fixtures = leer_csv(FIXTURES)

    modelo_local = joblib.load(MODELO_LOCAL)
    modelo_visitante = joblib.load(MODELO_VISITANTE)

    columnas = obtener_columnas_predictoras(fixtures)
    X = preparar_matriz(fixtures, columnas)

    pred_local_cruda = modelo_local.predict(X)
    pred_visitante_cruda = modelo_visitante.predict(X)

    filas = []

    for i, fila in fixtures.iterrows():
        l_cruda = float(pred_local_cruda[i])
        v_cruda = float(pred_visitante_cruda[i])

        l_ajustada, v_ajustada, ventaja = ajustar_lambdas(
            l_cruda,
            v_cruda,
            fila
        )

        prob_local, prob_empate, prob_visitante, marcadores = calcular_probabilidades(
            l_ajustada,
            v_ajustada
        )

        goles_local, goles_visitante, prob_marcador = marcadores[0]

        clasificado, criterio, resultado_90 = decidir_clasificado(
            fila,
            goles_local,
            goles_visitante,
            prob_local,
            prob_visitante,
            ventaja
        )

        filas.append({
            "match_id": fila.get("match_id", i + 1),
            "equipo_local": fila["home_team"],
            "equipo_visitante": fila["away_team"],
            "partido": f"{fila['home_team']} vs {fila['away_team']}",
            "lambda_local_cruda": round(l_cruda, 4),
            "lambda_visitante_cruda": round(v_cruda, 4),
            "lambda_local_ajustada": round(l_ajustada, 4),
            "lambda_visitante_ajustada": round(v_ajustada, 4),
            "ventaja_contexto_local": round(ventaja, 4),
            "marcador_90_min": f"{goles_local}-{goles_visitante}",
            "prob_marcador_pct": round(prob_marcador * 100, 2),
            "resultado_90_min": resultado_90,
            "prob_local_pct": round(prob_local * 100, 2),
            "prob_empate_pct": round(prob_empate * 100, 2),
            "prob_visitante_pct": round(prob_visitante * 100, 2),
            "clasificado_estimado": clasificado,
            "criterio": criterio
        })

    salida = pd.DataFrame(filas)
    salida.to_csv(SALIDA, index=False, encoding="utf-8")

    crear_panel(salida)

    print()
    print("========================================")
    print("LECTURA RIDGE - CUARTOS DE FINAL")
    print("========================================")
    print(salida.to_string(index=False))

    print()
    print("Archivos generados:")
    print(SALIDA)
    print(PANEL)
    print("========================================")


if __name__ == "__main__":
    main()
