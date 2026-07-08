# ============================================================
# 16_aplicar_ponderacion_balanceada.py
#
# Aplica una ponderacion balanceada a la prediccion de cuartos.
#
# Ponderacion:
# - 40% prediccion del modelo entrenado
# - 25% ranking Elo
# - 15% ranking FIFA
# - 15% forma reciente
# -  5% contexto / historial / Fair Play
#
# Uso recomendado:
# python 13_predecir_semifinales.py
# python 16_aplicar_ponderacion_balanceada.py
#
# Salidas:
# outputs/prediccion_cuartos_a_semifinales_balanceada.csv
# outputs/clasificados_semifinales_balanceado.csv
# outputs/panel_ponderacion_balanceada.png
#
# Tambien actualiza:
# outputs/prediccion_cuartos_a_semifinales.csv
# outputs/clasificados_semifinales.csv
#
# Antes de sobrescribir, crea backup automatico.
# ============================================================

from pathlib import Path
import math
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURACION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUT_DIR = BASE_DIR / "outputs"

ARCHIVO_PREDICCION = OUT_DIR / "prediccion_cuartos_a_semifinales.csv"
ARCHIVO_FIXTURES = DATA_DIR / "fixtures_cuartos_modelo.csv"
ARCHIVO_CLASIFICADOS = OUT_DIR / "clasificados_semifinales.csv"

SALIDA_BALANCEADA = OUT_DIR / "prediccion_cuartos_a_semifinales_balanceada.csv"
SALIDA_CLASIFICADOS = OUT_DIR / "clasificados_semifinales_balanceado.csv"
PANEL = OUT_DIR / "panel_ponderacion_balanceada.png"

SOBRESCRIBIR_SALIDAS_PRINCIPALES = True

PESO_MODELO = 0.40
PESO_ELO = 0.25
PESO_FIFA = 0.15
PESO_FORMA = 0.15
PESO_CONTEXTO = 0.05

PESOS = {
    "modelo": PESO_MODELO,
    "elo": PESO_ELO,
    "fifa": PESO_FIFA,
    "forma": PESO_FORMA,
    "contexto": PESO_CONTEXTO
}


# ============================================================
# FUNCIONES AUXILIARES
# ============================================================

def sigmoid(x):
    x = max(min(float(x), 8), -8)
    return 1 / (1 + math.exp(-x))


def leer_csv(ruta):
    if not ruta.exists():
        raise FileNotFoundError(f"No existe el archivo: {ruta}")
    df = pd.read_csv(ruta)
    df.columns = df.columns.str.strip()
    return df


def backup(ruta):
    if ruta.exists():
        copia = ruta.with_suffix(ruta.suffix + ".backup")
        shutil.copy2(ruta, copia)
        print(f"Backup creado: {copia}")


def valor(fila, posibles_columnas, default=0.0):
    for col in posibles_columnas:
        if col in fila.index:
            v = pd.to_numeric(fila[col], errors="coerce")
            if pd.notna(v):
                return float(v)
    return float(default)


def texto(fila, posibles_columnas, default=""):
    for col in posibles_columnas:
        if col in fila.index and pd.notna(fila[col]):
            return str(fila[col])
    return default


def ajustar_probabilidad(p):
    p = pd.to_numeric(p, errors="coerce")

    if pd.isna(p):
        return np.nan

    p = float(p)

    # Si viene en porcentaje, lo convertimos a proporcion.
    if p > 1:
        p = p / 100

    return float(np.clip(p, 0, 1))


def obtener_prob_modelo(fila):
    """
    Convierte la salida del modelo en probabilidad de avance del local.
    Si el modelo tiene probabilidad local/visitante, se usa.
    Si no, se usa 0.50.
    """

    prob_local_cols = [
        "prob_home_percent",
        "prob_home_pct",
        "prob_local_pct",
        "prob_local_percent",
        "prob_adv_home_percent",
        "prob_victoria_local_pct",
        "prob_victoria_local",
        "prob_home",
        "prob_local"
    ]

    prob_visitante_cols = [
        "prob_away_percent",
        "prob_away_pct",
        "prob_visitante_pct",
        "prob_visitante_percent",
        "prob_adv_away_percent",
        "prob_victoria_visitante_pct",
        "prob_victoria_visitante",
        "prob_away",
        "prob_visitante"
    ]

    p_local = np.nan
    p_visitante = np.nan

    for col in prob_local_cols:
        if col in fila.index:
            p_local = ajustar_probabilidad(fila[col])
            break

    for col in prob_visitante_cols:
        if col in fila.index:
            p_visitante = ajustar_probabilidad(fila[col])
            break

    if pd.notna(p_local) and pd.notna(p_visitante):
        total = p_local + p_visitante
        if total > 0:
            return float(p_local / total)

    if pd.notna(p_local):
        return float(p_local)

    return 0.50


def calcular_score_elo(fila):
    """
    Retorna probabilidad contextual local segun Elo.
    Se asume:
    diff_elo = Elo_local - Elo_visitante.
    """

    diff_elo = valor(
        fila,
        [
            "diff_elo",
            "elo_diff",
            "diferencia_elo"
        ],
        default=np.nan
    )

    if pd.isna(diff_elo):
        elo_local = valor(fila, ["home_elo", "elo_home", "elo_local"], default=np.nan)
        elo_visitante = valor(fila, ["away_elo", "elo_away", "elo_visitante"], default=np.nan)

        if pd.notna(elo_local) and pd.notna(elo_visitante):
            diff_elo = elo_local - elo_visitante
        else:
            return 0.50

    # 400 puntos Elo aproxima una diferencia fuerte.
    return sigmoid(diff_elo / 400)


def calcular_score_fifa(fila):
    """
    Retorna probabilidad contextual local segun ranking FIFA.
    En ranking FIFA, menor posicion es mejor.
    Se asume:
    diff_fifa = ranking_local - ranking_visitante.
    """

    diff_fifa = valor(
        fila,
        [
            "diff_fifa",
            "fifa_diff",
            "diferencia_fifa"
        ],
        default=np.nan
    )

    if pd.isna(diff_fifa):
        fifa_local = valor(
            fila,
            [
                "home_fifa_rank",
                "fifa_home",
                "ranking_fifa_local",
                "fifa_local"
            ],
            default=np.nan
        )

        fifa_visitante = valor(
            fila,
            [
                "away_fifa_rank",
                "fifa_away",
                "ranking_fifa_visitante",
                "fifa_visitante"
            ],
            default=np.nan
        )

        if pd.notna(fifa_local) and pd.notna(fifa_visitante):
            diff_fifa = fifa_local - fifa_visitante
        else:
            return 0.50

    # Si diff_fifa es negativo, el local tiene mejor ranking.
    return sigmoid((-diff_fifa) / 45)


def calcular_score_forma(fila):
    """
    Retorna probabilidad contextual local segun forma reciente.
    Combina:
    - puntos recientes
    - goles a favor recientes
    - goles en contra recientes
    """

    home_pts12 = valor(fila, ["home_pts12", "pts_home_12", "puntos_local_12"], default=0)
    away_pts12 = valor(fila, ["away_pts12", "pts_away_12", "puntos_visitante_12"], default=0)

    home_gf12 = valor(fila, ["home_gf12", "gf_home_12", "goles_favor_local_12"], default=0)
    away_gf12 = valor(fila, ["away_gf12", "gf_away_12", "goles_favor_visitante_12"], default=0)

    home_ga12 = valor(fila, ["home_ga12", "ga_home_12", "goles_contra_local_12"], default=0)
    away_ga12 = valor(fila, ["away_ga12", "ga_away_12", "goles_contra_visitante_12"], default=0)

    # Ventaja positiva favorece al local.
    ventaja_forma = 0.0
    ventaja_forma += (home_pts12 - away_pts12) / 30
    ventaja_forma += (home_gf12 - away_gf12) / 18
    ventaja_forma += (away_ga12 - home_ga12) / 18

    return sigmoid(ventaja_forma)


def calcular_score_contexto(fila):
    """
    Contexto auxiliar:
    - historial directo si existe
    - Fair Play / tarjetas si existen
    """

    h2h = valor(fila, ["h2h", "historial_directo", "head_to_head"], default=0)

    # En tarjetas/Fair Play, menor valor suele ser mejor.
    tarjetas_local = valor(
        fila,
        [
            "home_cards",
            "cards_home",
            "tarjetas_local",
            "home_tarjetas",
            "home_yellow_cards",
            "yellow_home",
            "fair_play_home",
            "home_fair_play"
        ],
        default=np.nan
    )

    tarjetas_visitante = valor(
        fila,
        [
            "away_cards",
            "cards_away",
            "tarjetas_visitante",
            "away_tarjetas",
            "away_yellow_cards",
            "yellow_away",
            "fair_play_away",
            "away_fair_play"
        ],
        default=np.nan
    )

    ventaja_contexto = 0.0
    ventaja_contexto += h2h * 0.08

    if pd.notna(tarjetas_local) and pd.notna(tarjetas_visitante):
        ventaja_contexto += (tarjetas_visitante - tarjetas_local) / 10

    return sigmoid(ventaja_contexto)


def obtener_partido(fila):
    local = texto(
        fila,
        [
            "home_team",
            "home_team_pred",
            "home_team_ctx",
            "equipo_local",
            "local"
        ],
        default="Local"
    )

    visitante = texto(
        fila,
        [
            "away_team",
            "away_team_pred",
            "away_team_ctx",
            "equipo_visitante",
            "visitante"
        ],
        default="Visitante"
    )

    return local, visitante


def decidir_clasificado(p_local_balanceada, local, visitante):
    if p_local_balanceada >= 0.50:
        return local
    return visitante


def interpretar(p_local_balanceada):
    ventaja = abs(p_local_balanceada - 0.50)

    if ventaja < 0.035:
        return "Partido muy parejo"
    if ventaja < 0.075:
        return "Ligera ventaja"
    if ventaja < 0.140:
        return "Ventaja moderada"
    return "Favorito claro"


def unir_prediccion_contexto(pred, fixtures):
    if "match_id" in pred.columns and "match_id" in fixtures.columns:
        return pred.merge(
            fixtures,
            on="match_id",
            how="left",
            suffixes=("", "_ctx")
        )

    # Plan B: unir por equipos si existen.
    if (
        "home_team" in pred.columns and "away_team" in pred.columns and
        "home_team" in fixtures.columns and "away_team" in fixtures.columns
    ):
        return pred.merge(
            fixtures,
            on=["home_team", "away_team"],
            how="left",
            suffixes=("", "_ctx")
        )

    # Si no se puede unir, se trabaja con la prediccion sola.
    return pred.copy()


def crear_panel(df):
    columnas = [
        "match_id",
        "partido",
        "p_modelo_local_pct",
        "p_elo_local_pct",
        "p_fifa_local_pct",
        "p_forma_local_pct",
        "p_contexto_local_pct",
        "p_local_balanceada_pct",
        "p_visitante_balanceada_pct",
        "clasificado_balanceado",
        "lectura"
    ]

    columnas = [c for c in columnas if c in df.columns]
    tabla = df[columnas].copy()

    for col in tabla.columns:
        if col not in ["partido", "clasificado_balanceado", "lectura"]:
            tabla[col] = pd.to_numeric(tabla[col], errors="coerce").round(2)

    fig, ax = plt.subplots(figsize=(22, 6))
    ax.axis("off")

    fig.suptitle(
        "Ponderacion balanceada para cuartos de final",
        fontsize=16,
        fontweight="bold"
    )

    subtitulo = (
        "40% modelo entrenado | 25% Elo | 15% FIFA | "
        "15% forma reciente | 5% contexto/Fair Play"
    )

    ax.set_title(subtitulo, fontsize=10, pad=12)

    t = ax.table(
        cellText=tabla.values,
        colLabels=tabla.columns,
        cellLoc="center",
        loc="center"
    )

    t.auto_set_font_size(False)
    t.set_fontsize(8)
    t.scale(1, 1.6)

    plt.tight_layout(rect=[0, 0, 1, 0.90])
    plt.savefig(PANEL, dpi=300, bbox_inches="tight")
    plt.close()


# ============================================================
# PROCESO PRINCIPAL
# ============================================================

def main():
    OUT_DIR.mkdir(exist_ok=True)

    pred = leer_csv(ARCHIVO_PREDICCION)
    fixtures = leer_csv(ARCHIVO_FIXTURES)

    df = unir_prediccion_contexto(pred, fixtures)

    filas = []

    for _, fila in df.iterrows():
        local, visitante = obtener_partido(fila)
        partido = f"{local} vs {visitante}"

        p_modelo = obtener_prob_modelo(fila)
        p_elo = calcular_score_elo(fila)
        p_fifa = calcular_score_fifa(fila)
        p_forma = calcular_score_forma(fila)
        p_contexto = calcular_score_contexto(fila)

        p_local_balanceada = (
            PESO_MODELO * p_modelo +
            PESO_ELO * p_elo +
            PESO_FIFA * p_fifa +
            PESO_FORMA * p_forma +
            PESO_CONTEXTO * p_contexto
        )

        p_local_balanceada = float(np.clip(p_local_balanceada, 0.01, 0.99))
        p_visitante_balanceada = 1 - p_local_balanceada

        clasificado = decidir_clasificado(
            p_local_balanceada,
            local,
            visitante
        )

        fila_nueva = fila.to_dict()

        fila_nueva["partido"] = partido
        fila_nueva["p_modelo_local_pct"] = round(p_modelo * 100, 2)
        fila_nueva["p_elo_local_pct"] = round(p_elo * 100, 2)
        fila_nueva["p_fifa_local_pct"] = round(p_fifa * 100, 2)
        fila_nueva["p_forma_local_pct"] = round(p_forma * 100, 2)
        fila_nueva["p_contexto_local_pct"] = round(p_contexto * 100, 2)

        fila_nueva["peso_modelo"] = PESO_MODELO
        fila_nueva["peso_elo"] = PESO_ELO
        fila_nueva["peso_fifa"] = PESO_FIFA
        fila_nueva["peso_forma"] = PESO_FORMA
        fila_nueva["peso_contexto"] = PESO_CONTEXTO

        fila_nueva["p_local_balanceada_pct"] = round(p_local_balanceada * 100, 2)
        fila_nueva["p_visitante_balanceada_pct"] = round(p_visitante_balanceada * 100, 2)
        fila_nueva["clasificado_balanceado"] = clasificado
        fila_nueva["lectura"] = interpretar(p_local_balanceada)

        filas.append(fila_nueva)

    salida = pd.DataFrame(filas)

    # Guardar salida balanceada completa.
    salida.to_csv(SALIDA_BALANCEADA, index=False, encoding="utf-8")

    # Generar clasificados balanceados.
    clasificados = pd.DataFrame({
        "fase": ["Semifinal"] * len(salida),
        "match_id_origen": salida.get("match_id", pd.Series(range(1, len(salida) + 1))),
        "partido_origen": salida["partido"],
        "clasificado": salida["clasificado_balanceado"],
        "probabilidad_balanceada_pct": salida.apply(
            lambda r: r["p_local_balanceada_pct"]
            if r["clasificado_balanceado"] == obtener_partido(r)[0]
            else r["p_visitante_balanceada_pct"],
            axis=1
        ),
        "lectura": salida["lectura"]
    })

    clasificados.to_csv(SALIDA_CLASIFICADOS, index=False, encoding="utf-8")

    crear_panel(salida)

    if SOBRESCRIBIR_SALIDAS_PRINCIPALES:
        backup(ARCHIVO_PREDICCION)
        backup(ARCHIVO_CLASIFICADOS)

        salida.to_csv(ARCHIVO_PREDICCION, index=False, encoding="utf-8")
        clasificados.to_csv(ARCHIVO_CLASIFICADOS, index=False, encoding="utf-8")

        print()
        print("Salidas principales actualizadas:")
        print(ARCHIVO_PREDICCION)
        print(ARCHIVO_CLASIFICADOS)

    print()
    print("========================================")
    print("PONDERACION BALANCEADA APLICADA")
    print("========================================")
    print("Pesos usados:")
    for k, v in PESOS.items():
        print(f"- {k}: {v * 100:.0f}%")

    print()
    print(salida[[
        "partido",
        "p_modelo_local_pct",
        "p_elo_local_pct",
        "p_fifa_local_pct",
        "p_forma_local_pct",
        "p_contexto_local_pct",
        "p_local_balanceada_pct",
        "p_visitante_balanceada_pct",
        "clasificado_balanceado",
        "lectura"
    ]].to_string(index=False))

    print()
    print("Archivos generados:")
    print(SALIDA_BALANCEADA)
    print(SALIDA_CLASIFICADOS)
    print(PANEL)
    print("========================================")


if __name__ == "__main__":
    main()
