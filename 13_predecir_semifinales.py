# ============================================================
# 13_predecir_semifinales.py
# Cuartos -> Semifinales
# ============================================================

import math
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import poisson


# ============================================================
# CONFIGURACION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUT_DIR = BASE_DIR / "outputs"

ARCHIVO_FIXTURES_CUARTOS_MODELO = DATA_DIR / "fixtures_cuartos_modelo.csv"
ARCHIVO_FAIR_PLAY = DATA_DIR / "fair_play.csv"

ARCHIVO_MODELO_HOME = OUT_DIR / "modelo_home.pkl"
ARCHIVO_MODELO_AWAY = OUT_DIR / "modelo_away.pkl"

ARCHIVO_RESUMEN = OUT_DIR / "prediccion_cuartos_a_semifinales.csv"
ARCHIVO_TOP10 = OUT_DIR / "top10_marcadores_cuartos.csv"
ARCHIVO_TARJETAS = OUT_DIR / "prediccion_tarjetas_cuartos.csv"
ARCHIVO_SEMIFINALES = OUT_DIR / "clasificados_semifinales.csv"
IMAGEN_PANEL = OUT_DIR / "panel_cuartos_marcadores_tarjetas.png"

MAX_GOLES = 5

# Dixon-Coles. Valor negativo suave para ajustar marcadores bajos.
RHO_DIXON_COLES = -0.08

# Rango razonable de goles esperados por equipo en futbol.
LAMBDA_MIN = 0.35
LAMBDA_MAX = 2.35

# Media conservadora para fase eliminatoria.
MEDIA_GOLES_EQUIPO = 1.18

# En eliminatorias se reduce un poco la expectativa ofensiva.
FACTOR_ELIMINATORIA = 0.92

VARIABLES = [
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
]


# ============================================================
# UTILIDADES
# ============================================================

def limitar(valor, minimo, maximo):
    return min(max(valor, minimo), maximo)


def safe_float(valor, defecto=0.0):
    try:
        valor = float(valor)
        if math.isnan(valor) or math.isinf(valor):
            return defecto
        return valor
    except Exception:
        return defecto


def tasa_por_partido(total, partidos, defecto=MEDIA_GOLES_EQUIPO):
    total = safe_float(total, 0.0)
    partidos = safe_float(partidos, 0.0)

    if partidos <= 0:
        return defecto

    return total / partidos


# ============================================================
# CARGA DE DATOS
# ============================================================

def validar_archivos():
    if not ARCHIVO_FIXTURES_CUARTOS_MODELO.exists():
        raise FileNotFoundError(
            "No existe data/fixtures_cuartos_modelo.csv. "
            "Ejecuta primero: python 09_actualizar_fixtures_modelo_eliminatorias.py"
        )

    if not ARCHIVO_MODELO_HOME.exists():
        raise FileNotFoundError("No existe outputs/modelo_home.pkl. Ejecuta primero el script 02.")

    if not ARCHIVO_MODELO_AWAY.exists():
        raise FileNotFoundError("No existe outputs/modelo_away.pkl. Ejecuta primero el script 02.")

    OUT_DIR.mkdir(exist_ok=True)


def cargar_datos():
    validar_archivos()

    fixtures = pd.read_csv(ARCHIVO_FIXTURES_CUARTOS_MODELO)
    fixtures.columns = fixtures.columns.str.strip()

    for col in VARIABLES:
        if col not in fixtures.columns:
            raise ValueError("Falta columna en fixtures_cuartos_modelo.csv: " + col)

    fixtures["date"] = pd.to_datetime(
        fixtures["date"],
        errors="coerce"
    ).dt.strftime("%Y-%m-%d")

    modelo_home = joblib.load(ARCHIVO_MODELO_HOME)
    modelo_away = joblib.load(ARCHIVO_MODELO_AWAY)

    fair_play = cargar_fair_play()

    return fixtures, modelo_home, modelo_away, fair_play


def cargar_fair_play():
    if not ARCHIVO_FAIR_PLAY.exists():
        print("ADVERTENCIA: No existe data/fair_play.csv. Se usarán valores neutros.")
        return pd.DataFrame(columns=[
            "team",
            "yellow_cards",
            "indirect_red_cards",
            "direct_red_cards",
            "team_conduct_score",
            "matches_count"
        ])

    fair = pd.read_csv(ARCHIVO_FAIR_PLAY)
    fair.columns = fair.columns.str.strip()

    if "team" not in fair.columns:
        raise ValueError("fair_play.csv debe tener columna team")

    columnas = [
        "yellow_cards",
        "indirect_red_cards",
        "direct_red_cards",
        "team_conduct_score",
        "matches_count"
    ]

    for col in columnas:
        if col not in fair.columns:
            fair[col] = 0

    for col in columnas:
        fair[col] = pd.to_numeric(
            fair[col],
            errors="coerce"
        ).fillna(0)

    fair["team"] = fair["team"].astype(str).str.strip()

    return fair


# ============================================================
# LAMBDAS REALISTAS
# ============================================================

def lambda_por_forma_y_rival(fila, lado):
    """
    Calcula una lambda base desde datos futbolísticos:
    ataque propio + defensa rival + puntos recientes + diferencia Elo/FIFA.

    Esta parte evita depender ciegamente de la regresión cuando el modelo
    devuelve valores extremos.
    """

    if lado == "home":
        gf = tasa_por_partido(
            fila.get("home_gf12"),
            fila.get("home_prev_matches"),
            MEDIA_GOLES_EQUIPO
        )
        ga_rival = tasa_por_partido(
            fila.get("away_ga12"),
            fila.get("away_prev_matches"),
            MEDIA_GOLES_EQUIPO
        )
        pts = tasa_por_partido(
            fila.get("home_pts12"),
            fila.get("home_prev_matches"),
            1.4
        )
        diff_elo = safe_float(fila.get("diff_elo"), 0.0)
        diff_fifa = safe_float(fila.get("diff_fifa"), 0.0)
        h2h = safe_float(fila.get("h2h"), 0.0)
    else:
        gf = tasa_por_partido(
            fila.get("away_gf12"),
            fila.get("away_prev_matches"),
            MEDIA_GOLES_EQUIPO
        )
        ga_rival = tasa_por_partido(
            fila.get("home_ga12"),
            fila.get("home_prev_matches"),
            MEDIA_GOLES_EQUIPO
        )
        pts = tasa_por_partido(
            fila.get("away_pts12"),
            fila.get("away_prev_matches"),
            1.4
        )
        diff_elo = -safe_float(fila.get("diff_elo"), 0.0)
        diff_fifa = -safe_float(fila.get("diff_fifa"), 0.0)
        h2h = -safe_float(fila.get("h2h"), 0.0)

    gf = limitar(gf, 0.25, 3.20)
    ga_rival = limitar(ga_rival, 0.25, 3.20)
    pts = limitar(pts, 0.0, 3.0)

    # Base: ataque propio pesa más que defensa rival.
    base = 0.52 * gf + 0.36 * ga_rival + 0.12 * MEDIA_GOLES_EQUIPO

    # Forma por puntos: equipo con buena forma sube poco, no exagerado.
    factor_forma = 1.0 + limitar((pts - 1.4) * 0.10, -0.12, 0.12)

    # Elo: ajuste suave. No dejar que domine.
    factor_elo = 1.0 + limitar(diff_elo / 900.0, -0.18, 0.18)

    # FIFA puede estar como puntos o ranking; por eso ajuste pequeño.
    factor_fifa = 1.0 + limitar(diff_fifa / 2500.0, -0.08, 0.08)

    # Historial directo, ajuste mínimo.
    factor_h2h = 1.0 + limitar(h2h * 0.015, -0.06, 0.06)

    lam = base * factor_forma * factor_elo * factor_fifa * factor_h2h

    # Eliminatoria: algo más conservador.
    lam = lam * FACTOR_ELIMINATORIA

    return limitar(lam, LAMBDA_MIN, LAMBDA_MAX)


def calibrar_lambda_modelo(lambda_raw, lambda_base):
    """
    Corrige salida del modelo entrenado.

    Si lambda_raw es extrema, se le baja mucho el peso.
    Si lambda_raw es razonable, se mezcla con la lambda futbolística base.
    """

    raw = safe_float(lambda_raw, lambda_base)

    if raw <= 0 or math.isnan(raw) or math.isinf(raw):
        peso_modelo = 0.10
        raw_corr = lambda_base
    elif raw > 5.0:
        # Caso típico que causaba 3-3 para todos tras el recorte.
        peso_modelo = 0.10
        raw_corr = lambda_base
    elif raw > 3.2:
        peso_modelo = 0.25
        raw_corr = 3.2
    elif raw < 0.25:
        peso_modelo = 0.20
        raw_corr = 0.35
    else:
        peso_modelo = 0.45
        raw_corr = raw

    lam = peso_modelo * raw_corr + (1.0 - peso_modelo) * lambda_base

    return limitar(lam, LAMBDA_MIN, LAMBDA_MAX), peso_modelo


def obtener_lambdas_partido(fila, modelo_home, modelo_away):
    # Usar exactamente las columnas con las que fue entrenado el modelo.
    columnas_modelo = list(getattr(modelo_home, "feature_names_in_", VARIABLES))

    equivalencias = {
        "elo_home": ["elo_home", "home_elo", "elo_local"],
        "elo_away": ["elo_away", "away_elo", "elo_visitante"],
        "fifa_home": ["fifa_home", "home_fifa_rank", "fifa_local"],
        "fifa_away": ["fifa_away", "away_fifa_rank", "fifa_visitante"],
        "home_elo": ["home_elo", "elo_home", "elo_local"],
        "away_elo": ["away_elo", "elo_away", "elo_visitante"],
        "home_fifa_rank": ["home_fifa_rank", "fifa_home", "fifa_local"],
        "away_fifa_rank": ["away_fifa_rank", "fifa_away", "fifa_visitante"],
    }

    datos = {}

    for col in columnas_modelo:
        valor = None

        if col in fila.index:
            valor = fila.get(col)
        elif col in equivalencias:
            for alt in equivalencias[col]:
                if alt in fila.index:
                    valor = fila.get(alt)
                    break

        datos[col] = safe_float(valor, 0)

    X = pd.DataFrame([datos], columns=columnas_modelo)

    raw_home = safe_float(modelo_home.predict(X)[0], MEDIA_GOLES_EQUIPO)
    raw_away = safe_float(modelo_away.predict(X)[0], MEDIA_GOLES_EQUIPO)

    base_home = lambda_por_forma_y_rival(fila, "home")
    base_away = lambda_por_forma_y_rival(fila, "away")

    lam_home, peso_home = calibrar_lambda_modelo(raw_home, base_home)
    lam_away, peso_away = calibrar_lambda_modelo(raw_away, base_away)

    return {
        "lambda_home_raw": raw_home,
        "lambda_away_raw": raw_away,
        "lambda_home_base": base_home,
        "lambda_away_base": base_away,
        "lambda_home_final": lam_home,
        "lambda_away_final": lam_away,
        "peso_modelo_home": peso_home,
        "peso_modelo_away": peso_away
    }


# ============================================================
# POISSON + DIXON-COLES
# ============================================================

def factor_dixon_coles(goles_home, goles_away, lambda_home, lambda_away, rho):
    if goles_home == 0 and goles_away == 0:
        factor = 1 - (lambda_home * lambda_away * rho)
    elif goles_home == 0 and goles_away == 1:
        factor = 1 + (lambda_home * rho)
    elif goles_home == 1 and goles_away == 0:
        factor = 1 + (lambda_away * rho)
    elif goles_home == 1 and goles_away == 1:
        factor = 1 - rho
    else:
        factor = 1.0

    return max(factor, 0.01)


def crear_matriz_poisson_dixon_coles(lambda_home, lambda_away):
    matriz = []

    for goles_home in range(MAX_GOLES + 1):
        fila = []

        for goles_away in range(MAX_GOLES + 1):
            prob_base = poisson.pmf(goles_home, lambda_home) * poisson.pmf(goles_away, lambda_away)

            ajuste_dc = factor_dixon_coles(
                goles_home=goles_home,
                goles_away=goles_away,
                lambda_home=lambda_home,
                lambda_away=lambda_away,
                rho=RHO_DIXON_COLES
            )

            fila.append(prob_base * ajuste_dc)

        matriz.append(fila)

    matriz = pd.DataFrame(matriz)

    total = matriz.values.sum()

    if total > 0:
        matriz = matriz / total

    return matriz


def calcular_probabilidades_1x2(matriz):
    prob_home = 0
    prob_draw = 0
    prob_away = 0

    for i in range(matriz.shape[0]):
        for j in range(matriz.shape[1]):
            prob = float(matriz.iloc[i, j])

            if i > j:
                prob_home += prob
            elif i == j:
                prob_draw += prob
            else:
                prob_away += prob

    return prob_home, prob_draw, prob_away


def obtener_top10_marcadores(matriz, match_id, home_team, away_team):
    filas = []

    for goles_home in range(matriz.shape[0]):
        for goles_away in range(matriz.shape[1]):
            filas.append({
                "match_id": int(match_id),
                "home_team": home_team,
                "away_team": away_team,
                "score": str(goles_home) + "-" + str(goles_away),
                "home_goals": goles_home,
                "away_goals": goles_away,
                "prob_score_percent": round(float(matriz.iloc[goles_home, goles_away]) * 100, 3),
                "modelo_probabilistico": "Poisson + Dixon-Coles"
            })

    top10 = pd.DataFrame(filas)

    top10 = top10.sort_values(
        by="prob_score_percent",
        ascending=False
    ).head(10).reset_index(drop=True)

    top10["ranking_score"] = range(1, len(top10) + 1)

    return top10


# ============================================================
# GANADOR REAL SI YA HAY RESULTADO
# ============================================================

def obtener_ganador_real(fila):
    home_score = pd.to_numeric(fila.get("home_score"), errors="coerce")
    away_score = pd.to_numeric(fila.get("away_score"), errors="coerce")
    pen_home = pd.to_numeric(fila.get("pen_home"), errors="coerce")
    pen_away = pd.to_numeric(fila.get("pen_away"), errors="coerce")

    if pd.isna(home_score) or pd.isna(away_score):
        return None

    home_team = fila["home_team"]
    away_team = fila["away_team"]

    home_score = int(home_score)
    away_score = int(away_score)

    if home_score > away_score:
        return {
            "winner": home_team,
            "source": "REAL_90_120",
            "score": str(home_score) + "-" + str(away_score)
        }

    if away_score > home_score:
        return {
            "winner": away_team,
            "source": "REAL_90_120",
            "score": str(home_score) + "-" + str(away_score)
        }

    if not pd.isna(pen_home) and not pd.isna(pen_away):
        pen_home = int(pen_home)
        pen_away = int(pen_away)

        if pen_home > pen_away:
            return {
                "winner": home_team,
                "source": "REAL_PENALES",
                "score": str(home_score) + "-" + str(away_score)
                + " pen " + str(pen_home) + "-" + str(pen_away)
            }

        if pen_away > pen_home:
            return {
                "winner": away_team,
                "source": "REAL_PENALES",
                "score": str(home_score) + "-" + str(away_score)
                + " pen " + str(pen_home) + "-" + str(pen_away)
            }

    return None


# ============================================================
# TARJETAS
# ============================================================

def obtener_disciplina_equipo(fair_play, equipo):
    fila = fair_play[fair_play["team"] == equipo]

    if fila.empty:
        return {
            "card_points": 6.0,
            "cards_per_match": 1.5
        }

    fila = fila.iloc[0]

    yellow = safe_float(fila.get("yellow_cards"), 0)
    indirect_red = safe_float(fila.get("indirect_red_cards"), 0)
    direct_red = safe_float(fila.get("direct_red_cards"), 0)
    conduct = safe_float(fila.get("team_conduct_score"), 0)
    matches = safe_float(fila.get("matches_count"), 0)

    if conduct < 0:
        card_points = abs(conduct)
    else:
        card_points = yellow + 3 * indirect_red + 4 * direct_red

    if matches <= 0:
        cards_per_match = card_points / 4
    else:
        cards_per_match = card_points / matches

    cards_per_match = limitar(cards_per_match, 0.6, 4.5)

    return {
        "card_points": card_points,
        "cards_per_match": cards_per_match
    }


def clasificar_riesgo_tarjetas(valor):
    if valor < 2.0:
        return "BAJO"
    elif valor < 3.2:
        return "MEDIO"
    else:
        return "ALTO"


def predecir_tarjetas(fila, fair_play, lambda_home, lambda_away, prob_draw):
    home_team = fila["home_team"]
    away_team = fila["away_team"]

    home_disc = obtener_disciplina_equipo(fair_play, home_team)
    away_disc = obtener_disciplina_equipo(fair_play, away_team)

    diff_elo = safe_float(fila.get("diff_elo"), 0)
    diff_fifa = safe_float(fila.get("diff_fifa"), 0)

    partido_parejo = abs(lambda_home - lambda_away) <= 0.35
    empate_alto = prob_draw >= 0.25

    extra_eliminatoria = 0.30
    extra_parejo = 0.35 if partido_parejo else 0.00
    extra_empate = 0.20 if empate_alto else 0.00

    presion_home = 0.00
    presion_away = 0.00

    if diff_elo < -80 or diff_fifa < -10:
        presion_home += 0.25

    if diff_elo > 80 or diff_fifa > 10:
        presion_away += 0.25

    home_cards = (
        home_disc["cards_per_match"]
        + extra_eliminatoria
        + extra_parejo
        + extra_empate
        + presion_home
    )

    away_cards = (
        away_disc["cards_per_match"]
        + extra_eliminatoria
        + extra_parejo
        + extra_empate
        + presion_away
    )

    total_cards = home_cards + away_cards

    return {
        "match_id": int(fila["match_id"]),
        "home_team": home_team,
        "away_team": away_team,
        "home_cards_expected": round(home_cards, 2),
        "away_cards_expected": round(away_cards, 2),
        "total_cards_expected": round(total_cards, 2),
        "home_cards_risk": clasificar_riesgo_tarjetas(home_cards),
        "away_cards_risk": clasificar_riesgo_tarjetas(away_cards),
        "match_cards_risk": clasificar_riesgo_tarjetas(total_cards / 2),
        "home_card_points_base": round(home_disc["card_points"], 2),
        "away_card_points_base": round(away_disc["card_points"], 2),
        "context_knockout": 1,
        "context_even_match": int(partido_parejo),
        "context_high_draw_probability": int(empate_alto),
        "modelo_tarjetas": "Fair Play + contexto + rival"
    }


# ============================================================
# PREDICCION
# ============================================================

def predecir_partido(fila, modelo_home, modelo_away, fair_play):
    home_team = fila["home_team"]
    away_team = fila["away_team"]

    lambdas = obtener_lambdas_partido(
        fila=fila,
        modelo_home=modelo_home,
        modelo_away=modelo_away
    )

    lambda_home = lambdas["lambda_home_final"]
    lambda_away = lambdas["lambda_away_final"]

    matriz = crear_matriz_poisson_dixon_coles(
        lambda_home=lambda_home,
        lambda_away=lambda_away
    )

    prob_home, prob_draw, prob_away = calcular_probabilidades_1x2(matriz)

    prob_adv_home = prob_home + 0.5 * prob_draw
    prob_adv_away = prob_away + 0.5 * prob_draw

    top10 = obtener_top10_marcadores(
        matriz=matriz,
        match_id=fila["match_id"],
        home_team=home_team,
        away_team=away_team
    )

    marcador_mas_probable = top10.iloc[0]["score"]

    if prob_adv_home >= prob_adv_away:
        predicted_winner = home_team
    else:
        predicted_winner = away_team

    resultado_real = obtener_ganador_real(fila)

    if resultado_real is not None:
        final_winner = resultado_real["winner"]
        source = resultado_real["source"]
        final_score = resultado_real["score"]
        acierto = int(predicted_winner == final_winner)
    else:
        final_winner = predicted_winner
        source = "PREDICHO_MODELO"
        final_score = marcador_mas_probable
        acierto = pd.NA

    tarjetas = predecir_tarjetas(
        fila=fila,
        fair_play=fair_play,
        lambda_home=lambda_home,
        lambda_away=lambda_away,
        prob_draw=prob_draw
    )

    resumen = {
        "match_id": int(fila["match_id"]),
        "date": fila["date"],
        "round": fila.get("stage", fila.get("round", "OCTAVOS")),
        "home_team": home_team,
        "away_team": away_team,
        "lambda_home_raw": round(lambdas["lambda_home_raw"], 4),
        "lambda_away_raw": round(lambdas["lambda_away_raw"], 4),
        "lambda_home_base": round(lambdas["lambda_home_base"], 4),
        "lambda_away_base": round(lambdas["lambda_away_base"], 4),
        "lambda_home_final": round(lambda_home, 4),
        "lambda_away_final": round(lambda_away, 4),
        "peso_modelo_home": round(lambdas["peso_modelo_home"], 2),
        "peso_modelo_away": round(lambdas["peso_modelo_away"], 2),
        "rho_dixon_coles": RHO_DIXON_COLES,
        "prob_home_win_percent": round(prob_home * 100, 2),
        "prob_draw_percent": round(prob_draw * 100, 2),
        "prob_away_win_percent": round(prob_away * 100, 2),
        "prob_adv_home_percent": round(prob_adv_home * 100, 2),
        "prob_adv_away_percent": round(prob_adv_away * 100, 2),
        "predicted_score": marcador_mas_probable,
        "predicted_winner": predicted_winner,
        "final_score_used": final_score,
        "final_winner": final_winner,
        "source": source,
        "modelo_acerto_si_hay_resultado": acierto,
        "modelo_probabilistico": "Forma+rival+modelo calibrado + Poisson + Dixon-Coles"
    }

    return resumen, top10, tarjetas


def ejecutar_predicciones(fixtures, modelo_home, modelo_away, fair_play):
    resumenes = []
    top10_list = []
    tarjetas_list = []

    fixtures = fixtures.sort_values(["date", "match_id"]).reset_index(drop=True)

    for _, fila in fixtures.iterrows():
        resumen, top10, tarjetas = predecir_partido(
            fila=fila,
            modelo_home=modelo_home,
            modelo_away=modelo_away,
            fair_play=fair_play
        )

        resumenes.append(resumen)
        top10_list.append(top10)
        tarjetas_list.append(tarjetas)

    df_resumen = pd.DataFrame(resumenes)
    df_top10 = pd.concat(top10_list, ignore_index=True)
    df_tarjetas = pd.DataFrame(tarjetas_list)

    return df_resumen, df_top10, df_tarjetas


def construir_semifinales(df_resumen):
    cuartos = df_resumen[[
        "match_id",
        "date",
        "home_team",
        "away_team",
        "final_winner",
        "source"
    ]].copy()

    cuartos = cuartos.rename(columns={
        "match_id": "match_id_cuartos",
        "final_winner": "team"
    })

    cuartos["classification_type"] = "CLASIFICADO_A_SEMIFINALES"

    return cuartos


# ============================================================
# VISUALIZACION
# ============================================================

def imprimir_detalle_partido(df_resumen, df_top10, df_tarjetas, match_id):
    resumen = df_resumen[df_resumen["match_id"] == match_id]
    top10 = df_top10[df_top10["match_id"] == match_id]
    tarjetas = df_tarjetas[df_tarjetas["match_id"] == match_id]

    if resumen.empty:
        return

    print()
    print("========================================")
    print("DETALLE DEL PARTIDO", match_id)
    print("========================================")
    print(resumen.to_string(index=False))

    print()
    print("TOP 10 MARCADORES - POISSON + DIXON-COLES")
    print("----------------------------------------")
    print(top10[[
        "ranking_score",
        "score",
        "prob_score_percent"
    ]].to_string(index=False))

    print()
    print("PREDICCION DE TARJETAS")
    print("----------------------------------------")
    print(tarjetas.to_string(index=False))


def graficar_panel(df_resumen, df_tarjetas):
    tabla = df_resumen.merge(
        df_tarjetas[[
            "match_id",
            "home_cards_expected",
            "away_cards_expected",
            "total_cards_expected",
            "match_cards_risk"
        ]],
        on="match_id",
        how="left"
    )

    tabla = tabla[[
        "match_id",
        "home_team",
        "away_team",
        "final_score_used",
        "final_winner",
        "source",
        "prob_adv_home_percent",
        "prob_adv_away_percent",
        "home_cards_expected",
        "away_cards_expected",
        "total_cards_expected",
        "match_cards_risk"
    ]].copy()

    tabla.columns = [
        "ID",
        "Equipo 1",
        "Equipo 2",
        "Marcador",
        "Clasifica",
        "Fuente",
        "% Avanza E1",
        "% Avanza E2",
        "Tarj. E1",
        "Tarj. E2",
        "Total tarj.",
        "Riesgo"
    ]

    fig, ax = plt.subplots(figsize=(22, 10))
    ax.axis("off")

    fig.suptitle(
        "Mundial 2026 - Cuartos: Poisson + Dixon-Coles + tarjetas",
        fontsize=18,
        fontweight="bold"
    )

    tabla_plot = ax.table(
        cellText=tabla.values,
        colLabels=tabla.columns,
        cellLoc="center",
        loc="center"
    )

    tabla_plot.auto_set_font_size(False)
    tabla_plot.set_fontsize(8)
    tabla_plot.scale(1, 1.5)

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    plt.savefig(
        IMAGEN_PANEL,
        dpi=300,
        bbox_inches="tight"
    )

    print("Imagen guardada:", IMAGEN_PANEL)
    plt.show()


# ============================================================
# MAIN
# ============================================================

def main():
    fixtures, modelo_home, modelo_away, fair_play = cargar_datos()

    df_resumen, df_top10, df_tarjetas = ejecutar_predicciones(
        fixtures=fixtures,
        modelo_home=modelo_home,
        modelo_away=modelo_away,
        fair_play=fair_play
    )

    semifinales = construir_semifinales(df_resumen)

    df_resumen.to_csv(ARCHIVO_RESUMEN, index=False, encoding="utf-8")
    df_top10.to_csv(ARCHIVO_TOP10, index=False, encoding="utf-8")
    df_tarjetas.to_csv(ARCHIVO_TARJETAS, index=False, encoding="utf-8")
    semifinales.to_csv(ARCHIVO_SEMIFINALES, index=False, encoding="utf-8")

    print()
    print("========================================")
    print("RESUMEN CUARTOS A SEMIFINALES")
    print("========================================")
    print(df_resumen[[
        "match_id",
        "home_team",
        "away_team",
        "lambda_home_raw",
        "lambda_away_raw",
        "lambda_home_base",
        "lambda_away_base",
        "lambda_home_final",
        "lambda_away_final",
        "predicted_score",
        "predicted_winner",
        "final_score_used",
        "final_winner",
        "source",
        "modelo_acerto_si_hay_resultado",
        "prob_adv_home_percent",
        "prob_adv_away_percent"
    ]].to_string(index=False))

    print()
    print("========================================")
    print("PREDICCION DE TARJETAS")
    print("========================================")
    print(df_tarjetas[[
        "match_id",
        "home_team",
        "away_team",
        "home_cards_expected",
        "away_cards_expected",
        "total_cards_expected",
        "match_cards_risk"
    ]].to_string(index=False))

    imprimir_detalle_partido(df_resumen, df_top10, df_tarjetas, 97)
    imprimir_detalle_partido(df_resumen, df_top10, df_tarjetas, 98)

    graficar_panel(df_resumen, df_tarjetas)

    print()
    print("========================================")
    print("ARCHIVOS GENERADOS")
    print("========================================")
    print(ARCHIVO_RESUMEN)
    print(ARCHIVO_TOP10)
    print(ARCHIVO_TARJETAS)
    print(ARCHIVO_SEMIFINALES)
    print(IMAGEN_PANEL)
    print("========================================")


if __name__ == "__main__":
    main()
