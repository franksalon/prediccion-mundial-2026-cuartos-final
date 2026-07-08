# ============================================================
# 14_panel_integral_cuartos.py
# - ranking FIFA
# - ranking Elo
# - forma reciente
# - competencia de modelos
# - Poisson + Dixon-Coles
# - tarjetas/Fair Play
# No entrena modelos.
# ============================================================

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUT_DIR = BASE_DIR / "outputs"

PRED_CUARTOS = OUT_DIR / "prediccion_cuartos_a_semifinales.csv"
TARJETAS_CUARTOS = OUT_DIR / "prediccion_tarjetas_cuartos.csv"
FIXTURES_CUARTOS_MODELO = DATA_DIR / "fixtures_cuartos_modelo.csv"
COMPETENCIA = OUT_DIR / "competencia_modelos.csv"
MEJOR_MODELO = OUT_DIR / "mejor_modelo.txt"
RESUMEN_CONTEXTO = OUT_DIR / "analisis_contexto_cuartos.csv"
PANEL_INTEGRAL = OUT_DIR / "panel_integral_cuartos.png"
PANEL_MODELOS = OUT_DIR / "panel_competencia_modelos.png"


def leer_csv(ruta):
    if not ruta.exists():
        raise FileNotFoundError(f"No existe: {ruta}")
    df = pd.read_csv(ruta)
    df.columns = df.columns.str.strip()
    return df


def leer_mejor_modelo():
    if MEJOR_MODELO.exists():
        return MEJOR_MODELO.read_text(encoding="utf-8").strip()
    return "No disponible"


def crear_contexto():
    pred = leer_csv(PRED_CUARTOS)
    tarjetas = leer_csv(TARJETAS_CUARTOS)
    fixtures = leer_csv(FIXTURES_CUARTOS_MODELO)

    cols_fix = [
        "match_id", "home_team", "away_team",
        "home_gf12", "home_ga12", "home_pts12", "home_prev_matches",
        "away_gf12", "away_ga12", "away_pts12", "away_prev_matches",
        "fifa_home", "fifa_away", "diff_fifa",
        "elo_home", "elo_away", "diff_elo", "h2h"
    ]
    cols_fix = [c for c in cols_fix if c in fixtures.columns]

    contexto = pred.merge(fixtures[cols_fix], on=["match_id", "home_team", "away_team"], how="left")
    contexto = contexto.merge(
        tarjetas[["match_id", "home_cards_expected", "away_cards_expected", "total_cards_expected", "match_cards_risk"]],
        on="match_id",
        how="left"
    )

    # Normalizar columnas de contexto después de merges.
    # Algunas columnas pueden venir como _ctx, _x o _y después de la ponderación balanceada.
    columnas_necesarias = [
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
        "fifa_home",
        "fifa_away",
        "elo_home",
        "elo_away",
    ]

    for col in columnas_necesarias:
        if col not in contexto.columns:
            for alt in [f"{col}_ctx", f"{col}_x", f"{col}_y"]:
                if alt in contexto.columns:
                    contexto[col] = contexto[alt]
                    break

        if col not in contexto.columns:
            contexto[col] = 0

        contexto[col] = pd.to_numeric(contexto[col], errors="coerce").fillna(0)

    # Evitar división entre cero.
    contexto["home_prev_matches"] = contexto["home_prev_matches"].replace(0, 1)
    contexto["away_prev_matches"] = contexto["away_prev_matches"].replace(0, 1)

    contexto["forma_home_goles_pp"] = (contexto["home_gf12"] / contexto["home_prev_matches"]).round(2)
    contexto["forma_away_goles_pp"] = (contexto["away_gf12"] / contexto["away_prev_matches"]).round(2)
    contexto["forma_home_pts_pp"] = (contexto["home_pts12"] / contexto["home_prev_matches"]).round(2)
    contexto["forma_away_pts_pp"] = (contexto["away_pts12"] / contexto["away_prev_matches"]).round(2)

    columnas_finales = [
        "match_id", "date", "home_team", "away_team",
        "fifa_home", "fifa_away", "diff_fifa",
        "elo_home", "elo_away", "diff_elo",
        "h2h",
        "forma_home_goles_pp", "forma_away_goles_pp",
        "forma_home_pts_pp", "forma_away_pts_pp",
        "lambda_home_base", "lambda_away_base",
        "lambda_home_final", "lambda_away_final",
        "prob_adv_home_percent", "prob_adv_away_percent",
        "predicted_score", "predicted_winner", "final_winner", "source",
        "home_cards_expected", "away_cards_expected", "total_cards_expected", "match_cards_risk"
    ]
    columnas_finales = [c for c in columnas_finales if c in contexto.columns]
    contexto = contexto[columnas_finales]
    contexto.to_csv(RESUMEN_CONTEXTO, index=False, encoding="utf-8")
    return contexto


def crear_panel_modelos():
    df = leer_csv(COMPETENCIA)
    mejor = leer_mejor_modelo()

    columnas = [c for c in ["modelo", "mae_promedio", "rmse_promedio", "r2_promedio", "accuracy_resultado"] if c in df.columns]
    tabla = df[columnas].copy()

    for col in tabla.columns:
        if col != "modelo":
            tabla[col] = pd.to_numeric(tabla[col], errors="coerce").round(4)

    fig, ax = plt.subplots(figsize=(14, 4.5))
    ax.axis("off")
    fig.suptitle(f"Competencia de modelos | Mejor modelo: {mejor}", fontsize=15, fontweight="bold")

    t = ax.table(cellText=tabla.values, colLabels=tabla.columns, loc="center", cellLoc="center")
    t.auto_set_font_size(False)
    t.set_fontsize(8)
    t.scale(1, 1.5)

    plt.tight_layout(rect=[0, 0, 1, 0.90])
    plt.savefig(PANEL_MODELOS, dpi=300, bbox_inches="tight")
    print("Imagen guardada:", PANEL_MODELOS)


def crear_panel_integral(contexto):
    mejor = leer_mejor_modelo()

    tabla = contexto.copy()
    tabla = tabla[[
        "match_id", "home_team", "away_team",
        "fifa_home", "fifa_away", "diff_fifa",
        "elo_home", "elo_away", "diff_elo",
        "forma_home_goles_pp", "forma_away_goles_pp",
        "lambda_home_final", "lambda_away_final",
        "prob_adv_home_percent", "prob_adv_away_percent",
        "predicted_score", "final_winner",
        "total_cards_expected", "match_cards_risk"
    ]].copy()

    rename = {
        "match_id": "ID",
        "home_team": "Equipo 1",
        "away_team": "Equipo 2",
        "fifa_home": "FIFA E1",
        "fifa_away": "FIFA E2",
        "diff_fifa": "Dif FIFA",
        "elo_home": "Elo E1",
        "elo_away": "Elo E2",
        "diff_elo": "Dif Elo",
        "forma_home_goles_pp": "GF/P E1",
        "forma_away_goles_pp": "GF/P E2",
        "lambda_home_final": "λ E1",
        "lambda_away_final": "λ E2",
        "prob_adv_home_percent": "% E1",
        "prob_adv_away_percent": "% E2",
        "predicted_score": "Marcador",
        "final_winner": "Clasifica",
        "total_cards_expected": "Tarj.",
        "match_cards_risk": "Riesgo"
    }
    tabla = tabla.rename(columns=rename)

    for c in tabla.columns:
        if c not in ["Equipo 1", "Equipo 2", "Marcador", "Clasifica", "Riesgo"]:
            convertido = pd.to_numeric(tabla[c], errors="coerce")
        if convertido.notna().sum() > 0:
            tabla[c] = convertido
            if pd.api.types.is_numeric_dtype(tabla[c]):
                tabla[c] = tabla[c].round(2)

    fig, ax = plt.subplots(figsize=(24, 7))
    ax.axis("off")
    titulo = (
        "Mundial 2026 - Modelo predictivo de cuartos a semifinales\n"
        f"Ranking FIFA + Elo + forma reciente + contexto + Poisson/Dixon-Coles + Fair Play | Mejor modelo base: {mejor}"
    )
    fig.suptitle(titulo, fontsize=16, fontweight="bold")

    t = ax.table(cellText=tabla.values, colLabels=tabla.columns, loc="center", cellLoc="center")
    t.auto_set_font_size(False)
    t.set_fontsize(7.4)
    t.scale(1, 1.6)

    plt.tight_layout(rect=[0, 0, 1, 0.86])
    plt.savefig(PANEL_INTEGRAL, dpi=300, bbox_inches="tight")
    print("Imagen guardada:", PANEL_INTEGRAL)


def main():
    contexto = crear_contexto()
    crear_panel_modelos()
    crear_panel_integral(contexto)

    print("="*60)
    print("PANEL INTEGRAL DE CUARTOS GENERADO")
    print("="*60)
    print("CSV contexto:", RESUMEN_CONTEXTO)
    print("Panel modelos:", PANEL_MODELOS)
    print("Panel integral:", PANEL_INTEGRAL)
    print("="*60)


if __name__ == "__main__":
    main()
