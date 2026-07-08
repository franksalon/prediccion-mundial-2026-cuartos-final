# ============================================================
# 00_actualizar_datos_oficiales.py
# Actualiza el proyecto con resultados oficiales de octavos,
# tarjetas y fixtures oficiales de cuartos.
# No entrena modelos.
# ============================================================

from pathlib import Path
import shutil
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUT_DIR = BASE_DIR / "outputs"

RESULTS = DATA_DIR / "results.csv"
FIXTURES_OCTAVOS = DATA_DIR / "fixtures_octavos.csv"
FIXTURES_CUARTOS = DATA_DIR / "fixtures_cuartos.csv"
TARJETAS_OCTAVOS = DATA_DIR / "tarjetas_octavos_reales.csv"
FAIR_PLAY = DATA_DIR / "fair_play.csv"
CLASIFICADOS_CUARTOS = OUT_DIR / "clasificados_cuartos.csv"
PRED_OCTAVOS = OUT_DIR / "prediccion_octavos_a_cuartos.csv"
BACKUP_RESULTS = DATA_DIR / "results_backup_antes_update_oficial_cuartos.csv"


def asegurar_dirs():
    DATA_DIR.mkdir(exist_ok=True)
    OUT_DIR.mkdir(exist_ok=True)


def guardar_fixtures_oficiales():
    octavos = pd.DataFrame([
        [89,"2026-07-04","OCTAVOS","Canada","Morocco",0,3,"","","REAL_OFICIAL: Morocco clasifica a cuartos"],
        [90,"2026-07-04","OCTAVOS","Paraguay","France",0,1,"","","REAL_OFICIAL: France clasifica a cuartos"],
        [91,"2026-07-05","OCTAVOS","Brazil","Norway",1,2,"","","REAL_OFICIAL: Norway clasifica a cuartos"],
        [92,"2026-07-05","OCTAVOS","Mexico","England",2,3,"","","REAL_OFICIAL: England clasifica a cuartos"],
        [93,"2026-07-06","OCTAVOS","Portugal","Spain",0,1,"","","REAL_OFICIAL: Spain clasifica a cuartos"],
        [94,"2026-07-06","OCTAVOS","United States","Belgium",1,4,"","","REAL_OFICIAL: Belgium clasifica a cuartos"],
        [95,"2026-07-07","OCTAVOS","Argentina","Egypt",3,2,"","","REAL_OFICIAL: Argentina clasifica a cuartos"],
        [96,"2026-07-07","OCTAVOS","Switzerland","Colombia",0,0,4,3,"REAL_OFICIAL: Switzerland clasifica por penales 4-3"],
    ], columns=["match_id","date","round","home_team","away_team","home_score","away_score","pen_home","pen_away","source_note"])

    cuartos = pd.DataFrame([
        [97,"2026-07-09","CUARTOS","France","Morocco","","","","","REAL_OFICIAL: ganador 90 vs ganador 89"],
        [98,"2026-07-10","CUARTOS","Spain","Belgium","","","","","REAL_OFICIAL: ganador 93 vs ganador 94"],
        [99,"2026-07-11","CUARTOS","Norway","England","","","","","REAL_OFICIAL: ganador 91 vs ganador 92"],
        [100,"2026-07-11","CUARTOS","Argentina","Switzerland","","","","","REAL_OFICIAL: ganador 95 vs ganador 96"],
    ], columns=["match_id","date","round","home_team","away_team","home_score","away_score","pen_home","pen_away","source_note"])

    tarjetas = pd.DataFrame([
        [89,"Canada","Morocco",4,4,0,0,"FOX boxscore"],
        [90,"Paraguay","France",0,3,0,0,"FOX boxscore"],
        [91,"Brazil","Norway",1,0,0,0,"FOX play-by-play"],
        [92,"Mexico","England",2,4,0,1,"FOX boxscore"],
        [93,"Portugal","Spain",2,1,0,0,"FOX boxscore"],
        [94,"United States","Belgium",2,0,0,0,"FOX boxscore"],
        [95,"Argentina","Egypt",0,2,0,0,"FOX play-by-play"],
        [96,"Switzerland","Colombia",3,2,0,0,"FOX play-by-play"],
    ], columns=["match_id","home_team","away_team","yellow_home","yellow_away","red_home","red_away","source_note"])

    octavos.to_csv(FIXTURES_OCTAVOS, index=False, encoding="utf-8")
    cuartos.to_csv(FIXTURES_CUARTOS, index=False, encoding="utf-8")
    tarjetas.to_csv(TARJETAS_OCTAVOS, index=False, encoding="utf-8")

    return octavos, cuartos, tarjetas


def ganador_octavos(fila):
    hs = int(fila["home_score"])
    aas = int(fila["away_score"])
    if hs > aas:
        return fila["home_team"]
    if aas > hs:
        return fila["away_team"]
    ph = int(fila["pen_home"])
    pa = int(fila["pen_away"])
    return fila["home_team"] if ph > pa else fila["away_team"]


def actualizar_results(octavos):
    if not RESULTS.exists():
        print("ADVERTENCIA: no existe data/results.csv; no se actualizó histórico.")
        return

    if not BACKUP_RESULTS.exists():
        shutil.copy2(RESULTS, BACKUP_RESULTS)

    results = pd.read_csv(RESULTS)
    results.columns = results.columns.str.strip()
    results["date"] = pd.to_datetime(results["date"], errors="coerce").dt.strftime("%Y-%m-%d")

    for _, r in octavos.iterrows():
        nuevo = {col: "" for col in results.columns}
        nuevo["date"] = r["date"]
        nuevo["home_team"] = r["home_team"]
        nuevo["away_team"] = r["away_team"]
        nuevo["home_score"] = int(r["home_score"])
        nuevo["away_score"] = int(r["away_score"])
        if "tournament" in results.columns:
            nuevo["tournament"] = "FIFA World Cup"
        if "neutral" in results.columns:
            nuevo["neutral"] = True
        if "city" in results.columns:
            nuevo["city"] = "Unknown"
        if "country" in results.columns:
            nuevo["country"] = "United States/Canada/Mexico"

        mask = (
            (results["date"] == r["date"]) &
            (results["home_team"] == r["home_team"]) &
            (results["away_team"] == r["away_team"])
        )
        if mask.any():
            idx = results[mask].index[-1]
            for k, v in nuevo.items():
                results.loc[idx, k] = v
        else:
            results = pd.concat([results, pd.DataFrame([nuevo])], ignore_index=True)

    results["_date"] = pd.to_datetime(results["date"], errors="coerce")
    results = results.sort_values(["_date", "home_team", "away_team"]).drop(columns=["_date"]).reset_index(drop=True)
    results.to_csv(RESULTS, index=False, encoding="utf-8")


def actualizar_fair_play(tarjetas):
    if FAIR_PLAY.exists():
        fair = pd.read_csv(FAIR_PLAY)
        fair.columns = fair.columns.str.strip()
    else:
        fair = pd.DataFrame(columns=["team","yellow_cards","indirect_red_cards","direct_red_cards","team_conduct_score","matches_count","fair_play_points","cards_per_match","source_note"])

    for col in ["yellow_cards","indirect_red_cards","direct_red_cards","team_conduct_score","matches_count","fair_play_points","cards_per_match"]:
        if col not in fair.columns:
            fair[col] = 0
        fair[col] = pd.to_numeric(fair[col], errors="coerce").fillna(0)
    if "source_note" not in fair.columns:
        fair["source_note"] = ""

    def add_team(team, y, red):
        nonlocal fair
        mask = fair["team"].astype(str).str.strip().eq(team)
        if not mask.any():
            fair = pd.concat([fair, pd.DataFrame([{
                "team": team, "yellow_cards": 0, "indirect_red_cards": 0, "direct_red_cards": 0,
                "team_conduct_score": 0, "matches_count": 0, "fair_play_points": 0, "cards_per_match": 0,
                "source_note": "agregado desde actualización oficial"
            }])], ignore_index=True)
            mask = fair["team"].astype(str).str.strip().eq(team)
        idx = fair[mask].index[0]
        fair.loc[idx, "yellow_cards"] += int(y)
        fair.loc[idx, "direct_red_cards"] += int(red)
        fair.loc[idx, "matches_count"] += 1
        puntos = fair.loc[idx, "yellow_cards"] + 3*fair.loc[idx, "indirect_red_cards"] + 4*fair.loc[idx, "direct_red_cards"]
        fair.loc[idx, "fair_play_points"] = puntos
        fair.loc[idx, "team_conduct_score"] = -puntos
        fair.loc[idx, "cards_per_match"] = round(puntos / max(fair.loc[idx, "matches_count"], 1), 4)
        fair.loc[idx, "source_note"] = "Actualizado con tarjetas oficiales de octavos"

    for _, r in tarjetas.iterrows():
        add_team(r["home_team"], r["yellow_home"], r["red_home"])
        add_team(r["away_team"], r["yellow_away"], r["red_away"])

    fair.to_csv(FAIR_PLAY, index=False, encoding="utf-8")


def actualizar_clasificados_y_pred(octavos):
    filas = []
    for _, r in octavos.iterrows():
        win = ganador_octavos(r)
        score = f"{int(r['home_score'])}-{int(r['away_score'])}"
        if int(r["home_score"]) == int(r["away_score"]):
            score += f" pen {int(r['pen_home'])}-{int(r['pen_away'])}"
        filas.append({
            "match_id_octavos": int(r["match_id"]),
            "date": r["date"],
            "home_team": r["home_team"],
            "away_team": r["away_team"],
            "team": win,
            "source": "REAL_OFICIAL",
            "classification_type": "CLASIFICADO_A_CUARTOS",
            "score": score
        })
    pd.DataFrame(filas).to_csv(CLASIFICADOS_CUARTOS, index=False, encoding="utf-8")

    if PRED_OCTAVOS.exists():
        pred = pd.read_csv(PRED_OCTAVOS)
        pred.columns = pred.columns.str.strip()
        for _, r in octavos.iterrows():
            mask = pred["match_id"].astype(int).eq(int(r["match_id"]))
            if not mask.any():
                continue
            win = ganador_octavos(r)
            score = f"{int(r['home_score'])}-{int(r['away_score'])}"
            if int(r["home_score"]) == int(r["away_score"]):
                score += f" pen {int(r['pen_home'])}-{int(r['pen_away'])}"
            for col in ["predicted_score", "final_score_used"]:
                if col in pred.columns:
                    pred.loc[mask, col] = score
            for col in ["predicted_winner", "final_winner"]:
                if col in pred.columns:
                    pred.loc[mask, col] = win
            if "source" in pred.columns:
                pred.loc[mask, "source"] = "REAL_OFICIAL"
            if "prob_adv_home_percent" in pred.columns and "prob_adv_away_percent" in pred.columns:
                if win == r["home_team"]:
                    pred.loc[mask, "prob_adv_home_percent"] = 100.0
                    pred.loc[mask, "prob_adv_away_percent"] = 0.0
                else:
                    pred.loc[mask, "prob_adv_home_percent"] = 0.0
                    pred.loc[mask, "prob_adv_away_percent"] = 100.0
        pred.to_csv(PRED_OCTAVOS, index=False, encoding="utf-8")


def main():
    asegurar_dirs()
    octavos, cuartos, tarjetas = guardar_fixtures_oficiales()
    actualizar_results(octavos)
    actualizar_fair_play(tarjetas)
    actualizar_clasificados_y_pred(octavos)

    print("="*60)
    print("DATOS OFICIALES ACTUALIZADOS")
    print("="*60)
    print("Octavos oficiales:")
    print(octavos.to_string(index=False))
    print("\nCuartos oficiales:")
    print(cuartos.to_string(index=False))
    print("\nTarjetas oficiales de octavos:")
    print(tarjetas.to_string(index=False))
    print("="*60)


if __name__ == "__main__":
    main()
