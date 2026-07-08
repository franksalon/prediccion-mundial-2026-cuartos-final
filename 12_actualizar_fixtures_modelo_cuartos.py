# ============================================================
# 12_actualizar_fixtures_modelo_cuartos.py
# Construye data/fixtures_cuartos_modelo.csv para predecir
# cuartos de final.
#
# Usa:
# - data/results.csv
# - data/fixtures_16avos.csv
# - outputs/prediccion_octavos_a_cuartos.csv
# - data/fixtures_cuartos.csv
#
# El archivo results.csv se actualiza como escenario de cuartos:
# resultados reales + resultados predichos de octavos pendientes.
# ============================================================

from pathlib import Path
import importlib.util
import shutil
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUT_DIR = BASE_DIR / "outputs"

ARCHIVO_09 = BASE_DIR / "09_actualizar_fixtures_modelo_eliminatorias.py"

ARCHIVO_RESULTS = DATA_DIR / "results.csv"
ARCHIVO_FIXTURES_MODELO = DATA_DIR / "fixtures_modelo.csv"
ARCHIVO_FIXTURES_16AVOS = DATA_DIR / "fixtures_16avos.csv"
ARCHIVO_PREDICCION_OCTAVOS = OUT_DIR / "prediccion_octavos_a_cuartos.csv"
ARCHIVO_FIXTURES_CUARTOS = DATA_DIR / "fixtures_cuartos.csv"

ARCHIVO_BACKUP = DATA_DIR / "results_backup_antes_cuartos.csv"
ARCHIVO_RESULTS_ESCENARIO = DATA_DIR / "results_actualizado_cuartos.csv"
ARCHIVO_FIXTURES_CUARTOS_MODELO = DATA_DIR / "fixtures_cuartos_modelo.csv"
ARCHIVO_FIXTURES_MODELO_CUARTOS = DATA_DIR / "fixtures_modelo_actualizado_cuartos.csv"


def cargar_modulo_09():
    if not ARCHIVO_09.exists():
        raise FileNotFoundError("No existe 09_actualizar_fixtures_modelo_eliminatorias.py")

    spec = importlib.util.spec_from_file_location("modulo_09", ARCHIVO_09)
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)

    return modulo


def leer_csv(ruta):
    if not ruta.exists():
        raise FileNotFoundError("No existe el archivo: " + str(ruta))

    df = pd.read_csv(ruta)
    df.columns = df.columns.str.strip()
    return df


def normalizar_fecha(df, columna="date"):
    df[columna] = pd.to_datetime(df[columna], errors="coerce").dt.strftime("%Y-%m-%d")
    return df


def parse_score(score):
    if pd.isna(score):
        return None, None

    score = str(score).strip()

    if "-" not in score:
        return None, None

    partes = score.split("-")

    if len(partes) != 2:
        return None, None

    try:
        return int(partes[0]), int(partes[1])
    except ValueError:
        return None, None


def construir_resultados_desde_octavos(pred_octavos, columnas_results):
    filas = []

    pred = pred_octavos.copy()
    pred = normalizar_fecha(pred, "date")

    for _, fila in pred.iterrows():
        score = fila.get("final_score_used", fila.get("predicted_score", ""))
        home_score, away_score = parse_score(score)

        if home_score is None or away_score is None:
            continue

        nuevo = {}

        for col in columnas_results:
            nuevo[col] = ""

        nuevo["date"] = fila["date"]
        nuevo["home_team"] = fila["home_team"]
        nuevo["away_team"] = fila["away_team"]
        nuevo["home_score"] = home_score
        nuevo["away_score"] = away_score

        if "tournament" in columnas_results:
            nuevo["tournament"] = "FIFA World Cup"

        if "neutral" in columnas_results:
            nuevo["neutral"] = True

        if "city" in columnas_results:
            nuevo["city"] = "Unknown"

        if "country" in columnas_results:
            nuevo["country"] = "United States/Canada/Mexico"

        filas.append(nuevo)

    return pd.DataFrame(filas)


def actualizar_results(results, nuevos):
    actualizado = results.copy()
    actualizado = normalizar_fecha(actualizado, "date")

    for _, fila in nuevos.iterrows():
        condicion = (
            (actualizado["date"] == fila["date"]) &
            (actualizado["home_team"] == fila["home_team"]) &
            (actualizado["away_team"] == fila["away_team"])
        )

        if condicion.any():
            idx = actualizado[condicion].index[-1]
            actualizado.loc[idx, "home_score"] = int(fila["home_score"])
            actualizado.loc[idx, "away_score"] = int(fila["away_score"])

            if "tournament" in actualizado.columns:
                actualizado.loc[idx, "tournament"] = "FIFA World Cup"

            if "neutral" in actualizado.columns:
                actualizado.loc[idx, "neutral"] = True

        else:
            actualizado = pd.concat(
                [actualizado, pd.DataFrame([fila])],
                ignore_index=True
            )

    actualizado["date_dt"] = pd.to_datetime(actualizado["date"], errors="coerce")
    actualizado = actualizado.sort_values(
        by=["date_dt", "home_team", "away_team"]
    ).drop(columns=["date_dt"]).reset_index(drop=True)

    return actualizado


def main():
    modulo_09 = cargar_modulo_09()

    results = leer_csv(ARCHIVO_RESULTS)
    fixtures_modelo = leer_csv(ARCHIVO_FIXTURES_MODELO)
    fixtures_16avos = leer_csv(ARCHIVO_FIXTURES_16AVOS)
    pred_octavos = leer_csv(ARCHIVO_PREDICCION_OCTAVOS)
    fixtures_cuartos = leer_csv(ARCHIVO_FIXTURES_CUARTOS)

    results = modulo_09.preparar_results(results)
    fixtures_16avos = normalizar_fecha(fixtures_16avos, "date")
    fixtures_cuartos = normalizar_fecha(fixtures_cuartos, "date")

    if not ARCHIVO_BACKUP.exists():
        shutil.copy2(ARCHIVO_RESULTS, ARCHIVO_BACKUP)

    # Primero asegura resultados de 16avos.
    results_actualizado = modulo_09.actualizar_results_con_16avos(
        results=results,
        fixtures_16avos=fixtures_16avos
    )

    results_actualizado = modulo_09.preparar_results(results_actualizado)

    # Luego agrega resultados reales/predichos de octavos para construir escenario de cuartos.
    nuevos_octavos = construir_resultados_desde_octavos(
        pred_octavos=pred_octavos,
        columnas_results=results_actualizado.columns.tolist()
    )

    results_actualizado = actualizar_results(
        results=results_actualizado,
        nuevos=nuevos_octavos
    )

    results_actualizado = modulo_09.preparar_results(results_actualizado)

    results_actualizado.to_csv(
        ARCHIVO_RESULTS,
        index=False,
        encoding="utf-8"
    )

    results_actualizado.to_csv(
        ARCHIVO_RESULTS_ESCENARIO,
        index=False,
        encoding="utf-8"
    )

    mapa_fifa, mapa_elo = modulo_09.construir_mapas_ranking(
        fixtures_modelo=fixtures_modelo
    )

    fixtures_cuartos_modelo = modulo_09.construir_fixture_modelo(
        fixture=fixtures_cuartos,
        results=results_actualizado,
        mapa_fifa=mapa_fifa,
        mapa_elo=mapa_elo,
        stage="CUARTOS"
    )

    fixtures_cuartos_modelo.to_csv(
        ARCHIVO_FIXTURES_CUARTOS_MODELO,
        index=False,
        encoding="utf-8"
    )

    # Acumulado: fase base + 16avos/octavos previos + cuartos.
    archivos_previos = []
    for nombre in [
        "fixtures_16avos_modelo.csv",
        "fixtures_octavos_modelo.csv"
    ]:
        ruta = DATA_DIR / nombre
        if ruta.exists():
            archivos_previos.append(pd.read_csv(ruta))

    partes = [fixtures_modelo] + archivos_previos + [fixtures_cuartos_modelo]

    fixtures_modelo_cuartos = pd.concat(
        partes,
        ignore_index=True,
        sort=False
    )

    fixtures_modelo_cuartos.to_csv(
        ARCHIVO_FIXTURES_MODELO_CUARTOS,
        index=False,
        encoding="utf-8"
    )

    print()
    print("========================================")
    print("ACTUALIZACION PARA CUARTOS COMPLETADA")
    print("========================================")
    print("Resultados escenario cuartos:", ARCHIVO_RESULTS_ESCENARIO)
    print("Fixtures cuartos modelo:", ARCHIVO_FIXTURES_CUARTOS_MODELO)
    print("Fixtures modelo acumulado:", ARCHIVO_FIXTURES_MODELO_CUARTOS)
    print()
    print("Fixtures de cuartos:")
    print(fixtures_cuartos_modelo[[
        "match_id",
        "date",
        "home_team",
        "away_team",
        "home_gf12",
        "home_ga12",
        "home_pts12",
        "away_gf12",
        "away_ga12",
        "away_pts12",
        "diff_fifa",
        "diff_elo"
    ]].to_string(index=False))
    print("========================================")


if __name__ == "__main__":
    main()
