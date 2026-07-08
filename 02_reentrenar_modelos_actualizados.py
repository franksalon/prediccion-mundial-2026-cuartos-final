# ============================================================
# 02_reentrenar_modelos_actualizados.py
# Reentrena los 5 modelos usando data/data_modelo.csv actualizado.
#
# Objetivo:
# - Usar la data general actualizada con resultados oficiales recientes.
# - Comparar modelos con métricas.
# - Guardar nuevamente los modelos .pkl.
# - Guardar mejor_modelo.txt.
# - Generar panel de competencia de modelos.
#
# Ejecutar después de actualizar la data:
# python 00_actualizar_datos_oficiales.py
# python 12_actualizar_fixtures_modelo_cuartos.py
# python 02_reentrenar_modelos_actualizados.py
# python 13_predecir_semifinales.py
# python 10_ver_metricas_modelos.py
# python 14_panel_integral_cuartos.py
# ============================================================

from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression, Ridge, PoissonRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


warnings.filterwarnings("ignore")


# ============================================================
# CONFIGURACION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUT_DIR = BASE_DIR / "outputs"

DATA_MODELO = DATA_DIR / "data_modelo.csv"

ARCHIVO_COMPETENCIA = OUT_DIR / "competencia_modelos.csv"
ARCHIVO_MEJOR_MODELO = OUT_DIR / "mejor_modelo.txt"
ARCHIVO_FEATURES = OUT_DIR / "columnas_modelo.pkl"

PANEL_MODELOS = OUT_DIR / "panel_competencia_modelos.png"
PANEL_METRICAS = OUT_DIR / "panel_metricas_modelos.png"

RANDOM_STATE = 42
TEST_SIZE = 0.20


# ============================================================
# MODELOS
# ============================================================

MODELOS = {
    "Regresion Lineal": LinearRegression(),
    "Ridge": Ridge(alpha=1.0, random_state=RANDOM_STATE),
    "Random Forest": RandomForestRegressor(
        n_estimators=250,
        max_depth=12,
        min_samples_leaf=2,
        random_state=RANDOM_STATE,
        n_jobs=-1
    ),
    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=250,
        learning_rate=0.04,
        max_depth=3,
        random_state=RANDOM_STATE
    ),
    "Poisson": PoissonRegressor(
        alpha=0.01,
        max_iter=1000
    )
}


# ============================================================
# FUNCIONES
# ============================================================

def cargar_data():
    if not DATA_MODELO.exists():
        raise FileNotFoundError(
            f"No existe {DATA_MODELO}. Primero actualiza/construye data_modelo.csv."
        )

    df = pd.read_csv(DATA_MODELO)
    df.columns = df.columns.str.strip()

    columnas_necesarias = ["home_score", "away_score"]

    for col in columnas_necesarias:
        if col not in df.columns:
            raise ValueError(f"Falta la columna obligatoria: {col}")

    df["home_score"] = pd.to_numeric(df["home_score"], errors="coerce")
    df["away_score"] = pd.to_numeric(df["away_score"], errors="coerce")

    df = df.dropna(subset=["home_score", "away_score"]).copy()

    df["home_score"] = df["home_score"].astype(float)
    df["away_score"] = df["away_score"].astype(float)

    return df


def seleccionar_features(df):
    # Variables principales esperadas del proyecto.
    candidatas_preferidas = [
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

    excluir = {
        "date",
        "home_team",
        "away_team",
        "home_score",
        "away_score",
        "tournament",
        "city",
        "country",
        "stage",
        "round",
        "match_id",
        "winner",
        "source",
        "notes"
    }

    features = [c for c in candidatas_preferidas if c in df.columns]

    # Si faltan varias candidatas, agregamos numéricas útiles automáticamente.
    numericas = []
    for col in df.columns:
        if col in excluir:
            continue

        serie = pd.to_numeric(df[col], errors="coerce")
        if serie.notna().sum() > 0:
            numericas.append(col)

    for col in numericas:
        if col not in features:
            features.append(col)

    if not features:
        raise ValueError("No se encontraron variables predictoras numéricas.")

    return features


def preparar_xy(df, features):
    data = df.copy()

    for col in features:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    data[features] = data[features].replace([np.inf, -np.inf], np.nan)
    data[features] = data[features].fillna(data[features].median(numeric_only=True))
    data[features] = data[features].fillna(0)

    X = data[features]
    y_home = data["home_score"].clip(lower=0)
    y_away = data["away_score"].clip(lower=0)

    return X, y_home, y_away


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def evaluar_modelo(nombre, modelo_home, modelo_away, X_test, y_home_test, y_away_test):
    pred_home = np.clip(modelo_home.predict(X_test), 0, None)
    pred_away = np.clip(modelo_away.predict(X_test), 0, None)

    mae_home = mean_absolute_error(y_home_test, pred_home)
    mae_away = mean_absolute_error(y_away_test, pred_away)

    rmse_home = rmse(y_home_test, pred_home)
    rmse_away = rmse(y_away_test, pred_away)

    r2_home = r2_score(y_home_test, pred_home)
    r2_away = r2_score(y_away_test, pred_away)

    mae_promedio = (mae_home + mae_away) / 2
    rmse_promedio = (rmse_home + rmse_away) / 2
    r2_promedio = (r2_home + r2_away) / 2

    # Score propio: menor RMSE y menor MAE es mejor; mayor R2 ayuda.
    score_compuesto = (1 / (1 + rmse_promedio)) * 0.55 + (1 / (1 + mae_promedio)) * 0.35 + max(r2_promedio, -1) * 0.10

    return {
        "modelo": nombre,
        "mae_home": round(mae_home, 4),
        "mae_away": round(mae_away, 4),
        "mae_promedio": round(mae_promedio, 4),
        "rmse_home": round(rmse_home, 4),
        "rmse_away": round(rmse_away, 4),
        "rmse_promedio": round(rmse_promedio, 4),
        "r2_home": round(r2_home, 4),
        "r2_away": round(r2_away, 4),
        "r2_promedio": round(r2_promedio, 4),
        "score_compuesto": round(score_compuesto, 4)
    }


def entrenar_modelos(X_train, X_test, y_home_train, y_home_test, y_away_train, y_away_test):
    resultados = []
    modelos_entrenados = {}

    for nombre, modelo_base in MODELOS.items():
        print(f"Entrenando: {nombre}")

        modelo_home = joblib.load(joblib.dump(modelo_base, OUT_DIR / "_tmp_model.pkl")[0])
        modelo_away = joblib.load(OUT_DIR / "_tmp_model.pkl")

        # Evitar reutilización accidental del mismo objeto.
        if nombre == "Regresion Lineal":
            modelo_home = LinearRegression()
            modelo_away = LinearRegression()
        elif nombre == "Ridge":
            modelo_home = Ridge(alpha=1.0, random_state=RANDOM_STATE)
            modelo_away = Ridge(alpha=1.0, random_state=RANDOM_STATE)
        elif nombre == "Random Forest":
            modelo_home = RandomForestRegressor(
                n_estimators=250,
                max_depth=12,
                min_samples_leaf=2,
                random_state=RANDOM_STATE,
                n_jobs=-1
            )
            modelo_away = RandomForestRegressor(
                n_estimators=250,
                max_depth=12,
                min_samples_leaf=2,
                random_state=RANDOM_STATE,
                n_jobs=-1
            )
        elif nombre == "Gradient Boosting":
            modelo_home = GradientBoostingRegressor(
                n_estimators=250,
                learning_rate=0.04,
                max_depth=3,
                random_state=RANDOM_STATE
            )
            modelo_away = GradientBoostingRegressor(
                n_estimators=250,
                learning_rate=0.04,
                max_depth=3,
                random_state=RANDOM_STATE
            )
        elif nombre == "Poisson":
            modelo_home = PoissonRegressor(alpha=0.01, max_iter=1000)
            modelo_away = PoissonRegressor(alpha=0.01, max_iter=1000)

        modelo_home.fit(X_train, y_home_train)
        modelo_away.fit(X_train, y_away_train)

        met = evaluar_modelo(
            nombre,
            modelo_home,
            modelo_away,
            X_test,
            y_home_test,
            y_away_test
        )

        resultados.append(met)
        modelos_entrenados[nombre] = {
            "home": modelo_home,
            "away": modelo_away
        }

    tmp = OUT_DIR / "_tmp_model.pkl"
    if tmp.exists():
        tmp.unlink()

    return pd.DataFrame(resultados), modelos_entrenados


def nombre_archivo_modelo(nombre):
    return (
        nombre.lower()
        .replace(" ", "_")
        .replace("ó", "o")
        .replace("í", "i")
        .replace("á", "a")
        .replace("é", "e")
        .replace("ú", "u")
    )


def guardar_modelos(modelos_entrenados, mejor_modelo, features):
    for nombre, modelos in modelos_entrenados.items():
        nombre_file = nombre_archivo_modelo(nombre)

        joblib.dump(
            modelos["home"],
            OUT_DIR / f"modelo_home_{nombre_file}.pkl"
        )

        joblib.dump(
            modelos["away"],
            OUT_DIR / f"modelo_away_{nombre_file}.pkl"
        )

    mejor_file = nombre_archivo_modelo(mejor_modelo)

    # Archivos estándar usados por scripts de predicción.
    joblib.dump(
        modelos_entrenados[mejor_modelo]["home"],
        OUT_DIR / "modelo_home.pkl"
    )

    joblib.dump(
        modelos_entrenados[mejor_modelo]["away"],
        OUT_DIR / "modelo_away.pkl"
    )

    joblib.dump(features, ARCHIVO_FEATURES)

    ARCHIVO_MEJOR_MODELO.write_text(
        mejor_modelo,
        encoding="utf-8"
    )

    print()
    print("Mejor modelo guardado:", mejor_modelo)
    print("Modelo home:", OUT_DIR / "modelo_home.pkl")
    print("Modelo away:", OUT_DIR / "modelo_away.pkl")
    print("Columnas modelo:", ARCHIVO_FEATURES)


def crear_panel_competencia(df_metricas):
    tabla = df_metricas.copy()

    columnas = [
        "modelo",
        "mae_promedio",
        "rmse_promedio",
        "r2_promedio",
        "score_compuesto"
    ]

    columnas = [c for c in columnas if c in tabla.columns]
    tabla = tabla[columnas]

    for col in tabla.columns:
        if col != "modelo":
            tabla[col] = pd.to_numeric(tabla[col], errors="coerce").round(4)

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.axis("off")

    fig.suptitle(
        "Competencia de modelos reentrenados con data actualizada",
        fontsize=15,
        fontweight="bold"
    )

    t = ax.table(
        cellText=tabla.values,
        colLabels=tabla.columns,
        cellLoc="center",
        loc="center"
    )

    t.auto_set_font_size(False)
    t.set_fontsize(9)
    t.scale(1, 1.5)

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.savefig(PANEL_MODELOS, dpi=300, bbox_inches="tight")
    plt.savefig(PANEL_METRICAS, dpi=300, bbox_inches="tight")
    plt.close()

    print("Panel guardado:", PANEL_MODELOS)
    print("Panel guardado:", PANEL_METRICAS)


def main():
    OUT_DIR.mkdir(exist_ok=True)

    df = cargar_data()
    features = seleccionar_features(df)
    X, y_home, y_away = preparar_xy(df, features)

    print()
    print("========================================")
    print("REENTRENAMIENTO DE MODELOS")
    print("========================================")
    print("Archivo usado:", DATA_MODELO)
    print("Filas usadas:", len(df))
    print("Variables usadas:", len(features))
    print()
    print("Columnas predictoras:")
    for col in features:
        print("-", col)

    X_train, X_test, y_home_train, y_home_test, y_away_train, y_away_test = train_test_split(
        X,
        y_home,
        y_away,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )

    metricas, modelos_entrenados = entrenar_modelos(
        X_train,
        X_test,
        y_home_train,
        y_home_test,
        y_away_train,
        y_away_test
    )

    metricas = metricas.sort_values(
        by=["rmse_promedio", "mae_promedio", "score_compuesto"],
        ascending=[True, True, False]
    ).reset_index(drop=True)

    mejor_modelo = metricas.iloc[0]["modelo"]

    metricas.to_csv(
        ARCHIVO_COMPETENCIA,
        index=False,
        encoding="utf-8"
    )

    # También dejamos el resumen actualizado para el script 10.
    metricas.to_csv(
        OUT_DIR / "resumen_metricas_modelos.csv",
        index=False,
        encoding="utf-8"
    )

    guardar_modelos(modelos_entrenados, mejor_modelo, features)
    crear_panel_competencia(metricas)

    print()
    print("========================================")
    print("METRICAS ACTUALIZADAS")
    print("========================================")
    print(metricas.to_string(index=False))

    print()
    print("========================================")
    print("ARCHIVOS GENERADOS")
    print("========================================")
    print(ARCHIVO_COMPETENCIA)
    print(OUT_DIR / "resumen_metricas_modelos.csv")
    print(ARCHIVO_MEJOR_MODELO)
    print(PANEL_MODELOS)
    print(PANEL_METRICAS)
    print("========================================")


if __name__ == "__main__":
    main()
