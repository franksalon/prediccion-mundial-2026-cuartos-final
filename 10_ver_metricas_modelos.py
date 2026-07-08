# ============================================================
# 10_ver_metricas_modelos.py
# Muestra SOLO las metricas de los modelos ya entrenados.
# ============================================================

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURACION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
OUT_DIR = BASE_DIR / "outputs"

ARCHIVO_COMPETENCIA = OUT_DIR / "competencia_modelos.csv"
ARCHIVO_MEJOR_MODELO = OUT_DIR / "mejor_modelo.txt"

ARCHIVO_RESUMEN = OUT_DIR / "resumen_metricas_modelos.csv"
IMAGEN_METRICAS = OUT_DIR / "panel_metricas_modelos.png"


# ============================================================
# UTILIDADES
# ============================================================

def cargar_competencia():
    if not ARCHIVO_COMPETENCIA.exists():
        raise FileNotFoundError(
            "No existe outputs/competencia_modelos.csv.\n"
            "Primero debes ejecutar una vez: python 02_entrenar_predecir_modelos.py\n"
            "Luego ya puedes usar este script sin volver a entrenar."
        )

    df = pd.read_csv(ARCHIVO_COMPETENCIA)
    df.columns = df.columns.str.strip()

    return df


def leer_mejor_modelo():
    if not ARCHIVO_MEJOR_MODELO.exists():
        return "No se encontró outputs/mejor_modelo.txt"

    return ARCHIVO_MEJOR_MODELO.read_text(encoding="utf-8").strip()


def detectar_columna_modelo(df):
    posibles = [
        "modelo",
        "model",
        "nombre_modelo",
        "Model",
        "Modelo"
    ]

    for col in posibles:
        if col in df.columns:
            return col

    return df.columns[0]


def detectar_metricas(df):
    columnas = df.columns.tolist()

    metricas = []

    palabras_metricas = [
        "mae",
        "rmse",
        "mse",
        "r2",
        "score",
        "accuracy",
        "mape",
        "error"
    ]

    for col in columnas:
        col_lower = col.lower()

        for palabra in palabras_metricas:
            if palabra in col_lower:
                metricas.append(col)
                break

    # Quitar duplicados manteniendo orden
    metricas = list(dict.fromkeys(metricas))

    return metricas


def ordenar_modelos(df, metricas):
    df_ordenado = df.copy()

    columnas_lower = {col.lower(): col for col in df_ordenado.columns}

    # Preferencia: menor RMSE, luego menor MAE, luego mayor R2/score.
    rmse_cols = [col for col in df_ordenado.columns if "rmse" in col.lower()]
    mae_cols = [col for col in df_ordenado.columns if "mae" in col.lower()]
    r2_cols = [col for col in df_ordenado.columns if "r2" in col.lower()]
    score_cols = [col for col in df_ordenado.columns if "score" in col.lower()]

    if len(rmse_cols) > 0:
        col = rmse_cols[0]
        df_ordenado[col] = pd.to_numeric(df_ordenado[col], errors="coerce")
        return df_ordenado.sort_values(by=col, ascending=True)

    if len(mae_cols) > 0:
        col = mae_cols[0]
        df_ordenado[col] = pd.to_numeric(df_ordenado[col], errors="coerce")
        return df_ordenado.sort_values(by=col, ascending=True)

    if len(r2_cols) > 0:
        col = r2_cols[0]
        df_ordenado[col] = pd.to_numeric(df_ordenado[col], errors="coerce")
        return df_ordenado.sort_values(by=col, ascending=False)

    if len(score_cols) > 0:
        col = score_cols[0]
        df_ordenado[col] = pd.to_numeric(df_ordenado[col], errors="coerce")
        return df_ordenado.sort_values(by=col, ascending=False)

    return df_ordenado


def crear_panel_metricas(df):
    df_panel = df.copy()

    # Limitar columnas si hay demasiadas
    if len(df_panel.columns) > 10:
        col_modelo = detectar_columna_modelo(df_panel)
        metricas = detectar_metricas(df_panel)
        columnas = [col_modelo] + metricas
        columnas = list(dict.fromkeys(columnas))
        df_panel = df_panel[columnas]

    # Redondear numeros
    for col in df_panel.columns:
        if pd.api.types.is_numeric_dtype(df_panel[col]):
            df_panel[col] = df_panel[col].round(4)

    fig, ax = plt.subplots(figsize=(16, 6))
    ax.axis("off")

    fig.suptitle(
        "Competencia de modelos - metricas guardadas",
        fontsize=16,
        fontweight="bold"
    )

    tabla = ax.table(
        cellText=df_panel.values,
        colLabels=df_panel.columns,
        cellLoc="center",
        loc="center"
    )

    tabla.auto_set_font_size(False)
    tabla.set_fontsize(8)
    tabla.scale(1, 1.5)

    plt.tight_layout(rect=[0, 0, 1, 0.93])

    plt.savefig(
        IMAGEN_METRICAS,
        dpi=300,
        bbox_inches="tight"
    )

    print("Imagen guardada:", IMAGEN_METRICAS)


# ============================================================
# MAIN
# ============================================================

def main():
    df = cargar_competencia()

    col_modelo = detectar_columna_modelo(df)
    metricas = detectar_metricas(df)

    df_ordenado = ordenar_modelos(df, metricas)

    df_ordenado.to_csv(
        ARCHIVO_RESUMEN,
        index=False,
        encoding="utf-8"
    )

    mejor_modelo_texto = leer_mejor_modelo()

    print()
    print("========================================")
    print("METRICAS DE LOS 5 MODELOS")
    print("========================================")
    print("Archivo leído:", ARCHIVO_COMPETENCIA)
    print("Filas:", len(df))
    print("Columnas:", len(df.columns))
    print()

    print("Columnas disponibles:")
    print(df.columns.tolist())

    print()
    print("Columna de modelo detectada:", col_modelo)
    print("Columnas de metricas detectadas:", metricas)

    print()
    print("========================================")
    print("TABLA ORDENADA DE MODELOS")
    print("========================================")
    print(df_ordenado.to_string(index=False))

    print()
    print("========================================")
    print("MEJOR MODELO GUARDADO")
    print("========================================")
    print(mejor_modelo_texto)

    crear_panel_metricas(df_ordenado)

    print()
    print("========================================")
    print("ARCHIVOS GENERADOS")
    print("========================================")
    print(ARCHIVO_RESUMEN)
    print(IMAGEN_METRICAS)
    print("========================================")


if __name__ == "__main__":
    main()
