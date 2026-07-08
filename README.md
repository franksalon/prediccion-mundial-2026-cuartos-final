# Modelo predictivo para cuartos de final del Mundial 2026

Repositorio académico orientado a la **predicción de semifinalistas del Mundial 2026** a partir de los cruces reales de cuartos de final.

El proyecto combina resultados oficiales actualizados, ranking Elo, puntaje FIFA, forma reciente, contexto competitivo, modelos de regresión, distribución de Poisson, análisis de tarjetas y una ponderación balanceada para estimar qué selecciones tienen mayor probabilidad de avanzar a semifinales.

---

## Integrantes

- Jean Frank Bustamante Vela
- Miguel Angel Marreros Cortegana
- Valentin Fernandez Campos
- Frank Salon Trigoso

---

## Enlace del repositorio

```text
https://github.com/Jeanki07/prediccion-mundial-2026-cuartos-final
```

---

## Objetivo general

Desarrollar un sistema predictivo que analice los partidos de cuartos de final del Mundial 2026 y estime los equipos clasificados a semifinales mediante modelos estadísticos y variables deportivas de contexto.

---

## Cruces analizados

```text
France vs Morocco
Spain vs Belgium
Norway vs England
Argentina vs Switzerland
```

El repositorio ya no se enfoca en predecir quién llega a cuartos. Los cruces de cuartos se consideran definidos y el objetivo es proyectar quién avanza a semifinales.

---

## Metodología

```text
1. Actualización de resultados oficiales.
2. Uso de fixtures reales de cuartos de final.
3. Construcción de variables deportivas de contexto.
4. Reentrenamiento de modelos con la data general actualizada.
5. Comparación de modelos mediante métricas.
6. Selección del mejor modelo base.
7. Estimación de goles esperados.
8. Cálculo probabilístico de marcadores mediante Poisson.
9. Aplicación de ponderación balanceada.
10. Predicción de semifinalistas.
11. Análisis de tarjetas y Fair Play.
12. Generación de paneles visuales.
```

---

## Variables consideradas

```text
- Goles recientes a favor.
- Goles recientes en contra.
- Puntos obtenidos en los últimos partidos.
- Ranking Elo.
- Puntaje FIFA.
- Diferencia Elo entre equipos.
- Diferencia FIFA entre equipos.
- Historial directo.
- Ventaja contextual.
- Rendimiento ofensivo reciente.
- Rendimiento defensivo reciente.
- Tarjetas amarillas.
- Tarjetas rojas.
- Fair Play.
```

En este proyecto, la variable FIFA corresponde a **puntaje FIFA**, por lo que un valor más alto representa mejor rendimiento relativo.

---

## Modelos evaluados

```text
1. Regresión Lineal
2. Ridge Regression
3. Random Forest Regressor
4. Gradient Boosting Regressor
5. Poisson Regressor
```

La comparación se realiza con:

```text
- MAE
- RMSE
- R²
- Score compuesto
```

Esto permite justificar la selección del modelo base con métricas y no solo con intuición.

---

## Ponderación balanceada

Para evitar que una sola variable domine la predicción, se aplica una ponderación final:

```text
40% modelo entrenado
25% ranking Elo
15% puntaje FIFA
15% forma reciente
5% contexto, historial y Fair Play
```

Fórmula:

```text
Probabilidad balanceada =
0.40(modelo) + 0.25(Elo) + 0.15(FIFA) + 0.15(forma reciente) + 0.05(contexto)
```

Esta ponderación equilibra el rendimiento reciente, la fuerza contextual del equipo y la lectura estadística del modelo.

---

## Estructura principal del repositorio

```text
.
├── README.md
├── requirements.txt
├── 00_actualizar_datos_oficiales.py
├── 02_reentrenar_modelos_actualizados.py
├── 10_ver_metricas_modelos.py
├── 13_predecir_semifinales.py
├── 14_panel_integral_cuartos.py
├── 15_ver_prediccion_ridge_cuartos.py
├── 16_aplicar_ponderacion_balanceada.py
├── data/
│   ├── results.csv
│   ├── data_modelo.csv
│   ├── fixtures_cuartos.csv
│   ├── fixtures_cuartos_modelo.csv
│   ├── fair_play.csv
│   ├── fifa_ranking.csv
│   ├── elo_ranking.csv
│   └── tarjetas_octavos_reales.csv
└── outputs/
    ├── competencia_modelos.csv
    ├── resumen_metricas_modelos.csv
    ├── mejor_modelo.txt
    ├── prediccion_cuartos_a_semifinales.csv
    ├── prediccion_cuartos_a_semifinales_balanceada.csv
    ├── clasificados_semifinales.csv
    ├── clasificados_semifinales_balanceado.csv
    ├── top10_marcadores_cuartos.csv
    ├── prediccion_tarjetas_cuartos.csv
    ├── prediccion_ridge_cuartos.csv
    ├── panel_competencia_modelos.png
    ├── panel_metricas_modelos.png
    ├── panel_ponderacion_balanceada.png
    ├── panel_integral_cuartos.png
    ├── panel_cuartos_marcadores_tarjetas.png
    └── panel_ridge_cuartos.png
```

---

## Scripts principales

### 00_actualizar_datos_oficiales.py

Actualiza la base con resultados oficiales recientes de octavos y define los cruces reales de cuartos.

### 02_reentrenar_modelos_actualizados.py

Reentrena los cinco modelos con la data general actualizada.

Genera:

```text
outputs/competencia_modelos.csv
outputs/resumen_metricas_modelos.csv
outputs/mejor_modelo.txt
outputs/panel_competencia_modelos.png
outputs/panel_metricas_modelos.png
outputs/modelo_home.pkl
outputs/modelo_away.pkl
```

### 13_predecir_semifinales.py

Predice los partidos de cuartos y estima los clasificados a semifinales.

Genera:

```text
outputs/prediccion_cuartos_a_semifinales.csv
outputs/top10_marcadores_cuartos.csv
outputs/prediccion_tarjetas_cuartos.csv
outputs/clasificados_semifinales.csv
outputs/panel_cuartos_marcadores_tarjetas.png
```

### 16_aplicar_ponderacion_balanceada.py

Aplica la ponderación balanceada entre modelo, Elo, FIFA, forma reciente y contexto.

Genera:

```text
outputs/prediccion_cuartos_a_semifinales_balanceada.csv
outputs/clasificados_semifinales_balanceado.csv
outputs/panel_ponderacion_balanceada.png
```

### 10_ver_metricas_modelos.py

Resume la competencia de modelos y genera el panel comparativo.

### 14_panel_integral_cuartos.py

Genera el panel general de interpretación del proyecto.

Incluye:

```text
- Partido
- Ranking Elo
- Puntaje FIFA
- Forma reciente
- Lambdas finales
- Probabilidades de clasificación
- Marcador probable
- Clasificado estimado
- Tarjetas esperadas
- Riesgo disciplinario
- Mejor modelo
```

Salida principal:

```text
outputs/panel_integral_cuartos.png
```

### 15_ver_prediccion_ridge_cuartos.py

Muestra una lectura complementaria del modelo Ridge. Este modelo permite analizar escenarios más conservadores y partidos cerrados.

---

## Instalación

### 1. Clonar repositorio

```bash
git clone https://github.com/Jeanki07/prediccion-mundial-2026-cuartos-final.git
cd prediccion-mundial-2026-cuartos-final
```

### 2. Crear entorno virtual

Linux o macOS:

```bash
python -m venv venv
source venv/bin/activate
```

Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

---

## Ejecución completa recomendada

```bash
python 00_actualizar_datos_oficiales.py
python 02_reentrenar_modelos_actualizados.py
python 13_predecir_semifinales.py
python 16_aplicar_ponderacion_balanceada.py
python 10_ver_metricas_modelos.py
python 14_panel_integral_cuartos.py
python 15_ver_prediccion_ridge_cuartos.py
```

---

## Ejecución rápida

Si solo se desea revisar las salidas ya generadas:

```bash
python 10_ver_metricas_modelos.py
python 14_panel_integral_cuartos.py
```

---

## Archivos de salida más importantes

### Predicción principal

```text
outputs/prediccion_cuartos_a_semifinales.csv
```

Contiene marcadores estimados, lambdas, probabilidades y clasificado proyectado.

### Predicción balanceada

```text
outputs/prediccion_cuartos_a_semifinales_balanceada.csv
```

Contiene la predicción final ajustada por modelo, Elo, FIFA, forma reciente y contexto.

### Clasificados estimados

```text
outputs/clasificados_semifinales.csv
outputs/clasificados_semifinales_balanceado.csv
```

Contienen los equipos proyectados como semifinalistas.

### Competencia de modelos

```text
outputs/competencia_modelos.csv
outputs/panel_competencia_modelos.png
outputs/panel_metricas_modelos.png
```

Permiten revisar qué modelo tuvo mejor desempeño.

### Panel integral

```text
outputs/panel_integral_cuartos.png
```

Es el panel principal de presentación porque resume la predicción, el contexto y la lectura estadística.

---

## Comandos útiles de revisión

### Ver predicción balanceada

```bash
python -c "import pandas as pd; df=pd.read_csv('outputs/prediccion_cuartos_a_semifinales_balanceada.csv'); print(df.to_string(index=False))"
```

### Ver clasificados

```bash
python -c "import pandas as pd; df=pd.read_csv('outputs/clasificados_semifinales_balanceado.csv'); print(df.to_string(index=False))"
```

### Ver competencia de modelos

```bash
python -c "import pandas as pd; df=pd.read_csv('outputs/competencia_modelos.csv'); print(df.to_string(index=False))"
```

### Abrir panel integral en Linux

```bash
xdg-open outputs/panel_integral_cuartos.png
```

### Abrir panel de ponderación balanceada

```bash
xdg-open outputs/panel_ponderacion_balanceada.png
```

---

## Interpretación académica

Este trabajo no busca asegurar el resultado exacto de un partido, sino construir una estimación fundamentada mediante datos. En fútbol existe incertidumbre, por lo que el resultado debe interpretarse como una probabilidad basada en evidencia.

La fortaleza del proyecto está en combinar:

```text
- Resultados oficiales actualizados.
- Modelos predictivos.
- Métricas comparativas.
- Ranking Elo.
- Puntaje FIFA.
- Forma reciente.
- Distribución de Poisson.
- Análisis de tarjetas.
- Ponderación balanceada.
- Visualizaciones interpretables.
```

---

## Conclusión

El repositorio presenta un sistema completo para analizar los cuartos de final del Mundial 2026 y proyectar semifinalistas. La predicción final no depende únicamente del modelo entrenado, sino de una ponderación balanceada que integra rendimiento estadístico, contexto deportivo y variables recientes.
