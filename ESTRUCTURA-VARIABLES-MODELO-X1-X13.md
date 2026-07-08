# 📊 Modelo Matemático del Proyecto de Predicción

En este proyecto se trabaja con dos variables objetivo numéricas e independientes para modelar el resultado de un encuentro deportivo:

* **y_home** = Goles anotados por el equipo local.
* **y_away** = Goles anotados por el equipo visitante.

Debido a esta naturaleza, cada algoritmo se entrena de forma independiente en dos ocasiones:
1. `modelo_home` &rarr; Predice $y_{\text{home}}$
2. `modelo_away` &rarr; Predice $y_{\text{away}}$

---

## 📋 Variables Predictoras

El vector de entrada para los modelos está compuesto por un conjunto de 13 características socio-deportivas:

$$X = (X_1, X_2, X_3, \dots, X_{13})$$

| Variable matemática | Variable en el código | Descripción |
| :---: | :--- | :--- |
| $X_1$ | `home_gf12` | Goles a favor recientes del equipo local |
| $X_2$ | `home_ga12` | Goles en contra recientes del equipo local |
| $X_3$ | `home_pts12` | Puntos recientes del equipo local |
| $X_4$ | `home_prev_matches` | Partidos recientes del local |
| $X_5$ | `away_gf12` | Goles a favor del visitante |
| $X_6$ | `away_ga12` | Goles en contra del visitante |
| $X_7$ | `away_pts12` | Puntos recientes del visitante |
| $X_8$ | `away_prev_matches` | Partidos recientes del visitante |
| $X_9$ | `diff_fifa` | Diferencia en el ranking FIFA |
| $X_{10}$ | `diff_elo` | Diferencia en el rating Elo |
| $X_{11}$ | `h2h` | Historial de enfrentamientos directos (*Head to Head*) |
| $X_{12}$ | `neutral` | Indicador de campo neutral (Variable binaria) |
| $X_{13}$ | `home_advantage` | Cuantificación del factor de ventaja de localía |

---

## 🧠 Algoritmos Implementados

### 1. Regresión Lineal Múltiple
Modela la relación entre las variables mediante una combinación lineal de los coeficientes $\beta$:

$$\hat{y} = \beta_0 + \beta_1 X_1 + \beta_2 X_2 + \dots + \beta_{13} X_{13}$$

Por lo tanto, el sistema se desglosa en:

$$\hat{y}_{\text{home}} = \beta_{0,\text{home}} + \sum_{i=1}^{13} \beta_{i,\text{home}} X_i$$

$$\hat{y}_{\text{away}} = \beta_{0,\text{away}} + \sum_{i=1}^{13} \beta_{i,\text{away}} X_i$$

> **Interpretación de Coeficientes:**
> * $\beta > 0$: El incremento de la variable aumenta la expectativa de goles.
> * $\beta < 0$: El incremento de la variable reduce la expectativa de goles.
> * $\beta \approx 0$: La variable posee nula o poca influencia en la predicción.

### 2. Regresión Ridge (Regularización $L_2$)
Mantiene la misma estructura lineal pero añade una penalización a la función de pérdida para evitar magnitudes desproporcionadas en los pesos y mitigar el sobreajuste:

$$\min_{\beta} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 + \alpha \sum_{j=1}^{13} \beta_j^2$$

* Reduce drásticamente el sobreajuste (*overfitting*).
* Controla analíticamente los coeficientes grandes.
* Maneja con éxito problemas de multicolinealidad (variables muy correlacionadas).

### 3. Random Forest (Bosques Aleatorios)
Un modelo de ensamble basado en el embolsado (*bagging*) de múltiples árboles de decisión independientes ($T_b$):

$$\hat{y} = \frac{1}{B} \sum_{b=1}^{B} T_b(X)$$

* Calcula las predicciones promediando la salida de $B$ árboles.
* Capaz de capturar interacciones complejas y relaciones no lineales sin transformaciones previas.

### 4. Gradient Boosting (Potenciación del Gradiente)
Técnica de aprendizaje secuencial aditivo donde cada nuevo árbol de decisión débil ($h_m$) se entrena para corregir los residuos (errores) del modelo anterior:

$$F_m(X) = F_{m-1}(X) + \nu h_m(X)$$

Donde los pseudo-residuos calculados en cada iteración $m$ corresponden a:
$$r_{im} = y_i - F_{m-1}(X_i)$$

### 5. Regresor de Poisson (Poisson Regressor)
Dado que los goles son datos de conteo discretos y no negativos ($y \in \{0, 1, 2, \dots\}$), se modelan matemáticamente mediante una distribución de Poisson con parámetro $\lambda$ (media de goles esperados):

$$Y \sim \text{Poisson}(\lambda)$$

Se utiliza una función de enlace logarítmica para garantizar que la tasa $\lambda$ sea siempre positiva:
$$\log(\lambda) = \beta_0 + \beta_1 X_1 + \dots + \beta_{13} X_{13}$$

Aislada la variable, obtenemos la tasa de goles:
$$\lambda = \exp\left(\beta_0 + \sum_{i=1}^{13} \beta_i X_i\right)$$

Definiendo así las tasas individuales:
$$\lambda_{\text{home}} = \exp\left(\beta_{0,\text{home}} + \sum_{i=1}^{13} \beta_{i,\text{home}} X_i\right)$$

$$\lambda_{\text{away}} = \exp\left(\beta_{0,\text{away}} + \sum_{i=1}^{13} \beta_{i,\text{away}} X_i\right)$$

---

## 🎲 Matriz de Probabilidades

Utilizando las tasas estimadas ($\lambda_{\text{home}}, \lambda_{\text{away}}$), se calculan las probabilidades marginales de que ocurra un marcador exacto de goles ($a$ para el local y $b$ para el visitante):

$$P(\text{Home} = a) = \frac{e^{-\lambda_{\text{home}}} \cdot \lambda_{\text{home}}^a}{a!}$$

$$P(\text{Away} = b) = \frac{e^{-\lambda_{\text{away}}} \cdot \lambda_{\text{away}}^b}{b!}$$

Asumiendo la independencia de los eventos, la probabilidad conjunta de un marcador exacto $(a, b)$ es:

$$P(a, b) = P(\text{Home} = a) \times P(\text{Away} = b)$$

A partir de esta matriz de resultados posibles, derivamos los tres escenarios del mercado 1X2:
* **Victoria Local:** $\sum P(a,b) \quad \forall \ a > b$
* **Empate:** $\sum P(a,b) \quad \forall \ a = b$
* **Victoria Visitante:** $\sum P(a,b) \quad \forall \ a < b$

---

## 🔧 Ajustes Avanzados

### Ajuste de Dixon-Coles
La distribución de Poisson clásica tiende a subestimar la ocurrencia de ciertos marcadores de pocos goles en el fútbol real. Se aplica el modelo de Dixon-Coles mediante un factor de corrección $\tau$:

$$P_{\text{ajustada}}(a,b) = P(a,b) \times \tau(a, b, \lambda_{\text{home}}, \lambda_{\text{away}}, \rho)$$

Este ajuste optimiza de forma precisa las probabilidades en los marcadores de baja anotación históricos: $(0-0), (1-0), (0-1), (1-1)$.

### Ajuste por Contexto Competitivo
Permite ponderar la importancia, urgencia o estilo de juego situacional mediante modificadores sobre las expectativas base:

$$\lambda_{\text{home-final}} = \lambda_{\text{home-base}} \times \text{factor-contexto}_{\text{home}}$$

$$\lambda_{\text{away-final}} = \lambda_{\text{away-base}} \times \text{factor-contexto}_{\text{away}}$$

Los factores se tipifican bajo la siguiente escala referencial:
* **1.10**: Planteamiento ofensivo / Necesidad crítica de victoria (+10% ataque).
* **1.00**: Planteamiento neutro / Escenario estándar según datos históricos.
* **0.90**: Planteamiento estrictamente conservador / Defensivo (-10% ataque).

---

## 🔄 Flujo de Ejecución del Modelo

El procesamiento de los datos sigue la siguiente secuencia lógica estructurada:

```mermaid
graph TD
    A[1. Construcción del vector X - 13 variables] --> B[2. Predicción de goles esperados Home/Away]
    B --> C[3. Obtención de parámetros λ Base]
    C --> D[4. Inyección de Ajustes Contextuales]
    D --> E[5. Establecimiento de λ Finales]
    E --> F[6. Construcción de Matriz de Poisson]
    F --> G[7. Aplicación de Ajuste Dixon-Coles]
    G --> H[8. Cálculo final de Probabilidades 1X2]
    H --> I[9. Identificación de Top Marcadores]
    I --> J[10. Simulación de Tabla de Posiciones]
