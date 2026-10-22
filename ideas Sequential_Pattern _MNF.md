# Ideas de implementación: Sequential Pattern Representation + NMF

**Proyecto:** Descubrimiento de patrones de juego y análisis del estilo de equipos y entrenadores mediante eventos de fútbol.  
**Estado:** Propuesta metodológica inicial; sin implementación ni validación empírica.  
**Nota sobre el nombre:** El archivo conserva el nombre solicitado (`MNF`); la técnica correcta se denomina **NMF** (*Non-negative Matrix Factorization*).

## 1. Objetivo

Desarrollar una representación **interpretable y computacionalmente eficiente** de las jugadas para descubrir familias recurrentes de comportamiento, **sin imponer previamente categorías tácticas** como tiki-taka, contraataque o juego directo. Las categorías se interpretarán *después* de descubrir los componentes.

El objetivo final tiene dos niveles:

1. **Perfil del equipo:** qué patrones aparecen y con qué frecuencia, y cómo cambia su uso según el contexto.
2. **Estudio del entrenador:** distinguir la variación asociada al entrenador de la asociada al club, los rivales, la plantilla y las circunstancias de partido, especialmente alrededor de cambios de entrenador.

### Esquema general

```text
DataFrames de jugadas (secuencias variables de eventos)
    → limpieza y normalización de eventos
    → tokenización de acciones, movimientos y zonas
    → extracción de n-gramas secuenciales
    → matriz dispersa TF-IDF
    → NMF: X ≈ W · H
    → pesos / clasificación de patrones por jugada
    → perfiles de equipos y análisis contextual
    → comparación entre entrenadores y clubes
```

## 2. Datos disponibles y consideraciones de origen

**Volumen actual:** aproximadamente **163,909 jugadas**.

La descripción de estructura procede del archivo `contexto-dataframes-jugadas.md`:

- Los eventos originales de StatsBomb están en `datos_statsbomb/competicion_*/temporada_*/eventos/<partido_id>.json`.
- Cada jugada se guarda en `datos_preparados_corpus/competicion_<id>/temporada_<id>/partido_<id>/jugada_0001.json` (etc.). `leer_jugada(ruta)` devuelve su DataFrame; el JSON contiene `columns` y `data`.
- **Cada fila es un evento original**, ordenado por `period` e `index`; el número de filas cambia entre jugadas. Algunas acciones futbolísticas generan varios eventos relacionados.
- Las columnas incluyen `type`, `team`, `possession_team`, `possession`, `location`, `timestamp`, `duration`, `pass`, `carry`, `shot`, `play_pattern`, `related_events`, `under_pressure`, etc.; ciertos valores son diccionarios/listas y pueden faltar.
- El **manifiesto del partido** conserva `play_id`, equipo protagonista, eventos iniciales/finales, posesiones de origen, pases completados e intervenciones rivales absorbidas. Estos metadatos no están agregados a cada DataFrame.
- `frames_360` están almacenados aparte y no forman parte de estas secuencias.

**Limitaciones fundamentales del corpus actual:**

1. Se guardan solo jugadas donde el equipo protagonista completa **al menos tres pases**. Esto puede excluir ataques directos y transiciones cortas: no generalizar los resultados al estilo ofensivo completo sin reconstruir un corpus sin ese filtro.
2. Una «jugada» **no equivale necesariamente a una posesión StatsBomb**: puede atravesar varias posesiones base e incluir intervenciones del rival. En la primera versión el modelo clasificará **episodios tal como fueron segmentados**, no posesiones puras.
3. Se debe distinguir al **equipo protagonista** de eventos del rival, utilizando el manifiesto, **no inferirlo de manera arbitraria** a partir de la mayoría de eventos.

**Decisión inicial:** empezar con los DataFrames de jugadas existentes para un prototipo y documentar sus limitaciones. Mantener como mejora prioritaria reconstruir posesiones sin el filtro de tres pases y comparar ambas segmentaciones.

## 3. Representación de eventos: de DataFrame a tokens

Crear una función conceptual `codificar_jugada(df, metadata) -> secuencias`, que reciba el DataFrame original y el protagonista desde el manifiesto.

### 3.1 Preparación necesaria

- Ordenar por `period`, `index`; tratar correctamente las interrupciones, cambios de periodo y timestamps.
- Extraer el nombre del evento desde el diccionario `type` y las ubicaciones desde `location` y los campos anidados `pass.end_location`, `carry.end_location`, cuando existan.
- Normalizar orientación de ataque utilizando la información real disponible (`attacking_direction`, equipo, periodo); verificar la convención y **no asumir que todos los equipos atacan hacia la misma dirección**.
- Evitar contar como varias decisiones independientes eventos auxiliares de una misma acción (`related_events`, recepciones relacionadas, etc.). Definir qué eventos se conservan y cuáles se usan como contexto.
- Conservar información de recuperación/intervención rival como marca o atributo explícito; definir delimitadores entre posesiones base si se incluyen episodios amplios.
- Distinguir valores faltantes, eventos sin destino y eventos no aplicables. No reemplazar una ausencia de coordenadas por un desplazamiento real igual a cero.

### 3.2 Dos representaciones complementarias

**A. Secuencia de acciones y desplazamientos (prioritaria).**

Cada acción se convierte en un token discreto, por ejemplo:

- `PASS_FORWARD`, `PASS_BACKWARD`, `PASS_LATERAL`, `PASS_SHORT`
- `CARRY_FORWARD`, `CARRY_LATERAL`
- `SHOT`, `RECOVERY`, `INTERCEPTION`
- `OPPONENT_EVENT`, `POSSESSION_BREAK` (si corresponde)

La dirección de los movimientos se obtiene de `dx = x_fin − x_inicio` y `dy = y_fin − y_inicio`, **solo cuando ambas coordenadas existan**. Los umbrales de distancia y dirección son parámetros que deben probarse, no reglas tácticas universales.

**B. Secuencia espacial.**

Discretizar el campo en zonas (p. ej., tercios longitudinales × carriles) y generar tokens de tipo `PASS_ZONE_1_2`, `CARRY_ZONE_2_1`. Conservar tanto la zona como las relaciones de desplazamiento. No usar cuadrículas excesivamente finas al inicio.

**Posible tercera representación futura:** categorías de ritmo/tiempo entre eventos. No incluirlas hasta confirmar calidad de timestamps y utilidad empírica.

**Principio:** los tokens describen hechos observados, **no estilos predefinidos**. La elección de tokenización sí introduce supuestos que habrá que estudiar mediante análisis de sensibilidad.

## 4. Extraer patrones locales mediante n-gramas

Para cada secuencia, extraer subsecuencias **consecutivas**:

- **Unigramas:** tipo de acción o token individual.
- **Bigramas:** dos tokens consecutivos.
- **Trigramas:** tres tokens consecutivos.

Ejemplo:

```text
Jugada: PASS_FORWARD → PASS_FORWARD → CARRY_FORWARD → SHOT
Bigramas:
  PASS_FORWARD PASS_FORWARD
  PASS_FORWARD CARRY_FORWARD
  CARRY_FORWARD SHOT
```

Dos episodios de longitudes diferentes pueden compartir n-gramas sin que sus secuencias completas sean casi idénticas. Se conserva **orden local**, pero no el orden global de toda la jugada; una jugada larga puede mezclar varios patrones.

**Configuración inicial para experimentar (no valores definitivos):**

| Representación | N-gramas | Filtro de frecuencia | Límite inicial |
|---|---|---|---|
| Acción/desplazamiento | 1–3 | `min_df ≈ 20` | ~3,000 variables |
| Acción/zona | 1–2 | `min_df ≈ 20` | ~2,000 variables |

No crear n-gramas que crucen delimitadores de posesión o interrupciones **salvo que se quiera estudiar explícitamente esa transición**. Conviene poder separar el corpus de juego abierto y balón parado; el campo `play_pattern` sirve como ayuda contextual, no como etiqueta táctica obligatoria.

## 5. Vectorización eficiente: TF-IDF y matriz dispersa

Construir una matriz de **jugadas × n-gramas**; cada fila corresponde a una jugada. `TfidfVectorizer` (o `CountVectorizer`) permite conservar un formato disperso `CSR`.

```text
X_accion  = TF-IDF(n-gramas de acciones/desplazamientos)
X_espacio = TF-IDF(n-gramas de acciones/zonas)
X = hstack([X_accion, λ · X_espacio])
```

- El peso espacial `λ` puede comenzar como **0.4**, solo como punto de partida y sujeto a comparación.
- Probar alternativamente **conteos** o **TF normalizada**. TF-IDF es útil para discriminar, pero puede sobreponderar n-gramas raros, que no siempre tienen relevancia táctica.
- Ajustar los vectorizadores en una partición de entrenamiento y aplicar `transform` al resto, evitando fuga de información.
- Conservar por separado los vocabularios y nombres de características de ambas representaciones.
- Evitar convertir `X` a un `numpy.ndarray` denso.

## 6. Descubrimiento de componentes mediante NMF

Factorizar:

\[
X \approx WH,
\quad X \in \mathbb{R}_{\geq 0}^{N\times M},
\quad W \in \mathbb{R}_{\geq 0}^{N\times K},
\quad H \in \mathbb{R}_{\geq 0}^{K\times M}.
\]

- **`H`:** importancia de cada n-grama dentro de cada componente; permite *explicar* los componentes.
- **`W`:** contribución de cada componente a la representación de cada jugada.
- Seleccionar inicialmente `K ∈ {8, 12, 16}` y explorar otros tamaños según los resultados.
- Empezar con `MiniBatchNMF` para limitar costo; una prueba piloto puede ajustar con una muestra **estratificada por partidos/equipos** (p. ej., unas 40,000 jugadas) y después aplicar `transform` al corpus completo. Contrastar con ajuste sobre el total si es viable.
- Fijar semillas y registrar hiperparámetros, versión de scikit-learn, distribución de longitudes y tamaño de `X`.

**Salida por jugada:**

- Pesos completos `W[i, :]` para todos los componentes.
- Etiqueta simple opcional `cluster_i = argmax_k W[i,k]`.
- Pesos relativos opcionales `W[i,k] / sum_j W[i,j]`, siempre que la suma sea positiva.
- Las ponderaciones normalizadas **no son probabilidades calibradas**, y NMF **no es clustering duro**. Una jugada puede tener contribución de varios componentes.
- Si una jugada queda sin representación (`W[i,:]` cercano a cero), marcarla para auditoría, no asignarla sin criterio.

**Nota sobre escalas:** NMF tiene ambigüedad de escala (`W D` y `D⁻¹ H` reconstruyen lo mismo para `D` diagonal positiva). Para comparar contribuciones entre componentes, normalizar las filas de `H` con una convención fija y ajustar `W` en consecuencia, o declarar que la comparación se limita a una parametrización concreta.

## 7. Interpretar las clasificaciones

Para cada componente:

1. Mostrar sus n-gramas de mayor peso en `H`.
2. Recuperar las jugadas con mayor contribución relativa o absoluta en `W` (tras normalización definida).
3. Visualizar sus trayectorias, acciones y zonas sobre el campo.
4. Formular una **interpretación posterior** (p. ej., «progresión mediante pases consecutivos»), sin asumir que el componente representa una táctica conocida.
5. Identificar componentes dominados por artefactos: duración, número de eventos, lugar de inicio, interpolaciones, cortes de posesión, eventos del rival o balón parado.

Si los componentes no resultan interpretables, revisar primero la segmentación y la tokenización antes de aumentar complejidad del modelo.

## 8. Validación y selección de hiperparámetros

No seleccionar el modelo solo por el error de reconstrucción o por la separación matemática de los grupos.

Evaluar:

- **Estabilidad:** entrenamientos con otras semillas, muestras de partidos y temporadas; emparejar componentes por similitud de sus n-gramas antes de compararlos (el orden de los componentes es arbitrario).
- **Interpretabilidad:** revisión de secuencias y trayectorias representativas; si hay video, anotación de una muestra para contrastar la coherencia futbolística.
- **Concentración/colapso:** si casi todas las jugadas quedan dominadas por pocos componentes, o si algunos son duplicados.
- **Robustez a la tokenización:** distinta cuadrícula, umbrales de movimientos, n-gramas 1–2 frente a 1–3, TF-IDF frente a conteos, peso espacial.
- **Validación fuera de muestra:** separar por partido y, si es posible, temporada; ajustar vocabulario y NMF solo con el conjunto de entrenamiento.
- **Baseline:** comparar con características agregadas + K-Means y, opcionalmente, un método secuencial clásico en una muestra pequeña.

La métrica final será una combinación de estabilidad, costo e **interpretabilidad táctica**, no solo una puntuación numérica.

## 9. Escalabilidad y organización de resultados

Con ~163,909 jugadas, priorizar operaciones vectorizadas y matrices dispersas. No construir matrices de distancia completas entre jugadas ni procesar todas las parejas (costo cuadrático).

Propuesta de artefactos intermedios:

```text
outputs/
  metadata_jugadas.parquet          # play_id, partido, equipo, temporada, tiempos, contexto
  secuencias_tokens.parquet         # secuencias de tokens, vinculadas por play_id
  X_ngramas.npz                     # matriz dispersa (CSR)
  vocab_acciones.json
  vocab_espacio.json
  nmf_model.joblib
  clasificaciones.parquet           # play_id, componente dominante, pesos W_1...W_K
  interpretacion_componentes.md    # top n-gramas, jugadas ejemplo, observaciones
```

Recomendaciones de ejecución:

- Leer los JSON por lotes y conservar una **clave estable** por jugada desde el manifiesto; guardar archivos intermedios para no reprocesar los DataFrames en cada experimento.
- Registrar el equipo protagonista, clubes, temporadas y `match_id` **fuera de la representación de acciones**.
- Usar `float32` donde sea posible, limitar vocabulario y medir RAM/tiempo con una muestra antes de escalar.
- Mantener el mismo vectorizador y modelo NMF para comparar periodos y entrenadores; no redefinir componentes por equipo.

## 10. Conexión con el estudio del entrenador

**Primera salida:** distribución de componentes por equipo, partido y periodo. Utilizar la asignación principal o un promedio de pesos relativos como perfil exploratorio, reconociendo que son indicadores descriptivos.

**Segunda salida:** modelar su relación con el contexto:

\[
P(\text{patrón } k \mid \text{equipo},\ \text{entrenador},\ \text{contexto}).
\]

Contexto que conviene adjuntar a cada jugada desde sus metadatos y el partido: marcador **al inicio**, minuto, localía, rival, zona inicial, temporada y posibles estados especiales (expulsión, balón parado, etc.). No usar características que solo se conocen **después** de la jugada para explicar decisiones previas.

Método posterior sugerido: **modelo multinomial jerárquico** para las etiquetas principales (o un modelo apropiado para los pesos si se conservan contribuciones múltiples), con términos de entrenador, club, rival y contexto; estudiar interacciones entrenador × contexto si hay suficientes datos.

Para investigar la influencia del entrenador:

- Comparar un mismo club antes y después de cambios de entrenador, estandarizando por situaciones comparables.
- Buscar entrenadores observados en **más de un club**, cuando existan datos suficientes.
- Considerar cambios de plantilla, rivales, calendario y selección no aleatoria de entrenadores.
- Diferenciar explícitamente **asociación estadística** de **responsabilidad o efecto causal**: el clustering no permite atribuir causalidad por sí solo.
- Analizar tanto **qué contextos genera el equipo** (p. ej., dónde recupera) como **qué patrón utiliza dado ese contexto**, porque controlar el primer fenómeno puede esconder parte del estilo.

## 11. Orden de implementación recomendado

1. **Auditar una muestra de DataFrames y manifiestos:** identificar eventos útiles, protagonista, coordenadas, orientación, intervenciones rivales y segmentación.
2. **Implementar tokenización determinista** sobre los objetos DataFrame; inspeccionar manualmente ejemplos.
3. **Extraer n-gramas y vectorizar** en formato disperso, empezando con una muestra diversa.
4. **Entrenar NMF** con 8, 12 y 16 componentes; guardar `W`, `H` y artefactos reproducibles.
5. **Interpretar y validar** componentes mediante top n-gramas y visualización de jugadas; iterar en tokens y pesos.
6. **Procesar el corpus completo** de 163,909 jugadas con un modelo estabilizado.
7. **Crear perfiles de equipos** y después modelos contextuales para comparar club y entrenador.
8. **Revisar la segmentación del corpus** para eliminar el sesgo de tres pases antes de presentar conclusiones generales sobre estilos ofensivos.

## 12. Decisiones abiertas (no resolver automáticamente)

- ¿Clasificar episodios actuales o reconstruir posesiones puras para el resultado principal?
- ¿Qué eventos múltiples deben fusionarse/excluirse para no duplicar acciones?
- ¿Cómo marcar interrupciones y acciones rivales sin hacer que el modelo agrupe sobre todo por esa circunstancia?
- ¿Qué granularidad espacial, umbrales de movimientos y tamaños de n-grama producen resultados útiles?
- ¿Cuántos componentes son estables e interpretables? ¿Asignación dominante o representación multicomponente?
- ¿Se dispone de entrenadores por partido y suficiente rotación entre clubes para separar efectos?

---

**Principio rector:** descubrir regularidades **interpretables** de las secuencias de eventos, con costo razonable, para construir una taxonomía común que luego permita describir y comparar el estilo observado de los equipos y su relación con los entrenadores. Una clasificación estadística no es, por sí misma, una táctica validada ni una estimación causal de la influencia del entrenador.
