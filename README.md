# Análisis de jugadas de fútbol

Proyecto de preparación y análisis exploratorio de jugadas a partir de eventos de StatsBomb. Convierte los eventos de cada partido en secuencias atribuibles a un **equipo protagonista**, conserva su procedencia y estudia patrones de acciones para comparar el repertorio de los equipos.

> **Estado actual:** el corpus preparado contiene 164,660 jugadas elegibles de 1,808 partidos. Las clasificaciones NMF y SVD + MiniBatchKMeans se conservaron para diagnóstico, pero **no son clases tácticas validadas**. Los nombres futbolísticos del notebook de equipos son interpretaciones aproximadas de sus n-gramas, no etiquetas oficiales ni una taxonomía definitiva.

## Qué permite hacer

- Leer los archivos locales de eventos por competición, temporada y partido, sin exigir datos 360.
- Segmentar eventos cronológicos en jugadas sostenidas y atribuirlas al equipo protagonista.
- Absorber intervenciones rivales breves cuando el protagonista recupera el balón dentro de los límites configurados: hasta 3 segundos y 3 acciones relevantes.
- Guardar solo jugadas con **al menos 3 pases completados** del protagonista, conservando todos los eventos y campos originales de cada una.
- Auditar secuencias descartadas, eventos no exportados, cortes y errores en manifiestos por partido y en un resumen del corpus.
- Separar opcionalmente tramos clasificables de **balón parado** y **juego abierto**. Un corte solo se publica si todos los tramos resultantes cumplen el mínimo de pases.
- Crear rasgos secuenciales de dos y tres acciones, con variantes exploratorias de posición en la jugada, ritmo, zonas del campo e intervenciones rivales.
- Comparar clasificadores NMF y SVD + MiniBatchKMeans por categoría, revisar estabilidad y ejemplos, y registrar la decisión de aceptación o rechazo.
- Explorar el uso de los grupos por equipo, temporada y fase con estadísticas y gráficas en un notebook.

## Flujo de datos

```text
datos_statsbomb/.../eventos/<partido_id>.json
    ↓  Data_Preparation.IPYNB / src.data_preparation
datos_preparados_corpus/.../partido_<id>/
    ├── jugada_0001.json, jugada_0002.json, ...
    └── manifiesto.json
    ↓  src.classification
evaluaciones, modelos y etiquetas exploratorias
    ↓  Analisis_Clasificaciones_Equipos.ipynb
dataframes, estadísticas y gráficas por equipo
```

Una **jugada** puede abarcar varias posesiones base e incluir acciones rivales absorbidas. El JSON de cada jugada tiene `columns` y `data`; conserva los nombres, el orden y los valores originales de las columnas de eventos del partido. `play_id`, equipo protagonista, pases completados, categoría, posesiones de origen y eventos de intervención se guardan en el `manifiesto.json`, sin añadirse a las filas de eventos. `resumen_corpus.json` reúne los totales y los errores de procesamiento. La definición formal está en [CONTEXT.md](CONTEXT.md) y el esquema se explica en [contexto de los DataFrames](docs/implementation/contexto-dataframes-jugadas.md).

La categoría `balon_parado` comprende `From Corner`, `From Free Kick` y `From Throw In`. `juego_abierto` comprende `Regular Play`, `From Counter`, `From Goal Kick`, `From Kick Off` y `From Keeper`. Son **fases para clasificar**; no equivalen a tipos tácticos de jugada. Los valores desconocidos se auditan y reciben una categoría mediante la regla documentada en [fases de clasificación](docs/implementation/fases-clasificacion.md).

## Instalación y datos necesarios

Requiere Python y los paquetes de [requirements.txt](requirements.txt). Desde la raíz del repositorio:

```bash
python -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m ipykernel install --user --name hackaton-america --display-name "Hackaton America"
```

En Windows, use el ejecutable equivalente de `.venv/Scripts/`. Para el análisis de equipos se necesita `matplotlib`, incluido en `requirements.txt`.

Los eventos deben estar en `datos_statsbomb/competicion_*/temporada_*/eventos/<partido_id>.json`. Los datos brutos, corpus y artefactos de modelos se excluyen de Git; para ejecutar los notebooks es necesario tenerlos localmente o generarlos con el flujo del proyecto. `EDA.IPYNB` contiene la exploración y descarga iniciales. La preparación y el análisis posteriores leen archivos locales y no dependen del estado interactivo de ese notebook.

## Uso

### 1. Preparar las jugadas

Abra [Data_Preparation.IPYNB](Data_Preparation.IPYNB) desde la raíz del proyecto y ajuste su celda de configuración: `RUTA_ENTRADA`, `RUTA_SALIDA`, mínimo de pases, límites de intervención, `DIVIDIR_FASES_CLASIFICABLES`, `SOBRESCRIBIR` y número de trabajadores. Ejecute las celdas en orden. La preparación también está disponible como función:

```python
from src.data_preparation.corpus import procesar_corpus

resumen = procesar_corpus(
    "datos_statsbomb",
    "datos_preparados_corpus",
    trabajadores=6,
    dividir_fases_clasificables=True,
)
print(resumen["estado"], resumen["total_jugadas_elegibles"])
```

Si ya existe un corpus preparado sin el corte de fases, puede actualizarse **en su ubicación actual** sin volver a leer los eventos originales:

```bash
.venv/bin/python -m src.data_preparation.prepared_corpus datos_preparados_corpus --trabajadores 6
```

La ejecución documentada del corpus vigente procesó 1,808 partidos, no registró fallos y terminó con 164,660 jugadas elegibles, de las cuales 60,852 son de balón parado y 103,808 de juego abierto. Son cifras de esta corrida local, no valores fijos del programa.

### 2. Evaluar clasificaciones exploratorias

El piloto alternativo compara SVD + MiniBatchKMeans con una referencia NMF sobre 30,000 jugadas de entrenamiento y 6,000 de evaluación, separadas por partido:

```bash
.venv/bin/python -m src.classification.svd_kmeans \
  datos_preparados_corpus datos_clasificacion_svd_kmeans_piloto
```

La evaluación y los criterios de selección están en [clustering SVD + MiniBatchKMeans](docs/implementation/clustering-svd-minibatchkmeans.md). El piloto actual fue rechazado: los mejores índices ARI entre semillas fueron 0.408 para balón parado y 0.485 para juego abierto, por debajo del umbral previsto de 0.75. Para reproducir las etiquetas exploratorias del corpus completo desde ese piloto rechazado:

```bash
.venv/bin/python -m src.classification.svd_kmeans \
  datos_preparados_corpus datos_clasificacion_svd_kmeans_completo_exploratorio \
  --piloto datos_clasificacion_svd_kmeans_piloto/evaluacion.json \
  --revision-piloto datos_clasificacion_svd_kmeans_piloto/revision_piloto.json \
  --exploratorio-rechazado
```

Use un **directorio de salida nuevo** al repetir experimentos. La salida completa contiene `evaluacion_completa.json`, dos modelos `joblib` y `clasificacion_exploratoria.jsonl` con una fila por jugada: `play_id`, categoría, cluster cercano, etiqueta exploratoria o `Sin patrón claro`, distancias, umbral y ruta de su manifiesto. Su `decision` sigue siendo `rechazado`. La publicación NMF en `datos_publicacion_clasificacion/` guarda diagnóstico y modelos, **sin tabla de clases por jugada**, porque sus componentes también fueron rechazados. Véase [publicación de clasificación](docs/implementation/publicacion-clasificacion.md).

### 3. Analizar equipos

Abra [Analisis_Clasificaciones_Equipos.ipynb](Analisis_Clasificaciones_Equipos.ipynb) desde la raíz, después de generar `datos_preparados_corpus/` y `datos_clasificacion_svd_kmeans_completo_exploratorio/`. El cuaderno:

1. Revisa las 24 fichas y sus n-gramas, propone una descripción futbolística aproximada y permite abrir ejemplos originales.
2. Une las 164,660 etiquetas exploratorias con el equipo protagonista y la temporada, comprobando IDs y categorías.
3. Crea `df_jugadas`, `df_catalogo`, `df_estadisticas_jugadas` y `df_estadisticas_familias`, además de tablas de uso por equipo.
4. Muestra frecuencias de **familias sugeridas** y **jugadas aproximadas**, pases y eventos medios, matriz de familias por equipo, ranking de una jugada elegida por su nombre, calidad de asignación y evolución por temporada.

`df_estadisticas_equipos` resume cada jugada aproximada por equipo y fase; `df_estadisticas_familias_equipos` agrupa esas lecturas por familia. Incluyen conteos, pases y eventos medios, porcentajes dentro de la fase y un índice de uso relativo respecto al corpus. `Sin patrón claro` permanece como fila propia. La última celda permite exportar los dataframes a CSV si se activa `EXPORTAR_CSV`.

## Estructura del repositorio

| Ruta | Función |
| --- | --- |
| `src/data_preparation/` | Lectura de eventos, segmentación, continuidad, fases, exportación y actualización del corpus. |
| `src/classification/features.py` | Representación de acciones y variantes espaciales, posicionales y temporales. |
| `src/classification/exploratory_nmf.py` | Pilotos NMF y comparaciones de rasgos. |
| `src/classification/selection.py` | Evaluación y decisión sobre componentes NMF. |
| `src/classification/publication.py` | Publicación reproducible o diagnóstico según la decisión. |
| `src/classification/svd_kmeans.py` | Piloto, reajuste completo y publicación condicionada a revisión. |
| `src/classification/rival_ablation.py` | Comparación de rasgos con y sin intervenciones rivales. |
| `tests/` | Pruebas de preparación, clasificación, exportación y casos integrados. |
| `docs/specs/` | Acuerdos funcionales y criterios de aceptación. |
| `docs/implementation/` | Detalles de implementación y resultados de las evaluaciones. |

## Comprobación

```bash
.venv/bin/python -m pytest -q
```

Las pruebas cubren integridad de eventos y jugadas, reglas de continuidad, cortes por fase, representación, reproducibilidad y decisiones de publicación. La interpretación futbolística de un grupo exige además revisar sus ejemplos y estabilidad; aprobar una prueba automática no valida por sí solo una clase táctica.

## Alcance y límites

- El análisis principal usa eventos; los datos 360 son un complemento espacial previsto, pero su ausencia no excluye jugadas. El proyecto no reconstruye trayectorias continuas de los 22 jugadores.
- Las agrupaciones actuales se forman por similitud de acciones. Varios grupos recogen longitud o dirección del pase y presión, y mezclan tipos de reanudación. Sus IDs no implican un esquema táctico oficial.
- Las cifras de uso por equipo describen **frecuencia**, no eficacia ni calidad. El número de partidos disponibles y la temporada parcial 2026/2027 afectan las comparaciones; conviene usar porcentajes dentro de fase y revisar tamaños de muestra.
- La clase `Sin patrón claro` identifica jugadas que superan el umbral de distancia de su cluster cercano; el umbral es un criterio exploratorio, no una probabilidad calibrada.

Para el diseño de segmentación consulte [clustering de jugadas](docs/specs/clustering-de-jugadas.md); para el diseño de clasificación secuencial consulte [clasificación secuencial y NMF](docs/specs/clasificacion-secuencial-nmf.md). Las especificaciones describen objetivos y acuerdos; los documentos de implementación indican qué se ejecutó y qué fue aceptado o rechazado.
