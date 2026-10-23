# Clustering alternativo: TruncatedSVD + MiniBatchKMeans

Este flujo consume el corpus preparado con las fases `balon_parado` y
`juego_abierto`. No cambia la segmentación ni `representar_jugada`: usa los
mismos n-gramas base de 2–3 acciones, `DictVectorizer` disperso y
`TfidfTransformer(norm="l2", use_idf=True)` de la referencia NMF.

## Piloto

```bash
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  .venv/bin/python -m src.classification.svd_kmeans \
  datos_preparados_corpus datos_clasificacion_svd_kmeans_piloto
```

La muestra tiene exactamente 30,000 jugadas de entrenamiento y 6,000 de
evaluación. Sus cuotas son proporcionales a las jugadas de cada temporada; los
partidos se eligen con semilla 42 y no se comparten entre particiones. Solo el
último partido de cada cuota puede recortarse. Se ajustan vectorizador y TF-IDF
con entrenamiento, y se proyectan los partidos reservados sin reajustarlos.

Por cada categoría se prueban 32 y 64 componentes de TruncatedSVD, normalización
L2 y MiniBatchKMeans con K = 6, 8, 12 y 16. Hay tres semillas de KMeans
(42, 43, 44). La comparación NMF se reentrena en exactamente la misma muestra,
partición y matriz TF-IDF, con cuatro componentes como la referencia anterior.
Ambos métodos informan silueta coseno, ARI entre semillas, fracciones por grupo,
distancias, sesgo de longitud, n-gramas y tres ejemplos de partidos distintos.
La asignación NMF utiliza `W_k × ||H_k||₂`, como la publicación NMF previa,
antes de normalizar sus contribuciones para las métricas.
Las siluetas usan como máximo 2,000 jugadas; las distancias a centroides son
matrices jugadas × K. Nunca se calculan distancias entre todas las jugadas.

Un candidato numérico exige silueta ≥ 0.03, ARI ≥ 0.75 y que cada cluster
contenga del 1 % al 40 % de su categoría en evaluación. Se elige el mejor por
silueta, estabilidad y menor complejidad. El archivo `evaluacion.json` registra
todos los intentos, incluso los rechazados. Un candidato numérico **no** es
una clasificación aceptada. Se deben inspeccionar los n-gramas y al menos tres
ejemplos por cluster para comprobar que el desarrollo de juego sea coherente y
distinguible, y que no predominen longitud, tipo de reanudación o presión. Un
cluster dudoso rechaza su categoría; ambas categorías deben aprobar.

## Reajuste completo y publicación

Si ambas categorías pasan, registrar la revisión del piloto en un JSON:

```json
{
  "aprobado": true,
  "justificacion": "Motivos distinguibles en ambas categorías",
  "categorias": {
    "balon_parado": {"aprobado": true, "justificacion": "Se revisaron todas las fichas"},
    "juego_abierto": {"aprobado": true, "justificacion": "Se revisaron todas las fichas"}
  }
}
```

Los textos deben describir la evidencia observada; los del ejemplo son solo el
esquema. El reajuste se ejecuta con:

```bash
.venv/bin/python -m src.classification.svd_kmeans \
  datos_preparados_corpus datos_clasificacion_svd_kmeans_completo \
  --piloto datos_clasificacion_svd_kmeans_piloto/evaluacion.json \
  --revision-piloto revision_piloto.json
```

El reajuste usa todas las jugadas vigentes del corpus; actualmente son 164,660,
frente a las 163,909 anteriores al corte por fase. Genera modelos, fichas y
`clasificacion_exploratoria.jsonl` con una fila por jugada. El cluster más
cercano queda registrado aunque la etiqueta sea «Sin patrón claro». Esta se
asigna cuando la distancia supera el percentil 95 de las distancias de su
cluster en el ajuste; también se conservan la distancia y el margen al segundo
centroide. Antes de publicar se vuelven a comprobar tamaños, silueta acotada,
n-gramas y tres ejemplos por cluster.

Tras una segunda revisión motivada del modelo completo, se publica:

```bash
.venv/bin/python -m src.classification.svd_kmeans \
  datos_preparados_corpus datos_clasificacion_svd_kmeans_publicado \
  --publicar datos_clasificacion_svd_kmeans_completo \
  --revision-completa revision_completa.json
```

La revisión completa usa el mismo esquema JSON. Si falla algún filtro o no se
aprueban todas las fichas, se conservan solo los diagnósticos exploratorios.
Los directorios de salida deben ser nuevos para no sobrescribir corridas.

## Resultado del piloto actual

El piloto se ejecutó sobre el corpus vigente de **164,660 jugadas**. Ajustó con
30,000 jugadas de 332 partidos y evaluó con 6,000 jugadas de otros 69 partidos;
las seis temporadas aparecen en ambas particiones. Los tamaños por categoría
son 11,103/2,197 jugadas de balón parado y 18,897/3,803 de juego abierto
(entrenamiento/evaluación).

| Método y categoría | Mejor silueta coseno | ARI de esa corrida | Tamaño mínimo–máximo | Resultado |
|---|---:|---:|---:|---|
| SVD 32, K=12, balón parado | 0.130 | 0.352 | 5.6 %–13.1 % | Rechazado |
| SVD 32, K=16, juego abierto | 0.133 | 0.485 | 4.3 %–9.4 % | Rechazado |
| NMF 4, balón parado | 0.559 | 1.000 | 17.4 %–34.0 % | Referencia no aceptada |
| NMF 4, juego abierto | 0.555 | 1.000 | 17.0 %–37.2 % | Referencia no aceptada |

Ninguna de las 16 combinaciones SVD/KMeans alcanza ARI 0.75. Entre las ocho
combinaciones de balón parado, el ARI máximo fue 0.408; en juego abierto, 0.485.
Las siluetas de NMF y SVD se miden en sus respectivos espacios reducidos, por
lo que no son una comparación directa de calidad táctica. NMF usa cuatro
componentes con inicialización casi determinista y sus buenos indicadores
numéricos, que ahora pasan los tres filtros en ambas categorías, tampoco validan
la interpretación. La evaluación NMF previa ya
rechazó estos motivos; al reentrenarlo sobre la muestra común, las fichas
vuelven a estar dominadas por alternancias de pase/conducción y presión.

En SVD/KMeans, los n-gramas principales de varias fichas de balón parado
separan presión, longitud/dirección del pase o recuperación tras despeje; en
otras se repite la misma alternancia de pase y conducción. Los ejemplos
inspeccionados incluyen saques de banda, faltas y córners mezclados dentro de
un mismo cluster. En juego abierto aparecen grupos de presión y varios grupos
de circulación con pequeñas variaciones de dirección; los grupos de
circulación más largos contienen ejemplos de 43 a 127 eventos. Esto no satisface
el criterio estricto de interpretación por **cada** cluster. El diagnóstico
completo, las fichas y los enlaces `play_id` están en
`datos_clasificacion_svd_kmeans_piloto/evaluacion.json`.

**Decisión del piloto:** conservarlo como diagnóstico exploratorio. El archivo
`revision_piloto.json` en la salida registra el rechazo. No publicar estas
etiquetas como clases tácticas validadas.

## Etiquetas exploratorias para el corpus completo

A petición posterior del usuario, el flujo permite ajustar el corpus completo
incluso cuando el piloto fue rechazado. Entre las variantes con silueta y
tamaños mínimos, elige la mayor estabilidad ARI; esto selecciona `SVD32/K8`
para balón parado (ARI 0.408) y `SVD32/K16` para juego abierto (ARI 0.485).
Ninguna supera el umbral 0.75, por lo que la salida se denomina
`clasificacion_exploratoria.jsonl` y conserva `decision="rechazado"`.

```bash
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  .venv/bin/python -m src.classification.svd_kmeans \
  datos_preparados_corpus datos_clasificacion_svd_kmeans_completo_exploratorio \
  --piloto datos_clasificacion_svd_kmeans_piloto/evaluacion.json \
  --revision-piloto datos_clasificacion_svd_kmeans_piloto/revision_piloto.json \
  --exploratorio-rechazado
```

La tabla incluye una fila por jugada y asigna `etiqueta_exploratoria` a un
cluster o a «Sin patrón claro». Conserva el cluster más cercano, la distancia,
el margen al segundo centroide, el umbral y la ruta del archivo de jugada. El
reajuste vuelve a calcular n-gramas y tres ejemplos por cluster. La función de
publicación validada rechaza expresamente un paquete generado desde un piloto
rechazado.

La corrida completa terminó con **164,660 filas únicas**: 60,852 de balón
parado y 103,808 de juego abierto. Hay 24 clusters con jugadas y 8,243 filas
marcadas «Sin patrón claro» (5.0 %). Las siluetas coseno sobre 2,000 jugadas
fueron 0.098 y 0.131, respectivamente. Los tamaños de cluster del ajuste
completo quedaron entre 6.3 % y 18.6 % en balón parado, y entre 3.8 % y 11.5 %
en juego abierto. El archivo de 76 MB se encuentra en
`datos_clasificacion_svd_kmeans_completo_exploratorio/clasificacion_exploratoria.jsonl`;
`evaluacion_completa.json` y dos modelos `joblib` están en la misma carpeta.
Estos datos no anulan el rechazo del piloto: la estabilidad entre semillas
continúa por debajo de 0.75 y las etiquetas siguen siendo exploratorias.
