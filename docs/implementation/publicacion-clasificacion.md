# Publicación reproducible de clasificación (#19)

La [evaluación del ticket #18](seleccion-componentes-nmf.md) rechazó los
componentes NMF como clasificación táctica válida. Por ello, la publicación
real de este ticket tiene estado `diagnostico_publicado`: **no crea una tabla
de etiquetas por jugada**. Conserva los modelos del piloto, sus vocabularios,
parámetros, semillas, umbrales, fichas de componentes, diagnósticos y decisión
para permitir auditar el intento. Un método distinto requiere otra
especificación.

## Ejecución real

Desde la raíz del repositorio:

```bash
.venv/bin/python -m src.classification.publication \
  datos_preparados_corpus \
  datos_seleccion_componentes_piloto/evaluacion/evaluacion.json \
  datos_publicacion_clasificacion
```

La corrida del 22 de octubre de 2026 recorrió los manifiestos de **1,808
partidos** y concilió **164,660 jugadas vigentes**: 60,852 de balón parado y
103,808 de juego abierto. El resumen del corpus declaró los mismos partidos,
jugadas y eventos exportados. Tardó 82.17 s y alcanzó 197,540 KiB de memoria
residente máxima en este entorno.

`datos_publicacion_clasificacion/` contiene `informe.json` y
`modelo_balon_parado.joblib` y `modelo_juego_abierto.joblib`. No contiene
`clasificacion.jsonl`; las 1,025 etiquetas exploratorias del piloto tampoco se
copian al informe. Este registra `decision=rechazado`, la justificación,
diagnósticos de las once corridas, comparaciones, umbrales y fichas. Guarda
SHA-256 de la evaluación, resultados del candidato, modelos, resumen del corpus
y conjunto ordenado de manifiestos. La salida local de datos se excluye de Git.
El comando exige un directorio de destino nuevo para no sobrescribir una
publicación anterior.

## Ruta para una evaluación aceptada

El mismo comando publica `clasificacion.jsonl` **solo** cuando la evaluación
indica `decision=aceptado` y `estado=clasificacion_validada`. Aplica, sin
reajustar, el vectorizador, escala y NMF guardados de cada categoría. Por cada
partido comprueba archivos, eventos, IDs y totales del manifiesto antes de
escribir una fila por jugada. La fila enlaza `play_id`, origen, partido,
temporada, categoría/modelo, clase principal o «Sin patrón claro», todos los
pesos, contribuciones normalizadas, error, cobertura, umbral y archivo de
eventos. La salida aparece solo después de conciliar todas las jugadas; una
falla no deja una publicación parcial.

La clase se decide con el umbral aprendido en entrenamiento. Las
contribuciones se comparan como `W_k × ||H_k||₂` y se dividen entre su suma;
no son probabilidades. Una jugada sin rasgos conocidos o por encima del umbral
recibe «Sin patrón claro» y conserva sus pesos. Los modelos y sus vocabularios
se copian a la publicación junto con la configuración exacta, por lo que
pueden usarse sin el directorio del piloto. Para aplicar **un paquete aceptado**
a otro corpus actualizado:

```bash
.venv/bin/python -m src.classification.publication \
  RUTA_CORPUS_NUEVO RUTA_PAQUETE_ACEPTADO RUTA_SALIDA_NUEVA \
  --usar-modelos-publicados
```

El modo de aplicación verifica las huellas de ambos modelos y rechaza un
paquete con decisión `rechazado`. La prueba integrada usa un corpus pequeño
con corte válido, tramo corto conservado, intervención rival absorbida,
reflexión de banda y timestamp ausente. Comprueba IDs y eventos de salida,
repite la publicación con bytes idénticos y aplica los modelos copiados tras
retirar los archivos originales del piloto.
