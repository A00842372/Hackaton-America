# Selección de componentes NMF y «Sin patrón claro» (#18)

## Decisión

**Se rechazan los componentes de este piloto como clasificación táctica válida.**
Las dos categorías conservan modelos y pesos para inspección, pero
`clase_principal` queda en `null` en la salida de evaluación. La columna
`etiqueta_exploratoria` permite auditar el componente dominante y el umbral;
no debe usarse como clase publicada ni para perfiles de equipo. El siguiente
método de clasificación requiere otra especificación, según
`docs/specs/clasificacion-secuencial-nmf.md`.

El motivo no es solo el error L2: los componentes de balón parado y juego
abierto se parecen mucho entre sí, principalmente alternancias de pase y
conducción; el modelo no recupera de forma convincente las jugadas de córner
con remate ni separa las estructuras de reanudación. Además, los errores
dependen de la longitud y la reserva de una temporada completa reduce la
cobertura. Aumentar los componentes mejora algo la reconstrucción, pero divide
motivos parecidos y deja componentes dominados por `Pressure`.

## Muestra y comparaciones

Se tomaron 12 partidos preparados de seis temporadas, con 14 equipos
protagonistas y **1,025 jugadas**. La partición por partido reserva 3799351,
3939817 y 4004041: quedan 292 jugadas de balón parado y 516 de juego abierto
para ajustar; 82 y 135, respectivamente, para evaluar. La comparación por
temporada reserva los dos partidos de la temporada 351; se ajusta con las otras
cinco temporadas y se evalúan 53 y 110 jugadas. Los vocabularios, escalas y NMF
se ajustan solo en entrenamiento. Se compara cada categoría por separado.

| Variante | Balón parado: cobertura / error L2 | Juego abierto: cobertura / error L2 | Observación |
|---|---:|---:|---|
| Base, TF-IDF L2, 4 componentes | 0.895 / 0.932 | 0.921 / 0.942 | Referencia |
| Frecuencia L2, 4 | 0.895 / 0.830 | 0.921 / 0.835 | Motivos similares; cambia el espacio de rasgos |
| Posición de n-gramas, 4 | 0.861 / 0.951 | 0.897 / 0.960 | Menos cobertura |
| Ritmo observado, 4 | 0.859 / 0.944 | 0.889 / 0.956 | Menos cobertura |
| Espacio con peso 0.5, 4 | 0.674 / 0.945 | 0.712 / 0.955 | Rasgos espaciales y ausencias ganan peso |
| Espacio con peso 1.5, 4 | 0.498 / 0.979 | 0.545 / 0.983 | Rasgos espaciales dominan |
| Sin tokens rivales, 4 | 0.905 / 0.904 | 0.928 / 0.923 | Se pierden interrupciones observadas |
| Base, 2 componentes | 0.895 / 0.953 | 0.921 / 0.954 | Motivos demasiado generales |
| Base, 8 componentes | 0.895 / 0.907 | 0.921 / 0.927 | Divide motivos parecidos |
| Base, semilla 43 | 0.895 / 0.932 | 0.921 / 0.942 | Mismo ajuste que semilla 42 |
| Base, reserva temporada 351 | 0.821 / 0.942 | 0.884 / 0.947 | Menor cobertura fuera de temporada |

La cobertura es la fracción media de n-gramas observados reconocidos por el
vocabulario de entrenamiento; el error es el residuo L2 medio en evaluación.
Todos los vectores se normalizan a norma L2. **Los errores entre espacios de
rasgos distintos no miden por sí solos utilidad táctica comparable**; por eso
la comparación también inspecciona composición y jugadas concretas. La matriz
base dispersa de entrenamiento es 292 × 1,782 (7,530 valores no nulos) y
516 × 2,626 (15,072). La variante espacial llega a 9,056 y 16,591 columnas;
las variantes posicional y temporal a 3,998/6,151 y 4,202/6,359.

## Estabilidad, sesgos y ejemplos

Los componentes se emparejaron por coseno de `H` sobre n-gramas compartidos,
sin depender de su número de etiqueta. Las semillas 42 y 43 dan coseno y
acuerdo de dominante de 1.00. Con posición el coseno medio es 0.998/0.998
(balón parado/juego abierto), y con ritmo 0.992/0.948; el acuerdo dominante
reservado es 0.951/0.933 y 0.890/0.807. La frecuencia cambia más las
asignaciones (0.707/0.770). Quitar rivales da 0.780/0.667. Al reservar la
temporada 351 el coseno sigue en 0.970/0.977, pero la distribución de
jugadas de evaluación cambia, de modo que no se calcula acuerdo reservado
directo. Esta estabilidad del vocabulario compartido no demuestra calidad
táctica.

En la referencia, el componente dominante más frecuente ocupa 26 de 82
jugadas reservadas de balón parado (31.7 %) y 47 de 135 de juego abierto
(34.8 %); no hay colapso en una sola etiqueta. Sin embargo, la correlación de
Spearman entre número de eventos y error es **−0.555/−0.470**: las jugadas
cortas tienden a encajar peor. Con 8 componentes aún es −0.429/−0.414. Las
cuatro fichas de balón parado cubren córners, tiros libres y saques de banda
sin una separación clara por reanudación. Dos fichas tienen 50 % y 52 % de
masa de `H` en n-gramas con intervención rival; una de juego abierto, 57 %.
Solo 1 de 82 y 5 de 135 jugadas reservadas contienen una intervención rival
absorbida, así que este contraste tiene poca potencia.

Ejemplos enlazados por `play_id` en `evaluacion.json`:

- `c73_t235_m3835101_p0026` representa `juego_abierto_c01`: salida desde
  saque de puerta con pases y conducciones repetidos, error 0.558. Explica el
  motivo predominante, pero no una estructura táctica distintiva.
- `c73_t351_m4063524_p0067` representa `juego_abierto_c02`: pases,
  conducciones y tres registros `Pressure`, error 0.836; el componente carga
  57 % de su masa en rasgos rivales.
- `c73_t318_m4004041_p0057` es un córner con tres pases y remate, ocho
  eventos, error 0.9996: queda entre los peores encajes de balón parado.
- `c73_t318_m4004041_p0031` es una salida desde saque de puerta con pases y
  presión, doce eventos, error 0.9990, entre los peores de juego abierto.

## Umbral experimental y escala

El umbral de mal ajuste se calcula por **categoría y solo con entrenamiento**.
Con el cuantil 0.90 de la referencia es 0.993924 para balón parado y
0.992726 para juego abierto: «Sin patrón claro» se asignaría a 6/82 y 14/135
jugadas reservadas. Con 0.95 serían 2/82 y 3/135; con 0.99, ninguna. Este
umbral alto deja pasar muchas jugadas con error cercano a 1; por eso la
categoría no rescata el modelo rechazado. El umbral se registra para auditoría,
no como calibración de confianza.

Los `pesos` guardan la `W` original. Para comparar contribuciones dentro de
una jugada se usa `W_k × ||H_k||₂`, dividido entre la suma de esas magnitudes.
Esto elimina la arbitrariedad de escala entre `W` y `H`; las contribuciones
normalizadas **no son probabilidades**. Incluso una jugada exploratoriamente
marcada «Sin patrón claro» conserva todos los pesos y el error. En una corrida
rechazada, `clase_principal` es `null` para todas las jugadas.

## Reproducción

Desde la raíz del repositorio, con el corpus `datos_preparados_corpus` y las
dependencias de `.venv`:

```bash
.venv/bin/python - <<'PY'
from pathlib import Path

from src.classification.exploratory_nmf import clasificar_muestra
from src.classification.features import ConfiguracionEspacial
from src.classification.selection import evaluar_corridas

salida = Path("datos_seleccion_componentes_piloto")
reservados = [3799351, 3939817, 4004041]
variantes = {
    "base_4": {},
    "frecuencia_4": {"ponderacion": "frecuencia_l2"},
    "posicional_4": {"posicion": True},
    "temporal_4": {"posicion": True, "ritmo": True},
    "espacial_05_4": {"configuracion_espacial": ConfiguracionEspacial(6, 4, 0.5)},
    "espacial_15_4": {"configuracion_espacial": ConfiguracionEspacial(6, 4, 1.5)},
    "sin_rivales_4": {"modo_rivales": "sin_rivales"},
    "base_2": {"n_componentes": 2},
    "base_8": {"n_componentes": 8},
    "semilla_43": {"semilla": 43},
    "reserva_temporada_351": {"partidos_reservados": [4063523, 4063524]},
}
corridas = {}
for nombre, cambio in variantes.items():
    opciones = dict(max_partidos=12, muestra_diversa=True, n_componentes=4,
                    semilla=42, partidos_reservados=reservados)
    opciones.update(cambio)
    ruta = salida / nombre
    clasificar_muestra("datos_preparados_corpus", ruta, **opciones)
    corridas[nombre] = ruta
evaluar_corridas(
    corridas, salida / "evaluacion", candidato="base_4", cuantil_error=0.90,
    decision="rechazado",
    justificacion="Motivos poco distintivos, ajuste débil y sesgo de longitud; reserva de temporada empeora cobertura",
)
PY
```

El comando crea directorios nuevos y falla si existen. `evaluacion.json`
contiene 1,025 filas, diagnósticos por categoría, temporada, reanudación e
intervenciones rivales, fichas con n-gramas y ejemplos, y comparaciones de
componentes. Las rutas del manifiesto y archivo permiten volver a la jugada
preparada. Los artefactos grandes quedan fuera de Git mediante `.gitignore`.
