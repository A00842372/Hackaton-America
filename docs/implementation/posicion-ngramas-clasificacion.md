# Posición de los n-gramas en la jugada

La comparación exploratoria añade a los n-gramas locales de dos y tres acciones
otros rasgos que indican si ocurrieron al `inicio`, en el `medio` o al `final`.
Los tercios se calculan por número de acciones **después** de agrupar lances
vinculados y, cuando procede, retirar las acciones rivales. Cada acción pertenece
a un solo tercio; los n-gramas con posición se forman dentro de ese tercio. Un
tercio de una sola acción no aporta n-gramas posicionales, pero la jugada conserva
sus n-gramas locales originales. Los límites de una jugada preparada nunca se
cruzan. La variante espacial, si se usa, añade los mismos prefijos de posición
a sus rasgos espaciales; los eventos del corpus no cambian.

Para repetir el piloto de doce partidos:

```bash
.venv/bin/python -m src.classification.exploratory_nmf \
  datos_preparados_corpus datos_clasificacion_posicional \
  --comparar-posicion --max-partidos 12 --componentes 4 --semilla 42
```

La corrida del 22 de octubre de 2026 cubrió 1,028 jugadas, nueve partidos de
entrenamiento y tres reservados, exactamente iguales en ambas variantes. El
comando tardó 56.72 s y alcanzó 355,600 KiB de memoria residente máxima en
este entorno. El vocabulario de entrenamiento pasó de 1,644 a 3,811 rasgos en
balón parado y de 2,521 a 5,908 en juego abierto. Por tanto, las matrices
dispersas de entrenamiento tienen respectivamente 288 × 1,644 y 288 × 3,811
columnas en balón parado, y 502 × 2,521 y 502 × 5,908 en juego abierto.

| Categoría | Variante | Cobertura media reservada | Error L2 medio reservado |
|---|---|---:|---:|
| Balón parado | Base | 0.8974 | 0.9327 |
| Balón parado | Posicional | 0.8651 | 0.9518 |
| Juego abierto | Base | 0.9275 | 0.9287 |
| Juego abierto | Posicional | 0.9038 | 0.9473 |

El error describe el ajuste en el espacio TF-IDF de cada variante. Los valores
cambian al añadir posición, pero los espacios de rasgos son distintos y sus
errores no prueban por sí solos una mejora o empeoramiento táctico. Este primer
piloto estableció el costo de la variante cronológica antes de medir el ritmo.
Ninguna etiqueta de esta corrida es una clase táctica validada.
`comparacion.json` enlaza los dos resultados; cada carpeta guarda
modelos, vocabularios, pesos y procedencia por `play_id`.

## Variante temporal y muestra diversa

El costo del primer piloto permitió probar una señal de ritmo. La variante
temporal conserva los n-gramas base y añade n-gramas de 2 y 3 acciones con la
diferencia **observada entre timestamps de inicio** de acciones consecutivas:
`rapido` hasta 1 s, `medio` más de 1 y hasta 3 s, y `lento` más de 3 s. Una
ventana temporal solo aporta un rasgo si todos sus intervalos son positivos y
válidos. Un timestamp ausente, ilegible o compartido no se sustituye por cero;
la duración de una acción tampoco se infiere de este intervalo. Los registros
vinculados de un mismo lance se agrupan antes de medirlo. Los n-gramas base
siguen presentes aunque falte tiempo. No se construyen matrices de distancias
entre jugadas.

Para repetir la comparación completa y medir recursos en Linux:

```bash
/usr/bin/time -v .venv/bin/python -m src.classification.exploratory_nmf \
  datos_preparados_corpus datos_clasificacion_cronologia_diversa \
  --comparar-ritmo --max-partidos 12 --componentes 4 --semilla 42
```

`--comparar-ritmo` selecciona de forma determinista dos partidos de cada una
de las seis temporadas disponibles. La muestra contiene 14 equipos protagonistas
distintos. La semilla 42 reserva tres partidos completos y mantiene las mismas
1,025 jugadas y particiones en `base`, `posicional` y `temporal`. Los vocabularios,
TF-IDF y NMF se ajustan solo con los nueve partidos de entrenamiento. Cada
subcarpeta conserva `resultados.json` y los modelos; `comparacion.json` enlaza
las tres variantes y sus métricas por categoría.

La corrida diversa del 22 de octubre de 2026 tardó **70.83 s** y alcanzó
**356,360 KiB** de memoria residente máxima. Las matrices dispersas de
entrenamiento fueron:

| Categoría | Variante | Filas × columnas | Elementos no nulos |
|---|---|---:|---:|
| Balón parado | Base | 292 × 1,782 | 7,530 |
| Balón parado | Posicional | 292 × 3,998 | 14,043 |
| Balón parado | Temporal | 292 × 4,202 | 15,045 |
| Juego abierto | Base | 516 × 2,626 | 15,072 |
| Juego abierto | Posicional | 516 × 6,151 | 28,599 |
| Juego abierto | Temporal | 516 × 6,359 | 29,772 |

| Categoría | Variante | Cobertura media reservada | Error L2 medio reservado |
|---|---|---:|---:|
| Balón parado | Base | 0.8946 | 0.9322 |
| Balón parado | Posicional | 0.8605 | 0.9505 |
| Balón parado | Temporal | 0.8588 | 0.9443 |
| Juego abierto | Base | 0.9213 | 0.9424 |
| Juego abierto | Posicional | 0.8973 | 0.9596 |
| Juego abierto | Temporal | 0.8892 | 0.9558 |

La corrida de las tres variantes sobre los doce partidos consecutivos usados
en el piloto inicial tardó 57.54 s, con 361,172 KiB máximos y 1,028 jugadas.
Sirve como control de costo; la tabla anterior corresponde a la muestra diversa.

Al emparejar componentes por coseno de sus pesos en los n-gramas **base
compartidos**, las similitudes de la variante posicional fueron 0.997–0.999 y
las de la temporal 0.841–0.998. En ambas variantes los n-gramas base siguieron
encabezando la mayoría de los componentes. En juego abierto, el componente
caracterizado por `Pass:avance:medio → Carry:lateral:corto` siguió presente; en
la variante temporal apareció junto a su intervalo `medio`.

La jugada representativa `c73_t235_m3835102_p0066` (Mazatlán, balón parado)
encabeza el mismo componente de pases y conducciones en las tres variantes.
En juego abierto, `c73_t351_m4063524_p0067` y
`c73_t351_m4063524_p0070` (Atlante ante presión de Necaxa) intercambian
posición entre los ejemplos altos del componente con interrupciones; en ambos
casos los eventos muestran pases, conducciones y presiones rivales. La señal
temporal cambia pesos, pero estos ejemplos no revelan una diferencia táctica
nueva y clara.

Se repitió la muestra diversa con semilla 43 y **los mismos tres partidos
reservados** (`3799351`, `3939817`, `4004041`), para aislar el cambio de
semilla. Los ocho pares de componentes de cada variante tuvieron coseno cercano
a 1 en sus vocabularios completos y ninguna de las 1,025 jugadas cambió de
componente dominante. Las mayores diferencias absolutas de peso `W` fueron
0.000112 (base), 0.000392 (posicional) y 0.000167 (temporal). Esta coincidencia
es evidencia limitada porque `nndsvda` proporciona una inicialización casi
determinista; no sustituye una prueba de estabilidad entre muestras. La segunda
corrida tardó 62.82 s y alcanzó 357,108 KiB.

El error L2 y la cobertura se calculan en vocabularios TF-IDF distintos, así
que sus valores entre variantes no equivalen a una comparación directa de
calidad. **El piloto no demuestra una mejora de los componentes** al añadir
posición o ritmo; se mantiene la base como referencia exploratoria. La
selección de componentes útiles y su interpretación futbolística más amplia
corresponden al ticket #18.
