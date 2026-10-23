# Zonas y equivalencia entre bandas (#15)

El piloto compara la representación base del #14 con otra que añade n-gramas
espaciales. Ambas corridas usan la misma muestra, partición por partido, semilla
y número de componentes solicitado:

```bash
python -m src.classification.exploratory_nmf \
  datos_preparados_corpus datos_clasificacion_espacial \
  --max-partidos 12 --componentes 4 --semilla 42 \
  --comparar-espacio --zonas-x 6 --zonas-y 4 --peso-espacial 1.0
```

Antes de crear rasgos espaciales se comprueba la convención de orientación en
los partidos elegidos: se requieren al menos dos tiros por cada valor físico de
`attacking_direction`; la mediana de `location.x` debe superar el centro del
campo y las medianas de desplazamiento de tiros y pases deben avanzar hacia
`x=120` en ambos sentidos. La evidencia queda en `espacial/resultados.json`. Si no se
confirma, la corrida falla sin crear resultados. En el corpus preparado, los
tiros de ambos sentidos físicos apuntan a `x=120` en coordenadas del equipo que
actúa; **no** se invierte `x` de una acción propia por `attacking_direction`.

Para acciones rivales, origen y destino se rotan 180 grados (`x'=120-x`,
`y'=80-y`) al marco del protagonista. Después se aplica una única reflexión
transversal a toda la jugada si la primera coordenada propia observada fuera
del centro, de origen o destino, está en `y>40`. Así, dos jugadas reflejadas entre bandas reciben las mismas zonas y
direcciones sin perder su geometría relativa. `banda_original` (`y_baja`,
`y_alta`, `centro` o `sin_dato`) y `reflejada` se guardan por `play_id` como
metadatos; los JSON de eventos preparados no se modifican.

La cuadrícula divide el campo 120×80 en las cantidades configuradas de zonas
longitudinales y transversales. Los tokens incluyen zona de origen, zona de
destino, avance longitudinal, desplazamiento transversal y tramo de distancia.
Los umbrales de avance (`5` unidades) y distancia (`10` y `30`) se registran en
los parámetros. La ausencia de coordenadas se expresa como `sin_dato`; una
coordenada fuera del campo se marca `fuera_campo` y no genera un movimiento
calculado. Los n-gramas espaciales de longitud 2 y 3 reciben el peso elegido
antes de TF-IDF; los n-gramas base conservan su tokenización y peso 1 en ambas
variantes. La rotación rival se aplica a los rasgos espaciales añadidos; la
tokenización base heredada del #14 se mantiene para aislar la comparación.

`comparacion.json` enlaza `base/resultados.json` y
`espacial/resultados.json` y resume errores y cobertura del vocabulario en
partidos reservados por categoría. Son diagnósticos exploratorios: cada
variante ajusta vocabulario y NMF propios, así que sus identificadores de
componente no se comparan por número ni se presentan como estilos validados.
