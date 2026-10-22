# Contexto de los DataFrames de jugadas

## De dónde salen

Los archivos descargados en `datos_statsbomb/competicion_*/temporada_*/eventos/<partido_id>.json` contienen los eventos de cada partido. Al cargarlos se forma un `df_eventos`: **cada fila es un evento de StatsBomb**, ordenado por `period` e `index`. Sus columnas proceden del archivo de ese partido.

Una **jugada** es un tramo continuo de esos eventos atribuido a un equipo protagonista. Se guarda como un DataFrame separado solo si el protagonista completó al menos tres pases. Puede incluir varias posesiones base y una intervención breve del rival que terminó con la recuperación del balón. Por eso, una jugada no equivale necesariamente a un único valor de `possession`.

## Cómo es el DataFrame de una jugada

Cada fila sigue siendo un evento original; el DataFrame de la jugada conserva **las mismas columnas, en el mismo orden, que `df_eventos` de su partido**. No se agrega una fila resumen ni se añaden columnas como `play_id` o `es_rival`. Una acción futbolística puede estar descrita por varios eventos relacionados.

Las columnas principales se pueden leer así:

| Columnas | Información |
|---|---|
| `id`, `match_id`, `index`, `period` | Identidad del evento, partido y orden dentro del partido. |
| `timestamp`, `minute`, `second`, `duration` | Momento y duración disponible del evento. |
| `type` | Tipo de evento, normalmente un objeto con `id` y `name` (`Pass`, `Carry`, `Shot`, etc.). |
| `team`, `possession_team`, `possession` | Equipo que realiza el evento, equipo al que StatsBomb atribuye la posesión y número de posesión base. `team` y `possession_team` pueden diferir en una acción defensiva. |
| `player`, `position`, `location` | Jugador, posición y coordenadas del evento cuando están disponibles. |
| `play_pattern`, `attacking_direction` | Contexto de la jugada y dirección de ataque informada en los datos originales. |
| `pass`, `carry`, `shot`, `dribble`, `duel`, `goalkeeper`, etc. | Detalles propios de cada tipo de evento. Son objetos anidados; normalmente valen `null` en filas de otros tipos. |
| `related_events` | Identificadores de otros eventos vinculados al mismo lance. |
| `under_pressure`, `out`, `counterpress` y campos `obv_*` | Indicadores y métricas que ya venían en los eventos descargados. La exportación los conserva, sin recalcularlos. |

También se conservan otros campos presentes en el partido, por ejemplo `ball_recovery`, `interception`, `foul_committed`, `foul_won`, `clearance`, `tactics` o `defensive_responsibility`. **La lista exacta varía entre partidos** y está en `manifiesto.json` → `columnas`. Una columna puede existir y contener `null` en todas las filas de una jugada si ese tipo de evento no ocurrió allí. Los diccionarios y listas permanecen anidados; los valores faltantes se guardan como `null`.

## Archivo de jugada y manifiesto

Cada jugada elegible se guarda en `datos_preparados_corpus/competicion_<id>/temporada_<id>/partido_<id>/jugada_0001.json` (y números siguientes). El JSON contiene `columns` y `data`; `leer_jugada(ruta)` devuelve el DataFrame. Las filas se mantienen en orden cronológico.

El `manifiesto.json` de ese partido resume cada jugada: `play_id`, equipo protagonista, periodo, primer y último evento, posesiones de origen, pases completados e IDs de intervenciones rivales absorbidas. `play_id` y los resúmenes de la jugada no se añaden como columnas al DataFrame. Allí también se auditan los eventos que no formaron parte de jugadas guardadas.

Los datos `frames_360` se almacenan aparte: no se insertan como filas ni columnas adicionales en estos DataFrames de jugadas.
