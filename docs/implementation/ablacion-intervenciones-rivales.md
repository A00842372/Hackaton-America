# Intervenciones rivales y ablación (#16)

La representación usa una acción por lance antes de formar n-gramas de dos y
tres acciones. Un `Duel`, `Ball Recovery`, `Interception` o `Dispossessed`
simultáneo y relacionado se agrupa, incluso si una fila registra al jugador
desposeído y otra al rival que disputa el balón. Se prefiere el registro de
recuperación para conservar su tipo y coordenadas; después, el de duelo sobre
el de desposesión. Las relaciones en cadena forman un solo lance sin depender
del orden de las filas. `Ball Receipt*`,
`Dribbled Past`, `Pressure` y `Foul Won` relacionados con una acción principal
se tratan como contexto. Para `Dribbled Past`, `Ball Receipt*` y `Foul Won`,
la acción principal puede pertenecer al otro equipo; la presión del rival
vinculada a una acción propia se conserva porque describe una intervención
distinta. Si aparecen aislados, pueden aportar
una acción. `Pass`, `Carry`, `Dribble` y `Shot` siguen siendo decisiones
distintas aunque sus registros estén vinculados. Se exige un `timestamp`
presente para agrupar registros simultáneos.

El prefijo del token identifica la relación con el equipo protagonista:

- `propio`: acción del protagonista.
- `rival_interrupcion`: acción rival mientras la posesión registrada aún
  pertenece al protagonista, como una presión.
- `rival_recuperacion`: primera acción rival en un tramo cuya
  `possession_team` pasa al equipo rival.
- `rival_control`: acciones posteriores durante ese tramo rival breve.

La distinción usa la transición de `possession_team`, no infiere el éxito de un
duelo por su tipo. Una transición no observada en los datos no se inventa. Las
zonas rivales se llevan al marco del protagonista con la rotación documentada
en [zonas-bandas-clasificacion.md](zonas-bandas-clasificacion.md). El prefijo
se conserva en los tokens básicos y espaciales. Los eventos preparados quedan
intactos.

Para medir la influencia rival, se ajustan dos modelos independientes por
categoría sobre las mismas jugadas y la misma partición por partido. La
variante `sin_rivales` elimina las acciones rivales **antes** de generar
n-gramas; así compara la secuencia propia que se leería sin esas
intervenciones. Ambas variantes usan la misma cuadrícula, semilla y número de
componentes solicitado. La corrida nueva se ejecuta con:

```bash
.venv/bin/python -m src.classification.exploratory_nmf \
  datos_preparados_corpus datos_clasificacion_rivales \
  --comparar-rivales --max-partidos 12 --componentes 4 --semilla 42
```

`comparacion.json` enlaza las dos carpetas de resultados y, por categoría,
publica fichas con los diez n-gramas de mayor peso, tres jugadas ejemplo por
componente y fracciones de masa de `H` que contienen tokens rivales,
interrupciones, recuperaciones, registros contextuales o zonas `sin_dato`.
Una fracción de al menos 0.5 dispara una alerta de posible dominio rival,
interrupción o artefacto de registro. Son señales de revisión, no un veredicto
táctico. La masa se calcula sobre **todo** el vocabulario del componente, no
solo sobre sus diez términos destacados.

Los componentes se emparejan por coseno de `H` restringido a n-gramas no
rivales presentes en ambos vocabularios; la asignación global maximiza la
similitud. Si no hay contenido compartido, el par se marca no comparable y su
cambio de contribución queda vacío. La comparación registra los n-gramas y
ejemplos que entran o salen del grupo destacado. Para cada `play_id`, la
contribución comparable es `W_k × ||H_k||₂` dividida entre la suma de esos
productos en la jugada. Esto elimina la escala arbitraria entre `W` y `H`,
pero no produce probabilidades calibradas. Los pesos NMF originales siguen
en cada `resultados.json`.

El piloto de doce partidos contiene 1,028 jugadas, con nueve partidos de
entrenamiento y tres reservados. Dos de los ocho componentes de la variante
con rivales superan la alerta de masa rival: uno de balón parado y uno de
juego abierto, ambos dominados principalmente por n-gramas con
`rival_interrupcion`. Esto exige revisar sus ejemplos antes de tratar esas
etiquetas exploratorias como patrones futbolísticos. No se les asignan nombres
tácticos ni se publica una clasificación validada en este ticket.
