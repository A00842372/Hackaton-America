# Intervenciones rivales breves

La segmentación primero delimita tramos continuos por partido, periodo y equipo de
`possession_team`. Una salida (`out`, pase `Out`, tiro fuera), falta sin ventaja,
detención, reanudación o cambio de periodo crea una barrera. Solo dos fronteras
entre tramos causadas por cambios de poseedor pueden formar A → rival → A.

La unión exige **ambos** límites, inclusivos: máximo tres segundos y máximo tres
acciones relevantes del rival. Se recorren los tramos en orden de evento; una
unión consume sus tres tramos y puede prolongarse con otra intervención breve.
Un tramo consumido nunca se emite por separado. Si falla cualquier límite o
hay una barrera, cada tramo conserva su propia evaluación de elegibilidad.

## Tiempo

Los `timestamp` se interpretan como inicios de acción y `duration` como segundos
desde ese inicio. El comienzo de la interrupción es el marcador más temprano
entre el fin de la última acción del protagonista y el inicio de la primera
acción rival. El final es el más tardío entre el inicio de la primera acción
del protagonista al recuperar el control y los fines de las acciones rivales.
Esto incluye el tiempo con balón suelto y evita ignorar una acción rival cuya
duración se extiende más allá del inicio del evento de retorno. Los cálculos
usan segundos decimales, de modo que 3.000 se acepta y 3.001 se rechaza.
Una duración ausente vale cero; un `timestamp` ausente o una duración inválida
impide confirmar el límite y por tanto impide la unión.

## Acciones relevantes

Se cuentan los registros del equipo rival durante su tramo de control. Pase,
conducción, regate y tiro son acciones distintas incluso cuando se enlazan
mediante `related_events`. Los registros simultáneos relacionados de contacto
o recuperación (`Duel`, `Ball Recovery`, `Interception`, `Dispossessed`) se
cuentan como un mismo lance. `Ball Receipt*`, `Dribbled Past` y `Pressure` no
añaden otra acción si están relacionados con una acción principal del tramo;
si aparecen aislados, cuentan como una. Las filas defensivas del protagonista
dentro del tramo rival se conservan, pero no aumentan el contador rival.

La exportación aplica el mínimo de pases después de estas uniones. El
manifiesto conserva todas las posesiones de origen y enumera los `id` de cada
evento cuyo `possession_team` es rival dentro de la jugada guardada. Los pases
rivales permanecen en el archivo de eventos y no suman para la elegibilidad
del protagonista. Los límites usados quedan en `parametros` del manifiesto.
