# Análisis de jugadas de fútbol

Este contexto describe las secuencias de juego obtenidas de eventos de StatsBomb para analizar patrones tácticos.

## Language

**Evento**:
Registro individual de StatsBomb asociado con un partido. Una acción futbolística puede estar representada por varios eventos relacionados.

**Posesión base**:
Bloque inicial de eventos de un partido y periodo con un mismo identificador `possession` y un equipo `possession_team`.

**Secuencia candidata**:
Secuencia delimitada por las reglas de continuidad y cierre, antes de comprobar el mínimo de pases del equipo protagonista.

**Jugada**:
Secuencia de juego de un equipo protagonista, delimitada por las reglas de continuidad y cierre. Puede abarcar varias posesiones base e incluir una intervención rival absorbida.
_Avoid_: Posesión, cuando se habla de la secuencia ya segmentada.

**Equipo protagonista**:
Equipo al que se atribuye la secuencia sostenida de una jugada.

**Jugada elegible**:
Jugada con al menos tres pases completados del equipo protagonista. Es el subconjunto de jugadas que entra a la clasificación principal.
