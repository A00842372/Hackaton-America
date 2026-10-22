# Segmentación y clustering de jugadas sostenidas

Fecha: 21 de octubre de 2026.
Revisión: 22 de octubre de 2026; se precisa la entrega de la segmentación en `Data_Preparation.IPYNB`.

Estado: contexto y diseño funcional confirmados por el usuario. Publicada como [issue #1](https://github.com/A00842372/Hackaton-America/issues/1), con la etiqueta ready-for-agent. Las decisiones de pruebas que se proponen abajo requieren revisión; no se presentan como acuerdos previos.

## Problem Statement

El analista quiere identificar patrones tácticos en secuencias completas de juego, desde que un equipo obtiene control del balón hasta que pierde la posesión de forma sostenida o el juego se interrumpe. Los datos actuales describen eventos individuales; las secuencias tienen distintas duraciones y cantidades de acciones.

La clasificación debe cubrir todas las jugadas que cumplan un mínimo de tres pases completados. Exigir datos 360 completos como condición para entrar al análisis principal descartaría gran parte del material disponible. El analista necesita una clasificación común que conserve secuencia, progresión, ritmo y altura en el campo, con grupos de detalle intermedio e interpretación futbolística.

## Solution

Construir jugadas a partir de las posesiones de StatsBomb, incorporando una tolerancia explícita a intervenciones rivales fugaces. Conservar los eventos originales y la procedencia de cada jugada. Aplicar el mínimo de pases después de resolver la continuidad.

Clasificar todas las jugadas elegibles mediante características de eventos. Los grupos deben representar formas de desarrollar una jugada: distinguir un ataque rápido de una circulación prolongada aunque recorran zonas similares, y considerar equivalentes las ejecuciones reflejadas entre izquierda y derecha.

Desarrollar posteriormente un complemento con datos 360 para describir o subdividir las jugadas según el espacio disponible. La ausencia de 360 no afecta la elegibilidad ni la asignación principal.

## User Stories

1. Como analista, quiero procesar todos los partidos locales con eventos para aprovechar el conjunto disponible.
2. Como analista, quiero incluir partidos sin 360 para obtener una clasificación de todas las jugadas elegibles.
3. Como analista, quiero que cada jugada conserve sus eventos ordenados para reconstruir cómo se desarrolló.
4. Como analista, quiero identificar al equipo protagonista para atribuir correctamente pases y progresión.
5. Como analista, quiero iniciar jugadas desde recuperaciones para estudiar ataques nacidos en juego abierto.
6. Como analista, quiero iniciar jugadas desde saques y otras reanudaciones para incluir todas las formas de adquirir posesión.
7. Como analista, quiero conservar la continuidad ante desvíos que no interrumpen el control para evitar fragmentar una misma jugada.
8. Como analista, quiero absorber intervenciones rivales de hasta tres segundos y tres acciones relevantes cuando el balón vuelve al equipo original para representar secuencias sostenidas.
9. Como analista, quiero conservar las acciones rivales absorbidas y distinguirlas de las propias para entender las interrupciones.
10. Como analista, quiero que los pases rivales no cuenten para el mínimo de pases del protagonista para seleccionar correctamente las jugadas.
11. Como analista, quiero cerrar las jugadas ante pérdidas que superen la tolerancia para separar posesiones sostenidas distintas.
12. Como analista, quiero cerrar las jugadas ante salidas o detenciones para mantener una unidad de juego continuo.
13. Como analista, quiero mantener la jugada tras un tiro propio y rebote recuperado por el mismo equipo para conservar la continuidad ofensiva.
14. Como analista, quiero que una falta con ventaja no corte automáticamente la jugada para respetar la continuidad real del juego.
15. Como analista, quiero separar partidos y periodos para impedir uniones artificiales entre contextos distintos.
16. Como analista, quiero seleccionar secuencias con al menos tres pases completados del protagonista para concentrarme en jugadas sostenidas.
17. Como analista, quiero distinguir secuencias de acciones con igual cantidad de pases pero distinto desarrollo para preservar información táctica.
18. Como analista, quiero distinguir progresión directa, retrocesos y circulación para representar cómo avanza el equipo.
19. Como analista, quiero conservar el ritmo y la duración para diferenciar ataques rápidos y posesiones prolongadas.
20. Como analista, quiero conservar la altura en el campo para distinguir la construcción en campo propio de las acciones cerca del área rival.
21. Como analista, quiero que las ejecuciones reflejadas entre bandas sean equivalentes para reconocer el mismo patrón en ambos lados.
22. Como analista, quiero una etiqueta principal para cada jugada elegible para comparar todo el repertorio con un criterio común.
23. Como analista, quiero revisar ejemplos representativos y ejemplos de peor encaje de cada grupo para valorar su coherencia táctica.
24. Como analista, quiero una granularidad intermedia para evitar tanto categorías demasiado generales como grupos definidos por detalles irrelevantes.
25. Como analista, quiero conservar los identificadores de posesión y eventos de origen para auditar y revisar la segmentación.
26. Como analista, quiero usar el contexto de play_pattern para interpretar las jugadas sin confundirlo con su identificador.
27. Como analista, quiero conocer la cobertura 360 por jugada para seleccionar correctamente el análisis espacial complementario.
28. Como analista, quiero medir distancias a defensores y espacio disponible cuando existan datos 360 para enriquecer la interpretación táctica.
29. Como analista, quiero conservar la etiqueta principal al añadir el complemento espacial para mantener comparabilidad entre jugadas con y sin 360.
30. Como analista, quiero comparar resultados y cobertura por temporada para detectar diferencias del conjunto y revisar la estabilidad de los patrones.
31. Como analista, quiero guardar por separado cada jugada que cumpla los requisitos de continuidad y el mínimo de tres pases completados del protagonista, para analizar únicamente las secuencias seleccionadas por esta especificación.
32. Como analista, quiero recuperar todas las columnas originales de los eventos de cada jugada guardada y localizarla por competición, temporada y partido.

## Implementation Decisions

### Vocabulario y estado actual

- Un evento es un registro original de StatsBomb. Una acción futbolística puede producir varios registros asociados; el límite de acciones no equivale a contar filas.
- Una posesión base es un bloque identificado por partido, periodo y possession. El equipo poseedor se obtiene de possession_team.
- Una jugada es la secuencia resultante de aplicar las reglas de continuidad y cierre. El equipo protagonista es aquel cuya secuencia sostenida se está representando.
- Una jugada elegible contiene al menos tres pases completados del protagonista después de segmentar y resolver las uniones. El vocabulario de dominio se recoge también en `CONTEXT.md`.
- La segmentación produce secuencias candidatas; después de resolver la continuidad, solo las jugadas elegibles se guardan y entran a la clasificación principal. Las candidatas con menos de tres pases completados del protagonista se contabilizan, pero no se guardan como archivos de jugada.
- El proyecto contiene un notebook de exploración y descarga, datos locales de eventos y 360, y `Data_Preparation.IPYNB` aún sin código. No se encontró una interfaz existente de segmentación o clustering ni una suite automatizada de pruebas.

### Construcción de jugadas

- Procesar los eventos en orden de partido, periodo e index. Conservar timestamps, duraciones disponibles e identificadores originales.
- Usar partido, periodo y possession como bloques iniciales. Un cambio de team en una fila no implica por sí solo una pérdida: puede representar una acción defensiva dentro de la posesión rival.
- Iniciar desde control adquirido en recuperación o reanudación, incluidos saques de banda, de meta, córners y faltas.
- Mantener los desvíos y rebotes cuando persista la continuidad. Un tiro propio con recuperación propia no cierra por sí solo la jugada.
- Unir A → intervención rival → A cuando el intervalo intermedio dure como máximo tres segundos, comprenda como máximo tres acciones relevantes y no atraviese una salida, detención o cambio de periodo. Los límites son inclusivos. El umbral temporal de tres segundos reemplaza la propuesta anterior de cinco segundos.
- Conservar los eventos de la intervención dentro de la jugada unida, distinguiendo al rival del protagonista. El tramo absorbido no se emite además como una jugada independiente.
- El límite temporal tiene como propósito absorber controles fugaces. La propuesta de imponer una excepción adicional por tiro o tercer pase rival no fue aceptada como regla independiente; aplicar las reglas confirmadas de continuidad, límites y detención.
- Si el rival conserva el balón más allá de cualquiera de los límites o no se produce el retorno permitido, separar las secuencias y evaluar cada una con el filtro de elegibilidad.
- Una salida, una detención efectiva o un cambio de periodo cierra la continuidad, aunque el mismo equipo reanude. Una falta con ventaja no equivale automáticamente a una detención.
- Resolver la continuidad antes de filtrar por pases y antes de filtrar por cobertura 360. Conservar la cronología completa para identificar pérdidas e interrupciones.
- Para guardar y clasificar una jugada, exigir al menos tres pases completados del protagonista; los pases incompletos, las recepciones, las conducciones y los pases rivales no incrementan ese contador.
- En los JSON inspeccionados, un pase completado se reconoce por ausencia de pass.outcome. Adaptar la lectura de datos normalizados a esa semántica, distinguiendo campos ausentes de información corrupta.
- No inferir balón dividido únicamente de is_controlled_possession=False o control_degree=0: la inspección mostró acciones con balón y posesión controlada en esos casos.
- Conservar play_pattern como contexto. No utilizar sus categorías como identificadores únicos de jugada, ni los números de possession como variables numéricas de similitud.

### Representación y clasificación principal

- La primera clasificación utilizará únicamente eventos y abarcará todas las jugadas elegibles de todos los partidos locales con eventos.
- La representación conservará información de secuencia, progresión, ritmo y altura. Los agregados que eliminen por completo el orden de acciones no satisfacen por sí solos el requisito de secuencia.
- Orientar las jugadas de manera comparable respecto a la portería atacada y tratar como equivalentes las reflexiones entre izquierda y derecha. Aplicar la transformación de forma coherente a los eventos de ambos equipos.
- La equivalencia entre bandas no autoriza a descartar información necesaria para medir recorridos, amplitud o espacio; la representación debe respetar la invariancia por reflexión acordada.
- Distinguir tiempos de juego e intervenciones para que una transformación temporal no vuelva indistinguibles un ataque rápido y una circulación prolongada.
- Dar una asignación principal a cada jugada elegible. La política de datos faltantes en eventos deberá preservar la cobertura y registrar su tratamiento; la ausencia de 360 nunca será motivo de descarte principal.
- Buscar grupos de granularidad intermedia. Revisar jugadas representativas y de peor encaje; subdividir mezclas tácticas evidentes y evitar divisiones sin una diferencia futbolística útil.
- Los nombres de grupos se obtendrán de la revisión de sus miembros. Los ejemplos discutidos son orientativos, no una taxonomía predefinida.
- El algoritmo, la representación numérica exacta, los pesos, el número de grupos y los criterios cuantitativos de ajuste todavía no están seleccionados. K-means fue una alternativa discutida, no una decisión cerrada.
- El componente analítico debe poder ejecutarse a partir de eventos locales sin depender de la descarga ni del estado interactivo del notebook. La organización concreta de módulos se decidirá durante la implementación.
- La salida de segmentación debe vincular cada jugada elegible guardada con su partido, periodo, protagonista, intervalo, eventos y posesiones de origen, y cantidad de pases completados. La clasificación posterior debe vincular su asignación principal con esas mismas jugadas. Registrar parámetros y transformaciones para poder reproducir ambos resultados.

### Entrega en `Data_Preparation.IPYNB`

- El notebook debe poder ejecutarse de principio a fin a partir de los archivos locales, sin credenciales ni variables creadas en `EDA.IPYNB`. Debe descubrir `datos_statsbomb/competicion_*/temporada_*/eventos/<match_id>.json` y procesar cada archivo de eventos disponible; `partidos.json` puede aportar nombres y fechas, pero su ausencia no debe impedir la segmentación.
- Los archivos locales de eventos son diccionarios JSON cuyas claves son identificadores de evento y cuyos valores son registros. Construir para cada partido `df_eventos` con todos esos registros y todas sus columnas; ese DataFrame es la referencia de integridad para sus jugadas. Validar la concordancia entre la clave, el campo `id` del registro y el `match_id` del nombre de archivo antes de guardar resultados. Si un DataFrame de otra procedencia se usa como entrada, declarar su esquema como referencia antes de segmentar.
- Segmentar sobre el DataFrame completo de cada partido, ordenado por `period` e `index`, con `id` como identificador de evento. Conservar el orden original de las columnas y el valor de cada celda en los subconjuntos de jugada. Resolver la continuidad antes de contar pases; la falta de 360 no impide guardar una jugada que cumple el mínimo de pases.
- Guardar **solo las jugadas elegibles**, una por archivo, después de aplicar todas las reglas de continuidad y exigir al menos tres pases completados del protagonista. Una secuencia candidata con menos de tres pases no genera archivo de jugada. Los registros administrativos ajenos al juego, como `Starting XI` o `Half Start`, no forman jugadas: contabilizarlos aparte y documentar qué tipos se excluyen, sin dejar de usar los marcadores de inicio y fin de periodo para cerrar la continuidad. Cada evento de una jugada guardada debe figurar una sola vez entre sus archivos, incluso cuando se absorbe una intervención rival. Los eventos no exportados deben figurar en la auditoría con su identificador y motivo; no se permite descartarlos silenciosamente.
- Usar JSON tabular válido para cada jugada, con la lista explícita de columnas en su orden original y las filas ordenadas cronológicamente (por ejemplo, `orient="split"` o una estructura equivalente). Al recargarlo se deben recuperar todas las columnas de `df_eventos`, incluidos datos anidados; no convertir diccionarios o listas a cadenas ni redondear deliberadamente valores numéricos. Representar los valores faltantes como `null`, incluidos los posibles `NaN` del origen. La salida de eventos no debe añadir columnas auxiliares. Si se necesitan indicadores como `play_id`, `is_opponent_intervention` o la elegibilidad, guardarlos en el manifiesto o en datos auxiliares separados. La equivalencia exigida es de nombres, orden y valores de columnas bajo esa normalización de faltantes; los tipos de pandas inferidos al releer JSON pueden diferir.
- Crear una carpeta por competición, temporada y partido. Usar los identificadores de las rutas y del archivo como claves estables; los nombres de equipos son información opcional del manifiesto. La estructura mínima será:

  ```text
  <directorio_de_salida>/
    competicion_73/temporada_351/partido_4063635/
      jugada_0001.json
      jugada_0002.json
      ...
      manifiesto.json
  ```

- Numerar `jugada_0001`, `jugada_0002`, etc. solo entre las jugadas elegibles, por el primer evento de cada una en orden de periodo e `index`, sin reiniciar por periodo. Un cambio de periodo separa jugadas; una jugada A → intervención rival → A puede contener varias posesiones base del mismo periodo.
- El `manifiesto.json` de cada partido debe relacionar cada `play_id` guardado con su archivo, periodo, protagonista, primer y último `id` e `index`, posesiones de origen, identificadores de eventos de las intervenciones rivales absorbidas y número de pases completados. Construir `play_id` de manera estable y única en el corpus a partir de competición, temporada, partido y número de jugada guardada. Registrar también las columnas originales en orden, el nombre de temporada si está disponible, número de eventos de entrada, eventos exportados, secuencias candidatas descartadas por no alcanzar tres pases, eventos no exportados con su `id` y motivo, y parámetros usados. No sustituye los archivos individuales.
- Definir en una celda de configuración la ruta de entrada, una ruta de salida distinta, los límites de continuidad y la política de reejecución. Ante una carpeta de partido ya existente, fallar antes de escribir, salvo que el usuario active explícitamente el reemplazo de los resultados generados para ese partido. Una reejecución con los mismos datos y parámetros debe producir la misma numeración y pertenencia de eventos.
- Al terminar, mostrar partidos procesados, jugadas elegibles guardadas, secuencias candidatas descartadas por tener menos de tres pases, eventos no exportados, errores por partido y ruta de salida. Los errores de lectura o integridad deben identificar el archivo afectado, quedar en el resumen y hacer que la ejecución se declare incompleta; los resultados de un partido fallido no deben presentarse como correctos.

### Complemento 360 posterior

- Incorporarlo después de consolidar la clasificación principal. Su papel es aportar información sobre espacio disponible y distancias a defensores, manteniendo una etiqueta principal común para todas las jugadas.
- Exigir cobertura completa de las acciones de balón utilizadas por la representación espacial de la jugada, incluidas las intervenciones rivales conservadas.
- La auditoría inicial seleccionó Pass, Carry, Dribble, Shot, Ball Recovery y Ball Receipt*. La selección definitiva de acciones representadas debe documentarse; si incorpora duelos, intercepciones u otras acciones, recalcular la cobertura correspondiente.
- Una coincidencia de identificador 360 con freeze_frame vacío no aporta posiciones. El enriquecimiento debe validar los datos requeridos por cada métrica y considerar el área visible.
- Tener un frame no implica ver los 22 jugadores. No interpretar ausencia de jugadores fuera de cámara como espacio libre.
- La unión debe respetar la cardinalidad: en el DataFrame 360 hay varias filas de jugadores por evento. El enriquecimiento no debe multiplicar artificialmente los eventos o pases.
- Los resultados espaciales pueden describir o subdividir las jugadas cubiertas; no sustituir las etiquetas principales por etiquetas de otro modelo como si fueran equivalentes.

### Detalles técnicos por resolver durante la implementación

- Definir y documentar cómo se agrupan registros asociados para contar acciones relevantes, utilizando la semántica de tipos y relaciones de eventos. El criterio debe cumplir el límite acordado sin contar un mismo lance varias veces.
- Precisar cómo se mide el comienzo de una interrupción y el retorno del control utilizando timestamps y duraciones, teniendo en cuenta que el timestamp de una acción puede marcar su inicio y su desenlace ocurrir después.
- Resolver de forma determinista las posibles uniones consecutivas o solapadas y conservar trazabilidad. Una intervención absorbida no debe generar duplicados independientes.
- Elegir representación, algoritmo, granularidad y política de datos faltantes mediante comparación y revisión de resultados. Estos detalles no se presentan como decisiones ya confirmadas por el usuario.

## Testing Decisions

Propuesta de puntos de prueba, pendiente de revisión: comprobar por un lado el flujo de segmentación y exportación del notebook a partir de un partido local o un conjunto pequeño de eventos, y por otro la clasificación sobre un conjunto representativo de partidos. Una interfaz analítica posterior puede recibir eventos locales, parámetros y datos 360 opcionales y devolver jugadas, trazabilidad, asignaciones y diagnóstico de cobertura. Evitar pruebas ligadas a funciones privadas, orden interno de cálculos o números arbitrarios de etiquetas.

La propuesta cubre segmentación, persistencia, representación y asignación mediante sus salidas observables. El antecedente disponible es el notebook y la auditoría de datos; no se identificó infraestructura previa de pruebas automatizadas que deba imitarse.

Casos de aceptación a verificar:

1. Recuperaciones y distintos tipos de reanudación producen inicios de jugada válidos.
2. Presiones o acciones defensivas intercaladas no causan cortes únicamente por cambiar team.
3. A → B → A se une cuando satisface simultáneamente tres segundos y tres acciones relevantes; comprobar exactamente los límites y casos que exceden cada uno por separado.
4. Un lance representado por varios registros no consume varias acciones por duplicación de la misma acción futbolística.
5. Salidas, detenciones y cambios de periodo impiden la unión incluso si A recupera dentro de los límites.
6. Un tiro propio con rebote propio mantiene la continuidad; una falta con ventaja no introduce un corte artificial.
7. Dos pases propios antes de una intervención tolerada y uno después producen tres pases para la jugada unida. Los pases rivales absorbidos no contribuyen.
8. Las secuencias de menos de tres pases completados no producen archivo de jugada ni entran a la clasificación; las de tres o más se guardan y se clasifican. Contar completados, no intentos ni recepciones.
9. Identificadores de posesión repetidos entre partidos o periodos no mezclan secuencias.
10. Las jugadas y sus eventos permanecen auditables; las intervenciones absorbidas no aparecen como jugadas independientes duplicadas.
11. Cada jugada elegible se escribe en un archivo independiente dentro de su carpeta de competición, temporada y partido; el manifiesto permite localizarla. Ninguna secuencia candidata que no cumpla el mínimo genera un archivo de jugada.
12. Al leer cualquier archivo de jugada, las columnas coinciden con las de `df_eventos` en nombres y orden; las filas y valores pertenecen solo a esa jugada y conservan la cronología y los datos anidados. Los faltantes se recuperan como faltantes y el JSON es válido.
13. Dos partidos con los mismos equipos en temporadas distintas producen carpetas distintas; un cambio de periodo impide unir jugadas, mientras que una unión permitida A → B → A conserva las distintas posesiones de origen en un único archivo.
14. Los identificadores de eventos exportados aparecen una sola vez entre los archivos de jugadas del partido. Los eventos de candidatas descartadas, los administrativos y los no asignados se contabilizan por separado con motivo; su suma con los eventos exportados concilia con el archivo de entrada.
15. Una segunda ejecución falla antes de sobrescribir una carpeta existente salvo reemplazo explícito; con el mismo origen y parámetros, la numeración y pertenencia de eventos son estables.
16. El resumen del notebook concilia partidos leídos, jugadas elegibles guardadas, candidatas descartadas por pases, eventos no exportados y errores. Un error hace que la ejecución se declare incompleta e identifica el partido afectado.
17. Reflejar una jugada de izquierda a derecha conserva su representación equivalente y su asignación con el mismo modelo; modificar ritmo o altura conserva una diferencia medible en la representación.
18. Dos secuencias con iguales totales pero distinto orden relevante mantienen información que permita distinguir su desarrollo. No exigir que toda diferencia pequeña fuerce un grupo distinto.
19. Cada jugada elegible recibe una asignación principal, incluidas las de partidos sin 360. Quitar los datos 360 no modifica la segmentación ni las características principales basadas en eventos.
20. La incorporación de múltiples jugadores 360 por evento no altera el número de eventos o pases de la jugada.
21. Un frame vacío o un evento sin frame impide declarar cobertura espacial completa, sin eliminar la jugada del conjunto principal.
22. Mantener parámetros y semilla, cuando corresponda, permite reproducir asignaciones. Comparar agrupaciones sin depender del valor numérico arbitrario de las etiquetas.
23. Revisar muestras representativas, casos de peor encaje y estabilidad entre muestras de partidos para detectar grupos excesivamente amplios o fragmentados. La valoración táctica requiere revisión de ejemplos, además de métricas numéricas.

## Out of Scope

- Guardar o clasificar secuencias de menos de tres pases completados del protagonista como jugadas elegibles en esta versión.
- Condicionar la clasificación principal a la disponibilidad de 360.
- Fijar de antemano una taxonomía táctica o un número definitivo de clusters sin evaluar las jugadas.
- Tratar los grupos como etiquetas supervisadas previamente conocidas o usar KNN por sí solo como método de clustering.
- Reconstruir tracking continuo de los 22 jugadores a partir de los frames 360.
- Inferir posiciones defensivas no observadas como si fueran datos reales.
- Incorporar el resultado final, goles o xG como criterio principal de similitud; el objetivo acordado es describir la construcción de la jugada.
- Descargar nuevos partidos, desplegar una aplicación o entrenar modelos como parte de la redacción de esta especificación.
- Implementar el complemento 360 antes de consolidar la clasificación común basada en eventos.

## Further Notes

### Evidencia de disponibilidad y cobertura

La auditoría del conjunto local revisó 1,793 pares de archivos de eventos y 360, sin errores de lectura JSON. Hay 1,808 partidos con eventos: los 15 sin archivo 360 siguen dentro del alcance principal.

| Temporada | Partidos con eventos | Partidos con 360 | Posesiones base elegibles | Posesiones base con cobertura 360 completa |
|---|---:|---:|---:|---:|
| 2021/2022 | 342 | 341 | 28,510 | 9,432 |
| 2022/2023 | 342 | 342 | 30,744 | 11,072 |
| 2023/2024 | 340 | 340 | 31,904 | 11,837 |
| 2024/2025 | 340 | 340 | 32,423 | 12,816 |
| 2025/2026 | 337 | 328 | 29,646 | 12,166 |
| 2026/2027 | 107 | 102 | 9,519 | 4,245 |
| **Total** | **1,808** | **1,793** | **162,746** | **61,568** |

Los conteos de posesiones de la tabla corresponden únicamente a los 1,793 partidos con ambos archivos. Se agruparon por partido, periodo y possession y se exigieron tres pases completados del poseedor, sin aplicar las uniones de esta especificación. Por tanto, son evidencia exploratoria, no un conteo final de jugadas ni una expectativa rígida de prueba.

Para esa auditoría se exigió que cada acción de los seis tipos seleccionados tuviera un identificador 360 con una lista freeze_frame no vacía. Se conservaron 61,568 de 162,746 grupos, aproximadamente 37.83%. Todos los partidos emparejados aportaron al menos un grupo completo; ninguno cubrió todas las acciones seleccionadas de todo el partido.

La auditoría encontró 69,227 registros 360 con freeze_frame vacío, ausente o nulo. No validó el área visible ni cada coordenada. La validación que requiera una métrica espacial deberá añadirse antes de utilizar esa métrica.

El notebook inspeccionado mantiene en sus DataFrames de exploración un solo partido, Toluca–Necaxa del 13 de julio de 2025. El alcance de la implementación es el corpus local completo, no únicamente esos DataFrames en memoria.

### Referencias y publicación

- [Especificación oficial de eventos StatsBomb](https://github.com/hudl/open-data/blob/master/doc/StatsBomb%20Open%20Data%20Specification%20v1.1.pdf): semántica de possession, possession_team y play_pattern.
- El usuario confirmó el diseño funcional antes de solicitar esta síntesis mediante to-spec. Las propuestas de pruebas y los detalles técnicos todavía abiertos están identificados expresamente.
- El usuario confirmó GitHub Issues de A00842372/Hackaton-America como tracker y las etiquetas estándar needs-triage, needs-info, ready-for-agent, ready-for-human y wontfix. Esta especificación se publica con ready-for-agent. La configuración local de las habilidades se prepara mediante setup-matt-pocock-skills.
