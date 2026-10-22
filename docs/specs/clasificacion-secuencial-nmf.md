# Clasificación de jugadas mediante patrones secuenciales y NMF

Fecha: 22 de octubre de 2026.

Estado: diseño funcional confirmado por el usuario. Publicada como [issue #11](https://github.com/A00842372/Hackaton-America/issues/11) con la etiqueta `ready-for-agent`. Especificación nueva para la clasificación de jugadas; la segmentación y exportación actuales proporcionan el corpus de partida.

## Problem Statement

El analista dispone de un corpus local de 163,909 jugadas elegibles, obtenido de 1,808 partidos. Cada jugada conserva eventos ordenados y metadatos de procedencia, pero todavía no tiene una clasificación que describa regularidades recurrentes de su desarrollo. Las jugadas tienen distinta longitud y pueden mezclar acciones, zonas, cambios de ritmo e intervenciones rivales. Asignar nombres tácticos de antemano impondría categorías que los datos aún no han demostrado.

El analista quiere descubrir componentes interpretables de las secuencias y obtener una clase principal para cada jugada, conservando también las contribuciones de otros componentes. Las jugadas iniciadas en córners, tiros libres o saques de banda requieren un modelo distinto al de juego abierto. Cuando una jugada abarca ambas fases y cada tramo es suficientemente largo, la preparación actual debe producir jugadas independientes antes de clasificarlas. El corpus preparado ya existe: esta entrega debe modificarlo en su ubicación actual, sin reconstruirlo desde los eventos originales.

La utilidad de NMF es una hipótesis que debe comprobarse. Una partición guiada principalmente por la longitud, por artefactos de registro o por eventos auxiliares no satisfaría el objetivo. Si los componentes no aportan información reconocible y estable, se deben entregar los diagnósticos del intento sin presentar sus etiquetas como una clasificación válida.

## Solution

Añadir a la preparación una regla configurable que separe tramos de balón parado y juego abierto dentro de una jugada ya elegible. La regla se aplica ahora sobre el corpus preparado, en la misma ubicación, y también queda disponible al preparar partidos futuros. Una jugada se divide en todos los cambios válidos entre ambas categorías solo cuando **cada** tramo resultante tiene al menos tres pases completados del equipo protagonista. Si algún tramo no cumple ese mínimo, se conserva la jugada completa y se asigna al modelo correspondiente a la fase con más acciones de juego del protagonista. El corte puede ocurrir dentro de una posesión continua: se registra expresamente como corte para clasificación. Ningún evento se pierde o se duplica por esta operación.

Entrenar dos modelos NMF independientes, uno para balón parado y otro para juego abierto. Representar cada jugada mediante n-gramas consecutivos de longitud 2–3 de acciones, desplazamientos y zonas. Comparar la representación base con otra que indique si los n-gramas ocurren al inicio, en medio o al final de la jugada; estos tres tramos se definen por acciones ordenadas. Comparar también una variante con información temporal cuando el piloto confirme un costo razonable. Los tokens deben conservar las intervenciones rivales como tales y distinguir interrupciones de recuperaciones.

Cada jugada resultante recibe la clase del componente dominante de su modelo o la categoría «Sin patrón claro» cuando un umbral experimental de error de reconstrucción indique mal ajuste. Se conservan los pesos de todos los componentes, incluso en jugadas mixtas. La salida permite localizar cada jugada y revisar la evidencia breve de cada componente. Esta etapa termina en la clasificación y su validación; el perfil de equipos y el análisis de entrenadores quedan para trabajos posteriores.

## User Stories

1. Como analista, quiero clasificar todas las jugadas elegibles del corpus preparado para disponer de un resultado completo y auditable.
2. Como analista, quiero que la nueva regla modifique el corpus preparado en su ubicación actual para aprovechar la exportación ya realizada.
3. Como analista, quiero aplicar la misma regla a partidos preparados en el futuro para que el criterio de clasificación sea reproducible.
4. Como analista, quiero configurar la división entre balón parado y juego abierto para poder ejecutar y registrar esa decisión explícitamente.
5. Como analista, quiero que córners, tiros libres y saques de banda alimenten el modelo de balón parado para compararlos dentro de una misma familia de reanudaciones.
6. Como analista, quiero que el juego regular, los contraataques, las acciones del portero, los saques de meta y los saques iniciales alimenten el modelo de juego abierto según la división acordada.
7. Como analista, quiero detectar cambios entre ambas categorías dentro de una jugada para evitar que una fase larga oculte otra clasificable.
8. Como analista, quiero permitir un corte entre fases aunque la posesión continúe para clasificar su desarrollo con los modelos correspondientes.
9. Como analista, quiero dividir una jugada solo cuando cada tramo resultante tenga al menos tres pases completados del protagonista para no crear jugadas demasiado cortas.
10. Como analista, quiero conservar una sola jugada cuando algún tramo no alcance el mínimo para mantener la cobertura del corpus original.
11. Como analista, quiero que una jugada conservada use el modelo de la fase con más acciones de juego del protagonista para representar su desarrollo predominante.
12. Como analista, quiero admitir más de dos partes si existen varios cambios válidos entre categorías para que cada archivo clasificable pertenezca a una sola categoría.
13. Como analista, quiero conservar los IDs de las jugadas que no se dividen para mantener sus referencias existentes.
14. Como analista, quiero que las partes de una jugada dividida reciban IDs únicos derivados del ID original para reconocer su procedencia.
15. Como analista, quiero localizar en el manifiesto los eventos, límites, pases y razón de cada corte para auditar la modificación.
16. Como analista, quiero conservar los valores y columnas originales de los eventos para reconstruir cada jugada sin pérdida de información.
17. Como analista, quiero que cada evento exportado figure una sola vez después de la actualización para evitar sesgos por duplicación.
18. Como analista, quiero clasificar juego abierto y balón parado con modelos NMF distintos para que sus componentes respondan a estructuras diferentes.
19. Como analista, quiero representar acciones y desplazamientos observados sin predefinir estilos tácticos para descubrir regularidades en los datos.
20. Como analista, quiero usar n-gramas de 2 y 3 acciones consecutivas para conservar el orden local de las jugadas.
21. Como analista, quiero comparar n-gramas situados al inicio, en medio y al final para saber si conservar parte del orden global mejora los componentes.
22. Como analista, quiero probar una señal de ritmo si su costo es razonable para valorar si el tiempo añade información útil.
23. Como analista, quiero comparar las alturas y zonas del campo bajo una orientación de ataque coherente para que la dirección física del equipo no invierta el significado de los movimientos.
24. Como analista, quiero tratar como equivalentes las secuencias reflejadas entre bandas para reconocer el mismo patrón en ambos lados del campo.
25. Como analista, quiero conservar la banda original como metadato para recuperar la procedencia espacial de cada jugada.
26. Como analista, quiero distinguir las acciones propias de las intervenciones rivales absorbidas para no atribuir al protagonista una acción contraria.
27. Como analista, quiero distinguir interrupciones y recuperaciones rivales en los tokens para representar su papel en la secuencia.
28. Como analista, quiero comparar modelos con y sin esos tokens para comprobar que las intervenciones no dominen los componentes.
29. Como analista, quiero que eventos vinculados de un mismo lance no se cuenten como decisiones independientes para evitar patrones creados por el formato de registro.
30. Como analista, quiero que la representación reconozca valores faltantes sin inventar desplazamientos o tiempos iguales a cero.
31. Como analista, quiero una clase principal por jugada para poder consultar y agrupar el resultado de forma sencilla.
32. Como analista, quiero conservar todos los pesos de NMF para reconocer jugadas que mezclan varios componentes.
33. Como analista, quiero una categoría «Sin patrón claro» basada en el error de reconstrucción para no forzar una etiqueta sin respaldo.
34. Como analista, quiero que los umbrales de mal ajuste se definan con experimentos y se registren para poder reproducir la decisión.
35. Como analista, quiero conocer los n-gramas principales y ejemplos de cada componente para juzgar si aporta información comprensible.
36. Como analista, quiero revisar jugadas representativas y de peor encaje para detectar componentes definidos por longitud, interrupciones o artefactos.
37. Como analista, quiero comparar resultados entre muestras, partidos y semillas para valorar su estabilidad.
38. Como analista, quiero limitar el costo de las variantes mediante un piloto medido antes de procesar el corpus completo.
39. Como analista, quiero una tabla enlazada por `play_id` con modelo, clase, pesos y error para consultar los resultados de todas las jugadas.
40. Como analista, quiero conservar modelos, vocabularios y parámetros para reproducir o aplicar la clasificación a nuevas jugadas.
41. Como analista, quiero que un resultado NMF fallido se documente como tal para no utilizar grupos débiles como categorías válidas.

## Implementation Decisions

### Alcance y preparación

- El corpus existente, con 163,909 jugadas elegibles procedentes de 1,808 partidos, es el punto de partida. Esta entrega no vuelve a segmentar todos los eventos originales para actualizarlo. Una transformación en la misma ubicación lee las jugadas ya exportadas y modifica únicamente las jugadas, manifiestos y totales afectados por la nueva regla. La regla de división también se incorpora al flujo que prepare partidos futuros; ambos caminos comparten la misma lógica.
- La nueva opción de división se activa explícitamente y su valor queda registrado junto a los parámetros de preparación. La operación sobre el corpus existente debe ser repetible: ejecutarla dos veces con los mismos parámetros no crea nuevos cortes ni cambia IDs o membresía de eventos.
- Balón parado comprende `From Corner`, `From Free Kick` y `From Throw In`. Juego abierto comprende `Regular Play`, `From Counter`, `From Goal Kick`, `From Kick Off` y `From Keeper`. Los valores desconocidos se auditan y reciben una política determinista que preserva la cobertura; no se descartan silenciosamente.
- Un cambio entre categorías puede ocurrir dentro de una misma posesión base y aun así activa la evaluación del corte. Los cambios de `play_pattern` dentro de una misma categoría no lo activan. Se construyen todos los tramos candidatos ante cambios entre categorías; solo se publican por separado si **todos** contienen al menos tres pases completados del protagonista. En otro caso se conserva la jugada original. Un empate en cantidad de acciones para seleccionar el modelo de una jugada conservada se resuelve de forma determinista y documentada.
- Al dividir, los IDs de los archivos nuevos derivan del `play_id` anterior mediante sufijos ordenados; los IDs de jugadas no divididas permanecen iguales. Cada manifiesto registra ID de origen, ordinal del tramo, categoría, eventos inicial y final, eventos contenidos, pases completados, motivo y parámetros del corte. Se actualizan los totales del partido y del corpus.
- Los eventos de la jugada original se reparten exactamente una vez entre sus partes; la operación no cambia sus columnas, valores anidados ni orden relativo. La suma de eventos exportados y no exportados sigue conciliando con los totales de entrada. La publicación por partido evita dejar un manifiesto o archivos parcialmente actualizados si falla la transformación.

### Representación y NMF

- Hay un vocabulario, vectorizador y NMF independientes por categoría. Los identificadores de componentes incluyen la categoría para evitar confundir un componente de balón parado con otro de juego abierto. `play_pattern` decide la ruta y queda como contexto; sus identificadores no se utilizan como variables numéricas de similitud.
- La tokenización determinista extrae acciones del protagonista, tipo y dirección de desplazamiento, y zonas de origen o destino cuando existan. Trata eventos auxiliares vinculados como contexto del mismo lance para no contarlos dos veces. Los umbrales de distancia, la cuadrícula espacial y los pesos entre familias de tokens son parámetros sujetos a comparación, no categorías tácticas fijadas de antemano.
- Se extraen n-gramas consecutivos de longitud 2 y 3. El modelo base conserva el orden local. Se compara una variante que etiqueta n-gramas por inicio, medio y final según tercios del número de acciones ordenadas; no se permite que un n-grama cruce un límite real de jugada. Se documenta cómo se tratan tramos con menos acciones que la longitud del n-grama.
- Una variante temporal usa señales de tiempo entre acciones cuando los datos y el costo del piloto lo permitan. Los eventos vinculados que comparten timestamp no producen intervalos ficticios; las duraciones faltantes no se sustituyen por cero. La variante base sin tiempo sigue siendo evaluable.
- Las coordenadas de pases y tiros locales aparecen orientadas hacia `x=120` para el equipo que ejecuta la acción, incluso cuando `attacking_direction` indica el sentido físico contrario. Esta observación se verifica antes de tokenizar; no se invierte `x` solo por el valor de ese campo. Para una trayectoria común desde el protagonista, las coordenadas de una intervención rival se convierten al marco del protagonista. La equivalencia entre bandas se aplica de forma independiente y se conserva la banda original como metadato. Los archivos de eventos no se alteran para realizar estas transformaciones analíticas.
- Las intervenciones rivales conservadas se representan con tokens específicos para interrupción y recuperación. Se prueba una ablación sin esos tokens para detectar si dominan indebidamente los componentes.
- La matriz de jugadas por n-gramas permanece dispersa y no negativa. Se prueban TF-IDF y alternativas de conteo o frecuencia normalizada según los resultados. Los vocabularios se ajustan solo con la partición de entrenamiento. El número de componentes y el peso de las familias de características se eligen por comparación, con los valores de la idea original como puntos de partida, no como resultados preaprobados.
- Se registra una convención de escala para `W` y `H` antes de comparar pesos de componentes. Para cada jugada se conserva el vector completo de contribuciones, una etiqueta principal derivada de su componente dominante y un error de reconstrucción comparable dentro del modelo. Si el error supera un umbral definido experimentalmente, la etiqueta publicada es «Sin patrón claro»; los pesos siguen disponibles. Las contribuciones normalizadas no se presentan como probabilidades calibradas.

### Evaluación y entrega

- El piloto utiliza una muestra diversa de partidos y equipos y mide tiempo, memoria y tamaño de la matriz antes de ejecutar todas las variantes sobre el corpus. La comparación temporal se hace si este piloto confirma que su costo es razonable. No se construyen matrices completas de distancias entre pares de jugadas.
- La selección del modelo comprueba reconstrucción, estabilidad entre semillas y muestras, coherencia de n-gramas y jugadas representativas, casos de peor encaje, concentración de componentes y sensibilidad a tokenización. Examina explícitamente si la longitud de la jugada influye sin convertirse en el factor principal. Se revisan las clases por tipo de reanudación y la ablación de tokens rivales.
- La validación fuera de muestra separa por partido y, cuando sea viable, por temporada. Vocabulario, escalas y NMF se ajustan en entrenamiento y se aplican sin reajuste a evaluación. La revisión de ejemplos sirve para aceptar o rechazar la utilidad de los componentes; no se generan perfiles de equipos ni nombres tácticos definitivos en esta etapa.
- Un proceso por lotes independiente de sesiones interactivas consume el corpus preparado. Produce una tabla con una fila por jugada publicada, su `play_id`, categoría/modelo, clase principal o «Sin patrón claro», pesos, error y procedencia. Persiste además los dos modelos, vocabularios, parámetros, semillas, diagnósticos y una ficha breve por componente con IDs, n-gramas destacados y ejemplos. La cobertura de la tabla concilia con los manifiestos vigentes.
- Si NMF no genera componentes suficientemente útiles tras los experimentos previstos, se guardan la evidencia y los diagnósticos del fallo. Sus etiquetas exploratorias no se publican como una clasificación final válida; la elección de otro método queda para una especificación posterior.

## Testing Decisions

- El punto de prueba automatizada es la **salida final de clasificación** tras ejecutar el flujo completo sobre un corpus pequeño y controlado. Debe observarse una fila por jugada final, con categoría/modelo correcto, clase principal o «Sin patrón claro», pesos y enlace al ID publicado. No se crean pruebas que dependan de funciones privadas, números arbitrarios de componentes ni de la numeración interna de etiquetas.
- El caso integrado incluye una jugada que se divide con tres pases completados en cada parte, otra que permanece unida porque algún tramo no cumple, una intervención rival, una reflexión entre bandas y ausencia de datos temporales. Las afirmaciones se hacen sobre la clasificación resultante y la conciliación observable de IDs y eventos, no sobre pasos internos.
- La actualización del corpus incorpora comprobaciones de integridad durante la ejecución: cada evento previamente exportado queda en exactamente una jugada vigente; IDs, archivos, manifiestos y totales concilian; una actualización fallida no se presenta como completa. Son condiciones de seguridad del proceso, no un segundo punto de prueba independiente.
- Se comprueba reproducibilidad de la salida con mismos datos, parámetros y semilla. Las comparaciones entre ajustes emparejan componentes por su contenido, no por el número arbitrario de etiqueta. La utilidad táctica se revisa mediante los ejemplos y diagnósticos publicados, no mediante una aserción fija sobre el nombre de un grupo.
- Como antecedente, el proyecto ya tiene pruebas de integración de exportación que comprueban lectura de jugadas, manifiestos, pase mínimo y conciliación de eventos. La nueva verificación se sitúa en la salida de clasificación solicitada por el usuario.

## Out of Scope

- Perfiles de equipos, descripción del estilo de juego, análisis de entrenadores o atribución causal.
- Complemento 360 y cualquier requisito de cobertura 360 para entrar a la clasificación.
- Taxonomía táctica predefinida o nombres definitivos de estilos para los componentes.
- Clasificar secuencias originales descartadas por no alcanzar tres pases completos del protagonista.
- Reconstruir el corpus actual desde los JSON de eventos originales como parte de la actualización solicitada.
- Sustituir NMF por otro algoritmo dentro de esta entrega si el experimento no produce grupos útiles.
- Publicar como clasificación válida los resultados de una corrida cuyo diagnóstico haya rechazado sus componentes.

## Further Notes

- El corpus local actual registra 1,808 partidos procesados, ningún partido fallido y 163,909 jugadas elegibles. Después de dividir jugadas que cumplan el mínimo en todos sus tramos, el número final de archivos puede aumentar; 163,909 es el conteo de partida, no una expectativa rígida del nuevo total.
- En una muestra de 18 partidos, 78 jugadas comenzaron en una de las tres categorías de balón parado y después pasaron a `Regular Play`. Solo 7 de esas 78 tenían al menos tres pases completos del protagonista a cada lado del primer cambio. Este dato motivó conservar la jugada completa si algún tramo es corto. Algunos cambios ocurren dentro de una misma posesión base; el corte acordado describe una fase para clasificación, aunque el juego haya continuado.
- La orientación de coordenadas se infirió de tiros y pases de ambos equipos en 12 partidos locales. El proceso debe verificar la convención del corpus antes de transformarlo; la equivalencia entre bandas es una decisión separada.
- En una lectura de muestra de 507 archivos de jugada, el parseo tomó aproximadamente tres segundos. La extrapolación simple para una lectura del corpus completo ronda 16 minutos en este entorno y no es una garantía de ejecución. Guardar 16 pesos `float32` por cada una de las 163,909 jugadas iniciales ocupa aproximadamente 10.5 MB sin comprimir; leer el corpus, construir la matriz y entrenar NMF será más costoso que persistir esos pesos.
- La definición de «jugada» del glosario deberá reflejar el nuevo corte opcional por categoría cuando se implemente. La especificación anterior de segmentación no determina el algoritmo de clasificación de esta entrega.
