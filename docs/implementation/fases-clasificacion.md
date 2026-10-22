# Corte opcional de fases para clasificación (#12)

`procesar_corpus(..., dividir_fases_clasificables=True)` y
`exportar_jugadas_partido(..., dividir_fases_clasificables=True)` aplican la
misma regla a las jugadas elegibles. La opción está desactivada por defecto.
`Data_Preparation.IPYNB` la expone en su celda de configuración.

Balón parado comprende `From Corner`, `From Free Kick` y `From Throw In`.
Juego abierto comprende `Regular Play`, `From Counter`, `From Goal Kick`,
`From Kick Off` y `From Keeper`. Solo las filas del protagonista señalan
cambios de fase; las intervenciones rivales absorbidas conservan la fase
vigente. Un cambio entre dos patrones de la misma categoría no corta. El
cambio puede cortar una posesión base continua.

Los patrones desconocidos, ausentes o sin nombre se auditan con el ID del
evento y la categoría asignada. Heredan la categoría anterior; si están al
inicio de una jugada usan la primera categoría conocida del protagonista;
si no existe ninguna, se asignan a juego abierto. No provocan cortes por sí
mismos. Los tramos se forman ante cada cambio de categoría y solo se publican
por separado cuando **todos** tienen al menos tres pases completados
del protagonista. Si alguno no cumple, la jugada conserva su ID y archivo.
La categoría de esa jugada es la que suma más acciones relevantes propias;
en empate gana la primera fase observada. Se usa el mismo criterio de acciones
relevantes que para las intervenciones breves: los registros contextuales
vinculados y los contactos vinculados de un mismo lance no se duplican.

Las partes reciben sufijos `_f01`, `_f02`, etc. en `play_id` y nombre de archivo.
Cada entrada de manifiesto con la opción activa registra `play_id_origen`,
`ordinal_tramo`, `categoria_clasificacion`, `eventos_ids`, `pases_completados`,
`motivo_corte` y los eventos a ambos lados de cada corte. El manifiesto registra
la opción, política para patrones desconocidos, auditoría de esos patrones y
totales de jugadas de origen divididas y cortes. Los archivos de eventos
mantienen exactamente sus columnas, orden y valores originales.

La aplicación a los partidos **ya preparados** y la actualización de sus
manifiestos y resumen en la ubicación actual corresponden al issue #13.
