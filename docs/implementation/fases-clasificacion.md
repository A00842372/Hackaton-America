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

## Corpus ya preparado (#13)

Para actualizar las jugadas existentes en su ubicación actual, sin volver a
leer los eventos originales:

```bash
python -m src.data_preparation.prepared_corpus datos_preparados_corpus --trabajadores 6
```

El proceso valida cada partido antes de publicar una carpeta completa. Los
archivos de jugadas que no se dividen conservan nombre, ID y contenido; los
que sí se dividen se sustituyen por archivos con sufijos `_f01`, `_f02`, etc.
El manifiesto incorpora la categoría de todas las jugadas vigentes y los
totales de cortes. `resumen_corpus.json` registra el resultado en
`actualizacion_fases`, incluidos partidos revisados, actualizados, sin cambios,
fallidos y errores. Mientras se ejecuta o si hay algún error, el estado global
es `incompleto`. Ejecutar el comando de nuevo valida las carpetas ya
actualizadas sin crear cortes ni alterar sus archivos.
