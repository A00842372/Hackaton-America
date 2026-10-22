# Clasificación NMF exploratoria por categoría (#14)

El piloto consume **jugadas ya preparadas** con `dividir_fases_clasificables=true`.
Se ejecuta con:

```bash
python -m src.classification.exploratory_nmf \
  datos_preparados_corpus datos_clasificacion_exploratoria \
  --max-partidos 12 --componentes 4 --semilla 42
```

La muestra son los primeros partidos en orden de ruta. Una semilla fija reserva
el 20 % de los partidos para evaluación; `--reservar-partido ID` permite fijarlos
explícitamente. Cada categoría ajusta por separado su vocabulario de n-gramas,
vectorizador de conteos, transformación TF-IDF L2 y NMF **solo con jugadas de
entrenamiento**. Los partidos reservados se transforman con esos objetos, sin
reajuste. El número efectivo de componentes se limita a las dimensiones de la
matriz de entrenamiento de cada categoría.

La tokenización conserva el orden de las acciones. Cada token indica actor,
tipo de acción y, si hay coordenadas de origen y destino, dirección de avance
en `x` y tramo de distancia observada. Si falta una coordenada, el token solo
indica actor y tipo: no se inventa un desplazamiento cero. `Ball Receipt*` y los
registros administrativos no forman decisiones adicionales. `Dribbled Past` y
`Foul Won` vinculados a otro evento tampoco se cuentan por separado. Sobre los
tokens de acción se cuentan n-gramas consecutivos de longitud 2 y 3. El patrón
de inicio (`play_pattern`) decide la categoría en la preparación y no entra al
vocabulario numérico.

`resultados.json` incluye una fila por jugada, `play_id`, categoría, partición,
componente dominante, todos los pesos NMF y error de reconstrucción L2, más la
ruta de su manifiesto y archivo. También registra parámetros, semilla, partidos
de cada partición y vocabulario por categoría. Los archivos `modelo_*.joblib`
guardan vectorizador, TF-IDF y NMF para aplicar la misma transformación a otras
jugadas. La salida se crea en un directorio nuevo para conservar cada corrida.
La cobertura del vocabulario se calcula con los n-gramas observados. Si ninguno
estaba en entrenamiento, la jugada conserva sus pesos nulos y se marca
`sin_rasgos_conocidos`; componente y error quedan nulos para evitar presentar
un ajuste perfecto sin evidencia. Con cobertura parcial, el error describe solo
la parte conocida por el modelo.

El estado se marca **exploratorio**. Los pesos son los coeficientes `W` crudos
de NMF sobre TF-IDF L2 y no son probabilidades. La selección de componentes,
los umbrales de «Sin patrón claro», las zonas, el ritmo y la evaluación táctica
corresponden a los tickets siguientes; estos IDs de componente no son nombres
tácticos validados.
