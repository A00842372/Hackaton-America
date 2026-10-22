# Issue tracker: GitHub

Las especificaciones y tareas viven en GitHub Issues de `A00842372/Hackaton-America`.

Usa GitHub CLI (`gh`) con `--repo A00842372/Hackaton-America` para las operaciones de issues y etiquetas. Comprueba la sesión mediante `gh auth status` cuando sea necesario. Las credenciales permanecen fuera del repositorio.

## Operaciones

- Consultar un issue: `gh issue view NUMERO --repo A00842372/Hackaton-America --comments`.
- Buscar antes de crear, para evitar duplicados: `gh issue list --repo A00842372/Hackaton-America --state all --search CONSULTA`.
- Crear una especificación: `gh issue create --repo A00842372/Hackaton-America --title TITULO --body-file ARCHIVO --label ready-for-agent`.
- Aplicar o retirar etiquetas: `gh issue edit NUMERO --repo A00842372/Hackaton-America --add-label ETIQUETA` o `--remove-label ETIQUETA`.
- Publicar comentarios autorizados: `gh issue comment NUMERO --repo A00842372/Hackaton-America --body-file ARCHIVO`.
- Cerrar un issue autorizado: `gh issue close NUMERO --repo A00842372/Hackaton-America`.

Los cuerpos multilínea se pasan mediante `--body-file`. Al publicar una especificación local, registra la URL del issue en el documento para mantener la trazabilidad. Al actualizarla, sincroniza el mismo issue.

Cuando una habilidad pide publicar en el tracker, crea o actualiza un issue de este repositorio. Cuando pide leer una tarea, consulta el issue y sus comentarios.

## Pull requests

PRs como superficie de recepción de solicitudes: no. Una petición explícita de trabajar sobre un PR sí se atiende. Para referencias numéricas ambiguas, identifica si corresponden a un issue o a un PR antes de operar.
