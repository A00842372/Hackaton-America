from typing import Any, Dict, List, Optional, Tuple, Union
import pandas as pd

ADMINISTRATIVE_EVENT_TYPES = {
    "Starting XI",
    "Half Start",
    "Half End",
    "Camera On",
    "Camera off",
    "Referee Ball-Drop",
}


def _extraer_nombre_entidad(val: Any) -> Optional[str]:
    """Extrae el nombre de una entidad StatsBomb que puede ser string o dict con clave 'name'."""
    if isinstance(val, dict):
        return val.get("name")
    if pd.isna(val):
        return None
    return str(val)


def obtener_equipo_protagonista(df: pd.DataFrame) -> Optional[str]:
    """Obtiene el nombre del equipo protagonista de una secuencia a partir de possession_team o team."""
    if df.empty:
        return None

    if "possession_team" in df.columns:
        for val in df["possession_team"]:
            nombre = _extraer_nombre_entidad(val)
            if nombre:
                return nombre

    if "team" in df.columns:
        for val in df["team"]:
            nombre = _extraer_nombre_entidad(val)
            if nombre:
                return nombre

    return None


def _es_pase_completado_protagonista(row: Union[pd.Series, Dict[str, Any]], equipo_protagonista: str) -> bool:
    """Comprueba si el registro es un pase completado por el equipo protagonista.
    
    StatsBomb reconoce un pase completado por ausencia de pass.outcome.
    """
    tipo = _extraer_nombre_entidad(row.get("type"))
    if tipo != "Pass":
        return False

    equipo = _extraer_nombre_entidad(row.get("team"))
    if equipo != equipo_protagonista:
        return False

    pass_data = row.get("pass")
    if isinstance(pass_data, dict):
        outcome = pass_data.get("outcome")
        if outcome is not None and not (isinstance(outcome, float) and pd.isna(outcome)):
            return False

    return True


def contar_pases_completados(df_secuencia: pd.DataFrame, equipo_protagonista: str) -> int:
    """Cuenta el número de pases completados por el equipo protagonista en la secuencia."""
    if df_secuencia.empty:
        return 0

    pases = 0
    for _, row in df_secuencia.iterrows():
        if _es_pase_completado_protagonista(row, equipo_protagonista):
            pases += 1
    return pases


def es_jugada_elegible(df_secuencia: pd.DataFrame, min_pases: int = 3) -> Tuple[bool, int]:
    """Determina si una secuencia cumple con el mínimo de pases completados del protagonista.
    
    Retorna (es_elegible, conteo_pases).
    """
    if df_secuencia.empty:
        return False, 0

    equipo_protagonista = obtener_equipo_protagonista(df_secuencia)
    if not equipo_protagonista:
        return False, 0

    pases = contar_pases_completados(df_secuencia, equipo_protagonista)
    return pases >= min_pases, pases


def extraer_secuencias_candidatas(
    df_eventos: pd.DataFrame,
) -> Tuple[List[pd.DataFrame], List[Dict[str, str]]]:
    """Extrae secuencias candidatas simples agrupadas por bloques contiguos de posesión base.
    
    Agrupa cronológicamente respetando transiciones de periodo y posesión. Los eventos
    administrativos que no forman jugadas se excluyen y se registran en la lista de
    eventos no exportados con su motivo.
    """
    if df_eventos.empty:
        return [], []

    eventos_no_exportados: List[Dict[str, str]] = []
    secuencias: List[pd.DataFrame] = []

    filas_jugables = []
    for _, row in df_eventos.iterrows():
        tipo = _extraer_nombre_entidad(row.get("type"))
        if tipo in ADMINISTRATIVE_EVENT_TYPES:
            eventos_no_exportados.append({
                "id": str(row.get("id")),
                "motivo": "evento_administrativo",
            })
        else:
            filas_jugables.append(row)

    if not filas_jugables:
        return [], eventos_no_exportados

    df_jugables = pd.DataFrame(filas_jugables).reset_index(drop=True)

    # Agrupar por bloques contiguos de (period, possession) para evitar unir apariciones no contiguas
    bloque_actual: List[Dict[str, Any]] = []
    clave_actual: Optional[Tuple[Any, Any]] = None

    for _, row in df_jugables.iterrows():
        periodo = row.get("period", 1)
        posesion = row.get("possession", 0)
        clave = (periodo, posesion)

        if clave_actual is None:
            clave_actual = clave
            bloque_actual.append(row.to_dict())
        elif clave == clave_actual:
            bloque_actual.append(row.to_dict())
        else:
            if bloque_actual:
                secuencias.append(pd.DataFrame(bloque_actual).reset_index(drop=True))
            clave_actual = clave
            bloque_actual = [row.to_dict()]

    if bloque_actual:
        secuencias.append(pd.DataFrame(bloque_actual).reset_index(drop=True))

    return secuencias, eventos_no_exportados


# Alias de compatibilidad
extraer_secuencias_posesion = extraer_secuencias_candidatas
