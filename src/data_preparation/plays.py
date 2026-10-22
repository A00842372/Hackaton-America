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


def _extraer_nombre(val: Any) -> Optional[str]:
    """Extrae el nombre de un campo que puede ser string o dict con clave 'name'."""
    if isinstance(val, dict):
        return val.get("name")
    if pd.isna(val):
        return None
    return str(val)


def _es_pase_completado_protagonista(row: Union[pd.Series, Dict[str, Any]], equipo_protagonista: str) -> bool:
    """Comprueba si el registro es un pase completado por el equipo protagonista.
    
    StatsBomb reconoce un pase completado por ausencia de pass.outcome.
    """
    tipo = _extraer_nombre(row.get("type"))
    if tipo != "Pass":
        return False

    equipo = _extraer_nombre(row.get("team"))
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

    # Determinar el equipo protagonista a partir de possession_team
    equipo_protagonista = None
    if "possession_team" in df_secuencia.columns:
        for val in df_secuencia["possession_team"]:
            nombre = _extraer_nombre(val)
            if nombre:
                equipo_protagonista = nombre
                break

    if not equipo_protagonista and "team" in df_secuencia.columns:
        equipo_protagonista = _extraer_nombre(df_secuencia["team"].iloc[0])

    if not equipo_protagonista:
        return False, 0

    pases = contar_pases_completados(df_secuencia, equipo_protagonista)
    return pases >= min_pases, pases


def extraer_secuencias_posesion(
    df_eventos: pd.DataFrame,
) -> Tuple[List[pd.DataFrame], List[Dict[str, str]]]:
    """Extrae secuencias simples agrupadas por posesión base (period y possession).
    
    Los eventos administrativos que no forman jugadas se excluyen y se registran
    en la lista de eventos no exportados con su motivo.
    """
    if df_eventos.empty:
        return [], []

    eventos_no_exportados: List[Dict[str, str]] = []
    secuencias: List[pd.DataFrame] = []

    # Filtrar eventos administrativos
    filas_jugables = []
    for _, row in df_eventos.iterrows():
        tipo = _extraer_nombre(row.get("type"))
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

    # Agrupar preservando orden cronológico por (period, possession)
    # Para respetar cambios de periodo y secuencia de posesiones:
    claves_vistas = []
    grupos: Dict[Tuple[Any, Any], List[Dict[str, Any]]] = {}

    for _, row in df_jugables.iterrows():
        periodo = row.get("period", 1)
        posesion = row.get("possession", 0)
        clave = (periodo, posesion)
        if clave not in grupos:
            grupos[clave] = []
            claves_vistas.append(clave)
        grupos[clave].append(row.to_dict())

    for clave in claves_vistas:
        df_sec = pd.DataFrame(grupos[clave]).reset_index(drop=True)
        secuencias.append(df_sec)

    return secuencias, eventos_no_exportados
