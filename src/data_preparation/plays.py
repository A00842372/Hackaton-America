from typing import Any, Dict, List, Optional, Tuple, Union
import pandas as pd
from src.data_preparation.interventions import Tramo, unir_intervenciones_cortas

STOP_ADMINISTRATIVE_EVENT_TYPES = {
    "Half Start", "Half End", "Injury Stoppage", "Substitution",
    "Referee Ball-Drop",
    "Player Off", "Player On", "Bad Behaviour",
}
ADMINISTRATIVE_EVENT_TYPES = STOP_ADMINISTRATIVE_EVENT_TYPES | {
    "Starting XI", "Camera On", "Camera off", "Tactical Shift",
}
STOP_EVENT_TYPES = STOP_ADMINISTRATIVE_EVENT_TYPES | {"Offside"}
RESTART_PASS_TYPES = {"Throw-in", "Goal Kick", "Corner", "Free Kick", "Kick Off"}
STOP_SHOT_OUTCOMES = {"Goal", "Off T", "Wayward"}
FOUL_TYPES = {"Foul Committed", "Foul Won"}


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
    max_segundos_intervencion: float = 3,
    max_acciones_intervencion: int = 3,
) -> Tuple[List[pd.DataFrame], List[Dict[str, str]]]:
    """Delimita candidatas por control, salidas, detenciones y reanudaciones.

    El poseedor de cada evento es ``possession_team``; una acción defensiva de
    otro ``team`` no cambia el protagonista. Un cambio de ``possession`` del
    mismo equipo no corta una jugada si el balón siguió en juego. Las
    intervenciones rivales se absorben cuando cumplen ambos límites sin cruzar
    un cierre de juego.
    """
    if df_eventos.empty:
        return [], []

    eventos_no_exportados: List[Dict[str, str]] = []
    tramos: List[Tramo] = []

    columnas_orden = [col for col in ("match_id", "period", "index") if col in df_eventos]
    ordenados = df_eventos.sort_values(columnas_orden, kind="stable").reset_index(drop=True)
    bloque_actual: List[int] = []
    clave_actual: Optional[Tuple[Any, Any, Any]] = None
    cierre_pendiente = False
    ultimo_tipo: Optional[str] = None
    ultimo_timestamp: Any = None

    def cerrar_bloque(barrera: bool) -> None:
        if bloque_actual:
            tramos.append(Tramo(
                eventos=ordenados.iloc[bloque_actual].reset_index(drop=True),
                barrera_despues=barrera,
                protagonista=clave_actual[2] if clave_actual else None,
            ))
            bloque_actual.clear()

    for posicion, row in ordenados.iterrows():
        tipo = _extraer_nombre_entidad(row.get("type"))
        if tipo in ADMINISTRATIVE_EVENT_TYPES:
            eventos_no_exportados.append({
                "id": str(row.get("id")), "motivo": "evento_administrativo"
            })
            if tipo in STOP_EVENT_TYPES:
                cerrar_bloque(True)
                cierre_pendiente = False
                clave_actual = None
            continue

        poseedor = _extraer_nombre_entidad(row.get("possession_team"))
        if poseedor is None:
            poseedor = _extraer_nombre_entidad(row.get("team"))
        clave = (row.get("match_id"), row.get("period"), poseedor)
        pase = row.get("pass") if isinstance(row.get("pass"), dict) else {}
        tiro = row.get("shot") if isinstance(row.get("shot"), dict) else {}
        reinicio = (
            tipo == "Pass" and _extraer_nombre_entidad(pase.get("type")) in RESTART_PASS_TYPES
        ) or (
            tipo == "Shot" and _extraer_nombre_entidad(tiro.get("type")) in {"Free Kick", "Penalty"}
        )
        # StatsBomb registra la falta cometida y la recibida como dos filas del
        # mismo lance; el cierre se aplica después de conservar ambas.
        pareja_falta = (
            cierre_pendiente and tipo in FOUL_TYPES and ultimo_tipo in FOUL_TYPES
            and row.get("timestamp") == ultimo_timestamp
        )

        if bloque_actual and (clave != clave_actual or reinicio or (cierre_pendiente and not pareja_falta)):
            barrera = clave[:2] != clave_actual[:2] or reinicio or cierre_pendiente
            cerrar_bloque(barrera)

        bloque_actual.append(posicion)
        clave_actual = clave
        ultimo_tipo = tipo
        ultimo_timestamp = row.get("timestamp")

        falta = row.get("foul_committed") if tipo == "Foul Committed" else row.get("foul_won")
        ventaja = isinstance(falta, dict) and falta.get("advantage") is True
        cierre_pendiente = (
            (tipo in FOUL_TYPES and not ventaja)
            or tipo in STOP_EVENT_TYPES
            or row.get("out") is True
            or (tipo == "Pass" and _extraer_nombre_entidad(pase.get("outcome")) == "Out")
            or (tipo == "Shot" and _extraer_nombre_entidad(tiro.get("outcome")) in STOP_SHOT_OUTCOMES)
        )

    cerrar_bloque(True)

    return unir_intervenciones_cortas(
        tramos, max_segundos_intervencion, max_acciones_intervencion,
    ), eventos_no_exportados


# Alias de compatibilidad
extraer_secuencias_posesion = extraer_secuencias_candidatas
