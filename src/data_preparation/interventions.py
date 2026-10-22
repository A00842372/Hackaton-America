"""Unión determinista de intervenciones rivales fugaces entre tramos de juego."""

from decimal import Decimal, InvalidOperation
from dataclasses import dataclass
from typing import Optional

import pandas as pd


CONTEXTUAL_TYPES = {"Ball Receipt*", "Dribbled Past", "Pressure"}
CONTACT_TYPES = {"Duel", "Ball Recovery", "Interception", "Dispossessed"}


@dataclass
class Tramo:
    eventos: pd.DataFrame
    barrera_despues: bool
    protagonista: Optional[str]


def _nombre(valor: object) -> Optional[str]:
    if isinstance(valor, dict):
        return valor.get("name")
    return valor if isinstance(valor, str) else None


def _segundo(fila: pd.Series) -> Optional[Decimal]:
    marca = fila.get("timestamp")
    if not isinstance(marca, str):
        return None
    try:
        horas, minutos, segundos = marca.split(":")
        return Decimal(horas) * 3600 + Decimal(minutos) * 60 + Decimal(segundos)
    except (ValueError, InvalidOperation):
        return None


def _fin(fila: pd.Series) -> Optional[Decimal]:
    inicio = _segundo(fila)
    if inicio is None:
        return None
    duracion = fila.get("duration")
    if duracion is None or pd.isna(duracion):
        return inicio
    try:
        segundos = Decimal(str(duracion))
    except InvalidOperation:
        return None
    if not segundos.is_finite() or segundos < 0:
        return None
    return inicio + segundos


def _filas_equipo(tramo: pd.DataFrame, equipo: str) -> list[pd.Series]:
    return [fila for _, fila in tramo.iterrows() if _nombre(fila.get("team")) == equipo]


def _relacionados(a: pd.Series, b: pd.Series) -> bool:
    a_id, b_id = str(a.get("id")), str(b.get("id"))
    a_rel = a.get("related_events")
    b_rel = b.get("related_events")
    a_rel = set(a_rel) if isinstance(a_rel, list) else set()
    b_rel = set(b_rel) if isinstance(b_rel, list) else set()
    return b_id in a_rel or a_id in b_rel or bool(a_rel & b_rel)


def _acciones_relevantes(tramo: pd.DataFrame, equipo: str, limite: int) -> int:
    """Cuenta acciones de balón del poseedor, sin duplicar registros del mismo lance.

    Pase, conducción y tiro son acciones distintas aunque estén relacionados.
    Duelos/recuperaciones simultáneos relacionados forman un lance. Recepción,
    presión y dribleado son registros contextuales: cuentan solo si no están
    relacionados con una acción principal del tramo.
    """
    filas = _filas_equipo(tramo, equipo)
    principales = [fila for fila in filas if _nombre(fila.get("type")) not in CONTEXTUAL_TYPES]
    contextuales = [fila for fila in filas if _nombre(fila.get("type")) in CONTEXTUAL_TYPES]
    conteo = 0
    vistas: list[pd.Series] = []
    for fila in principales:
        tipo = _nombre(fila.get("type"))
        mismo_lance = tipo in CONTACT_TYPES and any(
            _nombre(otra.get("type")) in CONTACT_TYPES
            and fila.get("timestamp") == otra.get("timestamp")
            and _relacionados(fila, otra)
            for otra in vistas
        )
        if not mismo_lance:
            conteo += 1
            if conteo > limite:
                return conteo
        vistas.append(fila)

    vistas_contextuales: list[pd.Series] = []
    for fila in contextuales:
        if any(_relacionados(fila, otra) for otra in principales):
            continue
        mismo_lance = any(
            fila.get("timestamp") == otra.get("timestamp") and _relacionados(fila, otra)
            for otra in vistas_contextuales
        )
        if not mismo_lance:
            conteo += 1
            if conteo > limite:
                return conteo
        vistas_contextuales.append(fila)
    return conteo


def contar_acciones_relevantes(tramo: pd.DataFrame, equipo: str) -> int:
    """Cuenta las acciones de juego del equipo sin duplicar registros vinculados."""
    return _acciones_relevantes(tramo, equipo, len(tramo))


def _dentro_de_limites(
    anterior: pd.DataFrame,
    rival: pd.DataFrame,
    retorno: pd.DataFrame,
    protagonista: str,
    equipo_rival: str,
    max_segundos: Decimal,
    max_acciones: int,
) -> bool:
    if _acciones_relevantes(rival, equipo_rival, max_acciones) > max_acciones:
        return False

    filas_antes = _filas_equipo(anterior, protagonista)
    filas_rival = _filas_equipo(rival, equipo_rival)
    filas_retorno = _filas_equipo(retorno, protagonista)
    if not filas_antes or not filas_rival or not filas_retorno:
        return False
    fin_antes = _fin(filas_antes[-1])
    inicio_rival = _segundo(filas_rival[0])
    inicio_retorno = _segundo(filas_retorno[0])
    fines_rival = [_fin(fila) for fila in filas_rival]
    if any(valor is None for valor in [fin_antes, inicio_rival, inicio_retorno, *fines_rival]):
        return False
    if inicio_retorno < inicio_rival:
        return False

    # El primer marcador de pérdida incluye el tiempo de balón suelto; el
    # último fin rival evita ignorar una acción que continúa tras el retorno.
    inicio = min(fin_antes, inicio_rival)
    fin = max(inicio_retorno, *fines_rival)
    return Decimal(0) <= fin - inicio <= max_segundos


def unir_intervenciones_cortas(
    tramos: list[Tramo],
    max_segundos: float = 3,
    max_acciones: int = 3,
) -> list[pd.DataFrame]:
    """Consume A→B→A de izquierda a derecha, sin reutilizar ningún tramo."""
    if max_segundos < 0 or max_acciones < 0:
        raise ValueError("Los límites de intervención deben ser no negativos")
    limite_segundos = Decimal(str(max_segundos))
    resultado: list[pd.DataFrame] = []
    i = 0
    while i < len(tramos):
        actual = tramos[i].eventos
        protagonista = tramos[i].protagonista
        j = i
        while (
            j + 2 < len(tramos)
            and not tramos[j].barrera_despues
            and not tramos[j + 1].barrera_despues
            and protagonista is not None
            and tramos[j + 1].protagonista is not None
            and tramos[j + 1].protagonista != protagonista
            and tramos[j + 2].protagonista == protagonista
            and _dentro_de_limites(
                tramos[j].eventos, tramos[j + 1].eventos, tramos[j + 2].eventos,
                protagonista, tramos[j + 1].protagonista, limite_segundos, max_acciones,
            )
        ):
            actual = pd.concat(
                [actual, tramos[j + 1].eventos, tramos[j + 2].eventos], ignore_index=True
            )
            j += 2
        resultado.append(actual)
        i = j + 1
    return resultado
