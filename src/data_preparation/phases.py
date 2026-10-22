"""Separación opcional de fases para la clasificación de jugadas elegibles."""

from dataclasses import dataclass
from typing import Any

import pandas as pd

from src.data_preparation.interventions import contar_acciones_relevantes
from src.data_preparation.plays import contar_pases_completados


BALON_PARADO = {"From Corner", "From Free Kick", "From Throw In"}
JUEGO_ABIERTO = {
    "Regular Play", "From Counter", "From Goal Kick", "From Kick Off", "From Keeper",
}
POLITICA_PATRON_DESCONOCIDO = "heredar_anterior_si_existe_sino_siguiente_sino_juego_abierto"
MIN_PASES_FASE = 3


def _nombre(valor: Any) -> str | None:
    if isinstance(valor, dict):
        return valor.get("name")
    return valor if isinstance(valor, str) else None


def _categoria(patron: str | None) -> str | None:
    if patron in BALON_PARADO:
        return "balon_parado"
    if patron in JUEGO_ABIERTO:
        return "juego_abierto"
    return None


@dataclass(frozen=True)
class FaseJugada:
    eventos: pd.DataFrame
    categoria: str
    ordinal: int
    motivo: str
    corte_antes: dict[str, str] | None = None
    corte_despues: dict[str, str] | None = None


@dataclass(frozen=True)
class ResultadoFases:
    fases: list[FaseJugada]
    patrones_desconocidos: list[dict[str, Any]]


def separar_fases_clasificables(
    jugada: pd.DataFrame, equipo_protagonista: str,
) -> ResultadoFases:
    """Parte una jugada solo si todos los tramos entre categorías son elegibles.

    Solo las filas del protagonista señalan cambios de fase: una intervención
    rival absorbida permanece en la fase activa. Los patrones desconocidos
    heredan la fase anterior; al inicio se usa la siguiente fase conocida y,
    si no existe ninguna, juego abierto. Un empate de acciones propias en una
    jugada conservada favorece la primera fase observada.
    """
    if jugada.empty:
        raise ValueError("No se puede clasificar una jugada vacía")

    marcadores: list[str | None] = []
    desconocidos: list[dict[str, Any]] = []
    indices_desconocidos: list[int] = []
    for indice, (_, fila) in enumerate(jugada.iterrows()):
        patron = _nombre(fila.get("play_pattern"))
        categoria = _categoria(patron)
        if categoria is None:
            desconocidos.append({"id": str(fila.get("id")), "play_pattern": patron})
            indices_desconocidos.append(indice)
        marcadores.append(categoria if _nombre(fila.get("team")) == equipo_protagonista else None)

    primera_conocida = next((valor for valor in marcadores if valor is not None), "juego_abierto")
    categorias: list[str] = []
    actual = primera_conocida
    for marcador in marcadores:
        if marcador is not None:
            actual = marcador
        categorias.append(actual)
    for indice, desconocido in zip(indices_desconocidos, desconocidos):
        desconocido["categoria_asignada"] = categorias[indice]

    limites = [0] + [
        i for i in range(1, len(categorias)) if categorias[i] != categorias[i - 1]
    ] + [len(jugada)]
    tramos = [jugada.iloc[inicio:fin] for inicio, fin in zip(limites, limites[1:])]
    pases = [contar_pases_completados(tramo, equipo_protagonista) for tramo in tramos]
    dividir = len(tramos) > 1 and all(total >= MIN_PASES_FASE for total in pases)

    if dividir:
        fases = []
        for i, (inicio, fin) in enumerate(zip(limites, limites[1:])):
            corte_antes = None if i == 0 else {
                "evento_anterior_id": str(jugada.iloc[inicio - 1]["id"]),
                "evento_siguiente_id": str(jugada.iloc[inicio]["id"]),
            }
            corte_despues = None if i == len(tramos) - 1 else {
                "evento_anterior_id": str(jugada.iloc[fin - 1]["id"]),
                "evento_siguiente_id": str(jugada.iloc[fin]["id"]),
            }
            fases.append(FaseJugada(
                tramos[i], categorias[inicio], i + 1,
                "cambio_categoria_clasificacion", corte_antes, corte_despues,
            ))
        return ResultadoFases(fases, desconocidos)

    conteos = {
        categoria: contar_acciones_relevantes(
            jugada.iloc[[i for i, valor in enumerate(categorias) if valor == categoria]],
            equipo_protagonista,
        )
        for categoria in ("balon_parado", "juego_abierto")
    }
    maxima = max(conteos.values())
    categoria = next(valor for valor in categorias if conteos[valor] == maxima)
    motivo = "tramo_con_menos_de_min_pases" if len(tramos) > 1 else "sin_cambio_categoria"
    return ResultadoFases([FaseJugada(jugada, categoria, 1, motivo)], desconocidos)
