"""Actualización in situ del corpus preparado con fases clasificables (#13)."""

import argparse
import json
import os
import shutil
import tempfile
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from src.data_preparation.corpus import _publicar_partido
from src.data_preparation.export import construir_detalle_jugada, ruta_carpeta_partido
from src.data_preparation.phases import (
    MIN_PASES_FASE, POLITICA_PATRON_DESCONOCIDO,
    separar_fases_clasificables,
)
from src.data_preparation.plays import contar_pases_completados


def _leer_json(ruta: Path) -> dict[str, Any]:
    with ruta.open(encoding="utf-8") as archivo:
        valor = json.load(archivo)
    if not isinstance(valor, dict):
        raise ValueError(f"JSON inválido para {ruta}")
    return valor


def _escribir_json(ruta: Path, valor: dict[str, Any]) -> None:
    with ruta.open("w", encoding="utf-8") as archivo:
        json.dump(valor, archivo, ensure_ascii=False, indent=2, allow_nan=False)


def _guardar_resumen(ruta: Path, resumen: dict[str, Any]) -> None:
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=ruta.parent, prefix=".resumen-", delete=False,
    ) as temporal:
        json.dump(resumen, temporal, ensure_ascii=False, indent=2, allow_nan=False)
        provisional = Path(temporal.name)
    provisional.replace(ruta)


def _leer_jugadas_validas(carpeta: Path, manifiesto: dict[str, Any]) -> list[tuple[dict[str, Any], dict[str, Any], pd.DataFrame]]:
    columnas = manifiesto["columnas"]
    if not isinstance(columnas, list) or len(set(columnas)) != len(columnas) or "id" not in columnas:
        raise ValueError(f"Columnas inválidas en {carpeta / 'manifiesto.json'}")
    if (
        carpeta.name != f"partido_{manifiesto['partido_id']}"
        or carpeta.parent.name != f"temporada_{manifiesto['temporada_id']}"
        or carpeta.parent.parent.name != f"competicion_{manifiesto['competicion_id']}"
    ):
        raise ValueError(f"Los identificadores del manifiesto no coinciden con su ruta: {carpeta}")
    prefijo = (
        f"c{manifiesto['competicion_id']}_t{manifiesto['temporada_id']}"
        f"_m{manifiesto['partido_id']}_p"
    )
    nombres: set[str] = set()
    play_ids: set[str] = set()
    ids_exportados: list[str] = []
    jugadas = []
    for detalle in manifiesto["jugadas"]:
        nombre = detalle["archivo"]
        if Path(nombre).name != nombre or not nombre.endswith(".json"):
            raise ValueError(f"Nombre de archivo de jugada inválido: {nombre}")
        if nombre in nombres or detalle["play_id"] in play_ids:
            raise ValueError(f"Archivo o play_id duplicado en {carpeta}: {nombre}")
        if not detalle["play_id"].startswith(prefijo):
            raise ValueError(f"play_id no pertenece al partido {carpeta}: {detalle['play_id']}")
        sufijo = detalle["play_id"][len(prefijo):]
        if not sufijo or nombre != f"jugada_{sufijo}.json":
            raise ValueError(f"play_id y archivo no concilian en {carpeta}: {detalle['play_id']}")
        nombres.add(nombre)
        play_ids.add(detalle["play_id"])
        bruto = _leer_json(carpeta / nombre)
        if bruto.get("columns") != columnas or not isinstance(bruto.get("data"), list):
            raise ValueError(f"Columnas o datos no concilian en {carpeta / nombre}")
        if any(not isinstance(fila, list) or len(fila) != len(columnas) for fila in bruto["data"]):
            raise ValueError(f"Filas inválidas en {carpeta / nombre}")
        if not bruto["data"]:
            raise ValueError(f"Jugada vacía en {carpeta / nombre}")
        eventos = pd.DataFrame(bruto["data"], columns=columnas)
        ids = [str(valor) for valor in eventos["id"]]
        if (
            len(ids) != detalle["total_eventos"]
            or ids[0] != detalle["primer_evento_id"]
            or ids[-1] != detalle["ultimo_evento_id"]
            or ("eventos_ids" in detalle and ids != detalle["eventos_ids"])
            or contar_pases_completados(eventos, detalle["equipo_protagonista"]) != detalle["pases_completados"]
        ):
            raise ValueError(f"Metadatos y eventos no concilian en {carpeta / nombre}")
        ids_exportados.extend(ids)
        jugadas.append((detalle, bruto, eventos))

    archivos_reales = {ruta.name for ruta in carpeta.glob("*.json")}
    if archivos_reales != nombres | {"manifiesto.json"}:
        raise ValueError(f"Archivos de jugada y manifiesto no concilian en {carpeta}")
    no_exportados = [str(evento["id"]) for evento in manifiesto["eventos_no_exportados"]]
    if (
        len(ids_exportados) != manifiesto["total_eventos_exportados"]
        or len(jugadas) != manifiesto["total_jugadas_elegibles"]
        or len(ids_exportados) + len(no_exportados) != manifiesto["total_eventos_entrada"]
        or len(set(ids_exportados + no_exportados)) != manifiesto["total_eventos_entrada"]
    ):
        raise ValueError(f"Auditoría de eventos no concilia en {carpeta}")
    return jugadas


def _vincular_archivo(origen: Path, destino: Path) -> None:
    try:
        os.link(origen, destino)
    except OSError:
        shutil.copy2(origen, destino)


def _totales_partido(manifiesto: dict[str, Any]) -> dict[str, int]:
    return {
        "total_eventos_entrada": manifiesto["total_eventos_entrada"],
        "total_eventos_exportados": manifiesto["total_eventos_exportados"],
        "total_eventos_no_exportados": len(manifiesto["eventos_no_exportados"]),
        "total_candidatas_descartadas": manifiesto["total_candidatas_descartadas"],
        "total_jugadas_elegibles": manifiesto["total_jugadas_elegibles"],
        "total_jugadas_origen_divididas": manifiesto["total_jugadas_origen_divididas"],
        "total_cortes_clasificacion": manifiesto["total_cortes_clasificacion"],
    }


def _validar_fases_publicadas(
    carpeta: Path, manifiesto: dict[str, Any],
    jugadas: list[tuple[dict[str, Any], dict[str, Any], pd.DataFrame]],
) -> None:
    grupos: dict[str, list[tuple[dict[str, Any], pd.DataFrame]]] = {}
    for detalle, _, eventos in jugadas:
        origen = detalle.get("play_id_origen")
        if not isinstance(origen, str) or not isinstance(detalle.get("eventos_ids"), list):
            raise ValueError(f"Falta la procedencia de una fase en {carpeta}")
        grupos.setdefault(origen, []).append((detalle, eventos))

    divididas = 0
    cortes = 0
    desconocidos = []
    for origen, partes in grupos.items():
        protagonista = partes[0][0]["equipo_protagonista"]
        if any(parte["equipo_protagonista"] != protagonista for parte, _ in partes):
            raise ValueError(f"Una jugada de origen mezcla protagonistas en {carpeta}: {origen}")
        eventos_origen = pd.concat([eventos for _, eventos in partes], ignore_index=True)
        resultado = separar_fases_clasificables(eventos_origen, protagonista)
        if len(resultado.fases) != len(partes):
            raise ValueError(f"Los tramos publicados no coinciden con la regla: {origen}")
        desconocidos.extend(resultado.patrones_desconocidos)
        if len(partes) > 1:
            divididas += 1
            cortes += len(partes) - 1
        numero = origen.rsplit("_p", 1)
        if len(numero) != 2 or not numero[1].isdigit():
            raise ValueError(f"play_id_origen inválido en {carpeta}: {origen}")
        for (detalle, _), fase in zip(partes, resultado.fases):
            sufijo = f"_f{fase.ordinal:02d}" if len(partes) > 1 else ""
            esperado = construir_detalle_jugada(
                fase.eventos, protagonista, f"{origen}{sufijo}",
                f"jugada_{numero[1]}{sufijo}.json", fase, origen,
            )
            if any(detalle.get(campo) != valor for campo, valor in esperado.items()):
                raise ValueError(f"Los metadatos de fase no concilian en {carpeta}: {origen}")

    if (
        manifiesto.get("total_jugadas_origen_divididas") != divididas
        or manifiesto.get("total_cortes_clasificacion") != cortes
        or manifiesto.get("patrones_desconocidos") != desconocidos
    ):
        raise ValueError(f"Totales o patrones de fase no concilian en {carpeta}")


def _actualizar_partido(carpeta: Path) -> tuple[dict[str, int], bool]:
    manifiesto = _leer_json(carpeta / "manifiesto.json")
    jugadas = _leer_jugadas_validas(carpeta, manifiesto)
    parametros = manifiesto.get("parametros", {})
    if parametros.get("dividir_fases_clasificables") is True:
        if (
            parametros.get("min_pases_por_fase") != MIN_PASES_FASE
            or parametros.get("politica_patron_desconocido") != POLITICA_PATRON_DESCONOCIDO
        ):
            raise ValueError(f"La actualización previa usó parámetros incompatibles: {carpeta}")
        _validar_fases_publicadas(carpeta, manifiesto, jugadas)
        return _totales_partido(manifiesto), False

    nuevos_detalles: list[dict[str, Any]] = []
    nuevos_archivos: dict[str, dict[str, Any]] = {}
    conservar: list[str] = []
    patrones_desconocidos: list[dict[str, Any]] = []
    divididas = 0
    cortes = 0
    ids_antes = []
    ids_despues = []
    for detalle, bruto, eventos in jugadas:
        ids_antes.extend(str(valor) for valor in eventos["id"])
        resultado = separar_fases_clasificables(eventos, detalle["equipo_protagonista"])
        patrones_desconocidos.extend(resultado.patrones_desconocidos)
        if len(resultado.fases) > 1:
            divididas += 1
            cortes += len(resultado.fases) - 1
        for fase in resultado.fases:
            dividida = len(resultado.fases) > 1
            sufijo = f"_f{fase.ordinal:02d}" if dividida else ""
            archivo = f"{Path(detalle['archivo']).stem}{sufijo}.json"
            play_id = f"{detalle['play_id']}{sufijo}"
            nuevo = {
                **detalle,
                **construir_detalle_jugada(
                    fase.eventos, detalle["equipo_protagonista"], play_id, archivo,
                    fase, detalle["play_id"],
                ),
            }
            nuevos_detalles.append(nuevo)
            ids_despues.extend(nuevo["eventos_ids"])
            if dividida:
                indices = list(fase.eventos.index)
                nuevos_archivos[archivo] = {
                    "columns": bruto["columns"],
                    "data": [bruto["data"][indice] for indice in indices],
                }
            else:
                conservar.append(detalle["archivo"])

    if Counter(ids_antes) != Counter(ids_despues) or len(set(ids_despues)) != len(ids_despues):
        raise ValueError(f"La actualización duplicaría o perdería eventos: {carpeta}")
    if len({detalle["play_id"] for detalle in nuevos_detalles}) != len(nuevos_detalles):
        raise ValueError(f"La actualización duplicaría play_id: {carpeta}")
    if len({detalle["archivo"] for detalle in nuevos_detalles}) != len(nuevos_detalles):
        raise ValueError(f"La actualización duplicaría archivos: {carpeta}")

    nuevo_manifiesto = dict(manifiesto)
    nuevo_manifiesto.update({
        "total_jugadas_elegibles": len(nuevos_detalles),
        "total_jugadas_origen_divididas": divididas,
        "total_cortes_clasificacion": cortes,
        "parametros": {
            **parametros,
            "dividir_fases_clasificables": True,
            "min_pases_por_fase": MIN_PASES_FASE,
            "politica_patron_desconocido": POLITICA_PATRON_DESCONOCIDO,
        },
        "jugadas": nuevos_detalles,
        "patrones_desconocidos": patrones_desconocidos,
    })
    if (
        len(ids_despues) != nuevo_manifiesto["total_eventos_exportados"]
        or len(ids_despues) + len(nuevo_manifiesto["eventos_no_exportados"])
        != nuevo_manifiesto["total_eventos_entrada"]
    ):
        raise ValueError(f"Los totales actualizados no concilian: {carpeta}")

    with tempfile.TemporaryDirectory(prefix=".actualizacion-", dir=carpeta.parent) as temporal:
        provisional = Path(temporal) / carpeta.name
        provisional.mkdir()
        for archivo in conservar:
            _vincular_archivo(carpeta / archivo, provisional / archivo)
        for archivo, contenido in nuevos_archivos.items():
            _escribir_json(provisional / archivo, contenido)
        _escribir_json(provisional / "manifiesto.json", nuevo_manifiesto)
        if {ruta.name for ruta in provisional.glob("*.json")} != {
            detalle["archivo"] for detalle in nuevos_detalles
        } | {"manifiesto.json"}:
            raise ValueError(f"Los archivos preparados no concilian: {carpeta}")
        _publicar_partido(provisional, carpeta, True)
    return _totales_partido(nuevo_manifiesto), True


def _totales_vigentes(resumen: dict[str, Any]) -> None:
    for campo in (
        "total_eventos_entrada", "total_eventos_exportados", "total_eventos_no_exportados",
        "total_candidatas_descartadas", "total_jugadas_elegibles",
        "total_jugadas_origen_divididas", "total_cortes_clasificacion",
    ):
        resumen[campo] = sum(partido.get(campo, 0) for partido in resumen["partidos"])
    if (
        resumen["total_eventos_exportados"] + resumen["total_eventos_no_exportados"]
        != resumen["total_eventos_entrada"]
    ):
        raise ValueError("Los totales vigentes del corpus no concilian")


def actualizar_corpus_preparado(
    ruta_corpus: str | Path,
    *,
    trabajadores: int = 1,
    progreso: Callable[[int, int], None] | None = None,
) -> dict[str, Any]:
    """Actualiza y audita los partidos ya preparados sin leer eventos originales."""
    if trabajadores < 1:
        raise ValueError("El número de trabajadores debe ser positivo")
    raiz = Path(ruta_corpus)
    ruta_resumen = raiz / "resumen_corpus.json"
    resumen = _leer_json(ruta_resumen)
    if resumen.get("partidos_fallidos") or len(resumen["partidos"]) != resumen["partidos_descubiertos"]:
        raise ValueError("El resumen de partida no describe un corpus completo")

    rutas = [
        ruta_carpeta_partido(
            raiz, partido["competicion_id"], partido["temporada_id"], partido["partido_id"],
        )
        for partido in resumen["partidos"]
    ]
    descubiertas = set(raiz.glob("competicion_*/temporada_*/partido_*/manifiesto.json"))
    esperadas = {ruta / "manifiesto.json" for ruta in rutas}
    if descubiertas != esperadas or len(esperadas) != len(rutas):
        raise ValueError("Los manifiestos descubiertos no concilian con resumen_corpus.json")

    lote = {
        "estado": "en_curso", "partidos_revisados": 0, "partidos_actualizados": 0,
        "partidos_sin_cambios": 0, "partidos_fallidos": 0, "errores": [],
    }
    resumen["estado"] = "incompleto"
    resumen["actualizacion_fases"] = lote
    _guardar_resumen(ruta_resumen, resumen)
    def incorporar(
        indice: int, carpeta: Path, partido: dict[str, Any],
        resultado: tuple[dict[str, int], bool] | None = None,
        error: Exception | None = None,
    ) -> None:
        if error is None:
            assert resultado is not None
            totales, actualizado = resultado
            partido.update(totales)
            lote["partidos_actualizados" if actualizado else "partidos_sin_cambios"] += 1
        else:
            lote["partidos_fallidos"] += 1
            lote["errores"].append({
                "manifiesto": str(carpeta / "manifiesto.json"),
                "tipo": type(error).__name__, "error": str(error),
            })
        lote["partidos_revisados"] = indice
        if progreso is not None:
            progreso(indice, len(rutas))
        if indice % 25 == 0:
            _totales_vigentes(resumen)
            _guardar_resumen(ruta_resumen, resumen)

    if trabajadores == 1:
        for indice, (carpeta, partido) in enumerate(zip(rutas, resumen["partidos"]), 1):
            try:
                resultado = _actualizar_partido(carpeta)
            except Exception as exc:
                incorporar(indice, carpeta, partido, error=exc)
            else:
                incorporar(indice, carpeta, partido, resultado)
    else:
        with ProcessPoolExecutor(
            max_workers=trabajadores, mp_context=get_context("spawn"),
        ) as ejecutor:
            pendientes = {
                ejecutor.submit(_actualizar_partido, carpeta): (carpeta, partido)
                for carpeta, partido in zip(rutas, resumen["partidos"])
            }
            for indice, futuro in enumerate(as_completed(pendientes), 1):
                carpeta, partido = pendientes[futuro]
                try:
                    resultado = futuro.result()
                except Exception as exc:
                    incorporar(indice, carpeta, partido, error=exc)
                else:
                    incorporar(indice, carpeta, partido, resultado)

    _totales_vigentes(resumen)
    lote["errores"].sort(key=lambda error: error["manifiesto"])
    lote["estado"] = "completo" if not lote["partidos_fallidos"] else "incompleto"
    resumen["estado"] = lote["estado"]
    if lote["estado"] == "completo":
        resumen["parametros"]["dividir_fases_clasificables"] = True
        resumen["parametros"]["min_pases_por_fase"] = MIN_PASES_FASE
        resumen["parametros"]["politica_patron_desconocido"] = POLITICA_PATRON_DESCONOCIDO
    _guardar_resumen(ruta_resumen, resumen)
    return resumen


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ruta_corpus", type=Path, help="Raíz del corpus preparado")
    parser.add_argument("--trabajadores", type=int, default=6)
    args = parser.parse_args()

    def mostrar_progreso(actual: int, total: int) -> None:
        if actual % 25 == 0 or actual == total:
            print(f"Partidos revisados: {actual}/{total}", flush=True)

    resumen = actualizar_corpus_preparado(
        args.ruta_corpus, trabajadores=args.trabajadores, progreso=mostrar_progreso,
    )
    lote = resumen["actualizacion_fases"]
    print(json.dumps({
        "estado": lote["estado"],
        "partidos_actualizados": lote["partidos_actualizados"],
        "partidos_sin_cambios": lote["partidos_sin_cambios"],
        "partidos_fallidos": lote["partidos_fallidos"],
        "total_jugadas_elegibles": resumen["total_jugadas_elegibles"],
        "total_cortes_clasificacion": resumen["total_cortes_clasificacion"],
        "errores": lote["errores"],
    }, ensure_ascii=False, indent=2))
    if lote["estado"] != "completo":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
