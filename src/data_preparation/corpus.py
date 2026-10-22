"""Procesamiento auditable de todos los partidos locales con eventos."""

import json
import shutil
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from multiprocessing import get_context
from pathlib import Path
from typing import Any, Union
from uuid import uuid4

from src.data_preparation.events import cargar_eventos_partido
from src.data_preparation.export import exportar_jugadas_partido, ruta_carpeta_partido


@dataclass(frozen=True)
class TareaPartido:
    ruta: Path
    salida: Path
    metadatos: dict[str, Any]
    sobrescribir: bool
    min_pases: int
    max_segundos_intervencion: float
    max_acciones_intervencion: int


def descubrir_archivos_eventos(ruta_entrada: Union[str, Path]) -> list[Path]:
    """Encuentra cada partido con eventos, tenga o no datos 360."""
    return sorted(Path(ruta_entrada).glob("competicion_*/temporada_*/eventos/*.json"))


def _identificadores(ruta: Path) -> tuple[int, int, int]:
    try:
        return (
            int(ruta.parents[2].name.removeprefix("competicion_")),
            int(ruta.parents[1].name.removeprefix("temporada_")),
            int(ruta.stem),
        )
    except ValueError as exc:
        raise ValueError(f"Ruta de eventos con identificadores no numéricos: {ruta}") from exc


def _nombre_temporada(ruta: Path, partido_id: int, cache: dict[Path, dict[int, str]]) -> str | None:
    ruta_partidos = ruta.parents[1] / "partidos.json"
    if ruta_partidos not in cache:
        nombres: dict[int, str] = {}
        if ruta_partidos.is_file():
            try:
                with ruta_partidos.open(encoding="utf-8") as archivo:
                    partidos = json.load(archivo)
                if isinstance(partidos, list):
                    nombres = {
                        int(partido["match_id"]): str(partido["season"])
                        for partido in partidos
                        if isinstance(partido, dict) and partido.get("match_id") is not None
                        and partido.get("season") is not None
                    }
            except (OSError, ValueError, TypeError, KeyError):
                # Los metadatos son opcionales; los eventos siguen siendo la fuente.
                pass
        cache[ruta_partidos] = nombres
    return cache[ruta_partidos].get(partido_id)


def _publicar_partido(origen: Path, destino: Path, sobrescribir: bool) -> None:
    """Publica una exportación completa y restaura la anterior si falla el cambio."""
    destino.parent.mkdir(parents=True, exist_ok=True)
    if destino.exists() and not sobrescribir:
        raise FileExistsError(f"Ya existe el directorio de exportación para el partido: {destino}")
    respaldo = None
    if destino.exists():
        respaldo = destino.parent / f".{destino.name}.respaldo-{uuid4().hex}"
        destino.rename(respaldo)
    try:
        origen.rename(destino)
    except Exception:
        if respaldo is not None:
            respaldo.rename(destino)
        raise
    if respaldo is not None:
        shutil.rmtree(respaldo, ignore_errors=True)


def _procesar_partido(tarea: TareaPartido) -> dict[str, Any]:
    competicion = tarea.metadatos["competicion_id"]
    temporada = tarea.metadatos["temporada_id"]
    partido = tarea.metadatos["partido_id"]
    destino = ruta_carpeta_partido(tarea.salida, competicion, temporada, partido)
    if destino.exists() and not tarea.sobrescribir:
        raise FileExistsError(f"Ya existe el directorio de exportación para el partido: {destino}")
    eventos = cargar_eventos_partido(tarea.ruta)
    with tempfile.TemporaryDirectory(prefix=".partido-", dir=tarea.salida) as temporal:
        manifiesto = exportar_jugadas_partido(
            eventos, temporal, tarea.metadatos,
            min_pases=tarea.min_pases,
            max_segundos_intervencion=tarea.max_segundos_intervencion,
            max_acciones_intervencion=tarea.max_acciones_intervencion,
        )
        if (
            manifiesto["total_eventos_exportados"]
            + len(manifiesto["eventos_no_exportados"])
            != manifiesto["total_eventos_entrada"]
        ):
            raise ValueError(f"La auditoría de eventos no concilia: {tarea.ruta}")
        generado = ruta_carpeta_partido(temporal, competicion, temporada, partido)
        _publicar_partido(generado, destino, tarea.sobrescribir)
    return {
        "partido_id": partido,
        "competicion_id": competicion,
        "temporada_id": temporada,
        "archivo_eventos": str(tarea.ruta),
        "manifiesto": str(destino / "manifiesto.json"),
        "total_eventos_entrada": manifiesto["total_eventos_entrada"],
        "total_eventos_exportados": manifiesto["total_eventos_exportados"],
        "total_eventos_no_exportados": len(manifiesto["eventos_no_exportados"]),
        "total_candidatas_descartadas": manifiesto["total_candidatas_descartadas"],
        "total_jugadas_elegibles": manifiesto["total_jugadas_elegibles"],
    }


def procesar_corpus(
    ruta_entrada: Union[str, Path],
    ruta_salida: Union[str, Path],
    *,
    sobrescribir: bool = False,
    min_pases: int = 3,
    max_segundos_intervencion: float = 3,
    max_acciones_intervencion: int = 3,
    trabajadores: int = 1,
) -> dict[str, Any]:
    """Procesa independientemente cada partido y deja un resumen del lote."""
    entrada, salida = Path(ruta_entrada), Path(ruta_salida)
    if entrada.resolve() == salida.resolve():
        raise ValueError("Las rutas de entrada y salida deben ser distintas")
    if trabajadores < 1:
        raise ValueError("El número de trabajadores debe ser positivo")
    archivos = descubrir_archivos_eventos(entrada)
    salida.mkdir(parents=True, exist_ok=True)
    cache_temporadas: dict[Path, dict[int, str]] = {}
    resumen: dict[str, Any] = {
        "estado": "completo",
        "ruta_entrada": str(entrada),
        "ruta_salida": str(salida),
        "partidos_descubiertos": len(archivos),
        "partidos_procesados": 0,
        "partidos_fallidos": 0,
        "alcance_totales": "partidos_procesados",
        "total_eventos_entrada": 0,
        "total_eventos_exportados": 0,
        "total_eventos_no_exportados": 0,
        "total_candidatas_descartadas": 0,
        "total_jugadas_elegibles": 0,
        "parametros": {
            "min_pases": min_pases,
            "max_segundos_intervencion": max_segundos_intervencion,
            "max_acciones_intervencion": max_acciones_intervencion,
            "sobrescribir": sobrescribir,
            "trabajadores": trabajadores,
        },
        "partidos": [],
        "errores": [],
    }
    tareas = []
    for ruta in archivos:
        try:
            competicion, temporada, partido = _identificadores(ruta)
            destino = ruta_carpeta_partido(salida, competicion, temporada, partido)
            if destino.exists() and not sobrescribir:
                raise FileExistsError(f"Ya existe el directorio de exportación para el partido: {destino}")
            metadatos = {
                "competicion_id": competicion,
                "temporada_id": temporada,
                "partido_id": partido,
                "temporada_nombre": _nombre_temporada(ruta, partido, cache_temporadas),
            }
            tareas.append(TareaPartido(
                ruta, salida, metadatos, sobrescribir, min_pases,
                max_segundos_intervencion, max_acciones_intervencion,
            ))
        except Exception as exc:
            resumen["errores"].append({
                "archivo_eventos": str(ruta),
                "tipo": type(exc).__name__,
                "error": str(exc),
            })
            resumen["partidos_fallidos"] += 1

    def incorporar(ruta: Path, resultado: dict[str, Any] | None = None, error: Exception | None = None) -> None:
        if error is not None:
            resumen["errores"].append({
                "archivo_eventos": str(ruta),
                "tipo": type(error).__name__,
                "error": str(error),
            })
            resumen["partidos_fallidos"] += 1
            return
        assert resultado is not None
        resumen["partidos"].append(resultado)
        resumen["partidos_procesados"] += 1
        for campo in (
            "total_eventos_entrada", "total_eventos_exportados",
            "total_eventos_no_exportados", "total_candidatas_descartadas",
            "total_jugadas_elegibles",
        ):
            resumen[campo] += resultado[campo]

    if trabajadores == 1:
        for tarea in tareas:
            try:
                incorporar(tarea.ruta, _procesar_partido(tarea))
            except Exception as exc:
                incorporar(tarea.ruta, error=exc)
    else:
        with ProcessPoolExecutor(max_workers=trabajadores, mp_context=get_context("spawn")) as ejecutor:
            pendientes = {ejecutor.submit(_procesar_partido, tarea): tarea.ruta for tarea in tareas}
            for futuro in as_completed(pendientes):
                ruta = pendientes[futuro]
                try:
                    incorporar(ruta, futuro.result())
                except Exception as exc:
                    incorporar(ruta, error=exc)
    resumen["partidos"].sort(key=lambda partido: partido["archivo_eventos"])
    resumen["errores"].sort(key=lambda error: error["archivo_eventos"])
    if resumen["partidos_fallidos"] or not archivos:
        resumen["estado"] = "incompleto"
    ruta_resumen = salida / "resumen_corpus.json"
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=salida, delete=False) as temporal:
        json.dump(resumen, temporal, ensure_ascii=False, indent=2)
        ruta_temporal = Path(temporal.name)
    ruta_temporal.replace(ruta_resumen)
    return resumen
