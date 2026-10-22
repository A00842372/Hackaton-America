import json
import math
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd

from src.data_preparation.plays import (
    _extraer_nombre,
    es_jugada_elegible,
    extraer_secuencias_posesion,
)


def _sanitizar_para_json(val: Any) -> Any:
    """Convierte tipos de numpy o NaN a tipos compatibles con JSON (None para null)."""
    if isinstance(val, (np.integer, int)):
        return int(val)
    if isinstance(val, (np.floating, float)):
        if math.isnan(val) or np.isnan(val):
            return None
        return float(val)
    if isinstance(val, np.ndarray):
        return [_sanitizar_para_json(x) for x in val.tolist()]
    if isinstance(val, list):
        return [_sanitizar_para_json(x) for x in val]
    if isinstance(val, dict):
        return {k: _sanitizar_para_json(v) for k, v in val.items()}
    if pd.isna(val):
        return None
    return val


def exportar_jugadas_partido(
    df_eventos: pd.DataFrame,
    ruta_salida: Union[str, Path],
    metadatos_partido: Dict[str, Any],
    sobrescribir: bool = False,
    min_pases: int = 3,
) -> Dict[str, Any]:
    """Exporta las jugadas elegibles de un partido y genera el manifiesto de auditoría.
    
    Crea la estructura <ruta_salida>/competicion_{c}/temporada_{t}/partido_{m}/.
    Si la carpeta existe y sobrescribir=False, genera un FileExistsError.
    """
    competicion_id = metadatos_partido.get("competicion_id", 0)
    temporada_id = metadatos_partido.get("temporada_id", 0)
    partido_id = metadatos_partido.get("partido_id", 0)

    dir_partido = (
        Path(ruta_salida)
        / f"competicion_{competicion_id}"
        / f"temporada_{temporada_id}"
        / f"partido_{partido_id}"
    )

    if dir_partido.exists():
        if not sobrescribir:
            raise FileExistsError(
                f"Ya existe el directorio de exportación para el partido: {dir_partido}. "
                f"Active sobrescribir=True para reemplazar los resultados."
            )
        # Limpiar si se permite sobrescribir
        for f in dir_partido.glob("*.json"):
            f.unlink()
    else:
        dir_partido.mkdir(parents=True, exist_ok=True)

    columnas_originales = list(df_eventos.columns)
    secuencias_candidatas, eventos_no_exportados = extraer_secuencias_posesion(df_eventos)

    jugadas_manifiesto: List[Dict[str, Any]] = []
    total_eventos_exportados = 0
    total_candidatas_descartadas = 0
    play_num = 0

    for secuencia in secuencias_candidatas:
        es_eleg, pases = es_jugada_elegible(secuencia, min_pases=min_pases)

        if es_eleg:
            play_num += 1
            archivo_nombre = f"jugada_{play_num:04d}.json"
            ruta_archivo = dir_partido / archivo_nombre
            play_id = f"c{competicion_id}_t{temporada_id}_m{partido_id}_p{play_num:04d}"

            # Determinar protagonista
            protagonista = None
            if "possession_team" in secuencia.columns:
                for v in secuencia["possession_team"]:
                    nom = _extraer_nombre(v)
                    if nom:
                        protagonista = nom
                        break
            if not protagonista and "team" in secuencia.columns:
                protagonista = _extraer_nombre(secuencia["team"].iloc[0])

            # Preparar datos tabulares orient="split"
            filas_serializadas = []
            for _, row in secuencia[columnas_originales].iterrows():
                fila_dict = {col: _sanitizar_para_json(row[col]) for col in columnas_originales}
                filas_serializadas.append([fila_dict[col] for col in columnas_originales])

            contenido_jugada = {
                "columns": columnas_originales,
                "data": filas_serializadas,
            }

            with open(ruta_archivo, "w", encoding="utf-8") as f:
                json.dump(contenido_jugada, f, ensure_ascii=False, indent=2)

            total_eventos_exportados += len(secuencia)

            posesiones_origen = []
            if "possession" in secuencia.columns:
                posesiones_origen = sorted(
                    list({int(p) for p in secuencia["possession"].dropna()})
                )

            jugadas_manifiesto.append({
                "play_id": play_id,
                "archivo": archivo_nombre,
                "period": int(secuencia.iloc[0].get("period", 1)),
                "equipo_protagonista": protagonista,
                "primer_evento_id": str(secuencia.iloc[0].get("id")),
                "ultimo_evento_id": str(secuencia.iloc[-1].get("id")),
                "primer_index": int(secuencia.iloc[0].get("index", 0)),
                "ultimo_index": int(secuencia.iloc[-1].get("index", 0)),
                "posesiones_origen": posesiones_origen,
                "pases_completados": pases,
                "total_eventos": len(secuencia),
            })
        else:
            total_candidatas_descartadas += 1
            for _, row in secuencia.iterrows():
                eventos_no_exportados.append({
                    "id": str(row.get("id")),
                    "motivo": "secuencia_descartada_menos_3_pases",
                })

    manifiesto = {
        "partido_id": partido_id,
        "competicion_id": competicion_id,
        "temporada_id": temporada_id,
        "temporada_nombre": metadatos_partido.get("temporada_nombre"),
        "total_eventos_entrada": len(df_eventos),
        "total_eventos_exportados": total_eventos_exportados,
        "total_candidatas_descartadas": total_candidatas_descartadas,
        "total_jugadas_elegibles": len(jugadas_manifiesto),
        "columnas": columnas_originales,
        "parametros": {"min_pases": min_pases},
        "jugadas": jugadas_manifiesto,
        "eventos_no_exportados": eventos_no_exportados,
    }

    with open(dir_partido / "manifiesto.json", "w", encoding="utf-8") as f:
        json.dump(manifiesto, f, ensure_ascii=False, indent=2)

    return manifiesto


def leer_jugada(ruta_jugada: Union[str, Path]) -> pd.DataFrame:
    """Relee un archivo de jugada exportado en formato JSON tabular.
    
    Recupera todas las columnas originales en el mismo orden, con datos anidados
    y valores nulos conservados.
    """
    path = Path(ruta_jugada)
    if not path.is_file():
        raise FileNotFoundError(f"No se encontró el archivo de jugada: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict) or "columns" not in data or "data" not in data:
        raise ValueError(f"El archivo {path.name} no tiene una estructura tabular válida.")

    df = pd.DataFrame(data=data["data"], columns=data["columns"])
    return df
