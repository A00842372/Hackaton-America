import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Union
import numpy as np
import pandas as pd

from src.data_preparation.plays import (
    contar_pases_completados,
    es_jugada_elegible,
    extraer_secuencias_candidatas,
    obtener_equipo_protagonista,
)
from src.data_preparation.phases import (
    FaseJugada, MIN_PASES_FASE, POLITICA_PATRON_DESCONOCIDO,
    separar_fases_clasificables,
)


def ruta_carpeta_partido(
    ruta_salida: Union[str, Path], competicion_id: int, temporada_id: int, partido_id: int,
) -> Path:
    return (
        Path(ruta_salida) / f"competicion_{competicion_id}"
        / f"temporada_{temporada_id}" / f"partido_{partido_id}"
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
    max_segundos_intervencion: float = 3,
    max_acciones_intervencion: int = 3,
    dividir_fases_clasificables: bool = False,
) -> Dict[str, Any]:
    """Exporta las jugadas elegibles de un partido y genera el manifiesto de auditoría.
    
    Crea la estructura <ruta_salida>/competicion_{c}/temporada_{t}/partido_{m}/.
    Si la carpeta existe y sobrescribir=False, genera un FileExistsError.
    """
    competicion_id = metadatos_partido.get("competicion_id", 0)
    temporada_id = metadatos_partido.get("temporada_id", 0)
    partido_id = metadatos_partido.get("partido_id", 0)

    dir_partido = ruta_carpeta_partido(ruta_salida, competicion_id, temporada_id, partido_id)

    if dir_partido.exists():
        if not sobrescribir:
            raise FileExistsError(
                f"Ya existe el directorio de exportación para el partido: {dir_partido}. "
                f"Active sobrescribir=True para reemplazar los resultados."
            )
        for f in dir_partido.glob("*.json"):
            f.unlink()
    else:
        dir_partido.mkdir(parents=True, exist_ok=True)

    columnas_originales = list(df_eventos.columns)
    secuencias_candidatas, eventos_no_exportados = extraer_secuencias_candidatas(
        df_eventos, max_segundos_intervencion, max_acciones_intervencion
    )

    jugadas_manifiesto: List[Dict[str, Any]] = []
    total_eventos_exportados = 0
    total_candidatas_descartadas = 0
    play_num = 0
    ids_exportados: List[str] = []
    patrones_desconocidos: List[Dict[str, Any]] = []
    total_jugadas_origen_divididas = 0
    total_cortes_clasificacion = 0

    for secuencia in secuencias_candidatas:
        es_eleg, pases = es_jugada_elegible(secuencia, min_pases=min_pases)

        if es_eleg:
            play_num += 1
            archivo_base = f"jugada_{play_num:04d}"
            play_id_origen = f"c{competicion_id}_t{temporada_id}_m{partido_id}_p{play_num:04d}"
            protagonista = obtener_equipo_protagonista(secuencia)
            if dividir_fases_clasificables:
                resultado_fases = separar_fases_clasificables(secuencia, protagonista)
                fases = resultado_fases.fases
                patrones_desconocidos.extend(resultado_fases.patrones_desconocidos)
                if len(fases) > 1:
                    total_jugadas_origen_divididas += 1
                    total_cortes_clasificacion += len(fases) - 1
            else:
                fases = [FaseJugada(secuencia, "", 1, "")]

            for fase in fases:
                eventos = fase.eventos
                dividida = len(fases) > 1
                sufijo = f"_f{fase.ordinal:02d}" if dividida else ""
                archivo_nombre = f"{archivo_base}{sufijo}.json"
                play_id = f"{play_id_origen}{sufijo}"
                filas_serializadas = [
                    [_sanitizar_para_json(row[col]) for col in columnas_originales]
                    for _, row in eventos[columnas_originales].iterrows()
                ]
                with open(dir_partido / archivo_nombre, "w", encoding="utf-8") as f:
                    json.dump({"columns": columnas_originales, "data": filas_serializadas},
                              f, ensure_ascii=False, indent=2)

                total_eventos_exportados += len(eventos)
                ids_exportados.extend(str(id_evento) for id_evento in eventos["id"])
                posesiones_origen = []
                if "possession" in eventos.columns:
                    posesiones_origen = sorted({int(p) for p in eventos["possession"].dropna()})
                intervenciones_rivales = []
                for _, row in eventos.iterrows():
                    poseedor = row.get("possession_team")
                    if isinstance(poseedor, dict):
                        poseedor = poseedor.get("name")
                    if isinstance(poseedor, str) and poseedor != protagonista:
                        intervenciones_rivales.append(str(row.get("id")))

                detalle = {
                    "play_id": play_id,
                    "archivo": archivo_nombre,
                    "period": int(eventos.iloc[0].get("period", 1)),
                    "equipo_protagonista": protagonista,
                    "primer_evento_id": str(eventos.iloc[0].get("id")),
                    "ultimo_evento_id": str(eventos.iloc[-1].get("id")),
                    "primer_index": int(eventos.iloc[0].get("index", 0)),
                    "ultimo_index": int(eventos.iloc[-1].get("index", 0)),
                    "posesiones_origen": posesiones_origen,
                    "intervenciones_rivales_absorbidas": intervenciones_rivales,
                    "pases_completados": contar_pases_completados(eventos, protagonista),
                    "total_eventos": len(eventos),
                }
                if dividir_fases_clasificables:
                    detalle.update({
                        "play_id_origen": play_id_origen,
                        "ordinal_tramo": fase.ordinal,
                        "categoria_clasificacion": fase.categoria,
                        "eventos_ids": [str(valor) for valor in eventos["id"]],
                        "motivo_corte": fase.motivo,
                        "corte_antes": fase.corte_antes,
                        "corte_despues": fase.corte_despues,
                    })
                jugadas_manifiesto.append(detalle)
        else:
            total_candidatas_descartadas += 1
            for _, row in secuencia.iterrows():
                eventos_no_exportados.append({
                    "id": str(row.get("id")),
                    "motivo": "secuencia_descartada_menos_3_pases",
                })

    ids_entrada = [] if df_eventos.empty else [str(id_evento) for id_evento in df_eventos["id"]]
    ids_auditados = ids_exportados + [evento["id"] for evento in eventos_no_exportados]
    if len(set(ids_entrada)) != len(ids_entrada) or Counter(ids_entrada) != Counter(ids_auditados):
        raise ValueError("La auditoría de identificadores de eventos no concilia exactamente")

    manifiesto = {
        "partido_id": partido_id,
        "competicion_id": competicion_id,
        "temporada_id": temporada_id,
        "temporada_nombre": metadatos_partido.get("temporada_nombre"),
        "total_eventos_entrada": len(df_eventos),
        "total_eventos_exportados": total_eventos_exportados,
        "total_candidatas_descartadas": total_candidatas_descartadas,
        "total_jugadas_elegibles": len(jugadas_manifiesto),
        "total_jugadas_origen_divididas": total_jugadas_origen_divididas,
        "total_cortes_clasificacion": total_cortes_clasificacion,
        "columnas": columnas_originales,
        "parametros": {
            "min_pases": min_pases,
            "max_segundos_intervencion": max_segundos_intervencion,
            "max_acciones_intervencion": max_acciones_intervencion,
            "dividir_fases_clasificables": dividir_fases_clasificables,
            "min_pases_por_fase": MIN_PASES_FASE,
            "politica_patron_desconocido": POLITICA_PATRON_DESCONOCIDO,
        },
        "jugadas": jugadas_manifiesto,
        "eventos_no_exportados": eventos_no_exportados,
        "patrones_desconocidos": patrones_desconocidos,
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
