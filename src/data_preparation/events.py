import json
from pathlib import Path
from typing import Union
import pandas as pd


def cargar_eventos_partido(ruta_archivo: Union[str, Path]) -> pd.DataFrame:
    """Carga y valida los eventos locales de un partido desde su archivo JSON.
    
    Valida la concordancia entre la clave del diccionario, el campo 'id' de cada
    registro y el 'match_id' deducido del nombre de archivo. Devuelve el DataFrame
    ordenado por 'period' e 'index'.
    """
    path = Path(ruta_archivo)
    if not path.is_file():
        raise FileNotFoundError(f"No se encontró el archivo de eventos: {path}")

    stem = path.stem
    expected_match_id = int(stem) if stem.isdigit() else None

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError(f"El archivo {path.name} debe contener un diccionario de eventos.")

    records = []
    for key, event in data.items():
        if not isinstance(event, dict):
            raise ValueError(f"El evento bajo la clave '{key}' no es un diccionario válido.")

        event_id = event.get("id")
        if event_id != key:
            raise ValueError(
                f"Discrepancia de identificador: la clave '{key}' no coincide con event['id'] '{event_id}'."
            )

        event_match_id = event.get("match_id")
        if expected_match_id is not None and event_match_id is not None and event_match_id != expected_match_id:
            raise ValueError(
                f"Discrepancia de match_id: el evento {event_id} tiene match_id={event_match_id}, "
                f"pero el archivo es {path.name} (esperado {expected_match_id})."
            )

        records.append(event)

    df = pd.DataFrame(records)
    if not df.empty:
        sort_cols = [c for c in ["period", "index"] if c in df.columns]
        if sort_cols:
            df = df.sort_values(by=sort_cols).reset_index(drop=True)

    return df
