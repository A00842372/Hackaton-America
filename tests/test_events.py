import json
import tempfile
from pathlib import Path
import pytest
import pandas as pd

from src.data_preparation.events import cargar_eventos_partido


def test_cargar_eventos_validos():
    data = {
        "evt-2": {
            "id": "evt-2",
            "match_id": 12345,
            "period": 1,
            "index": 2,
            "type": {"id": 30, "name": "Pass"},
            "team": {"id": 1, "name": "Team A"},
        },
        "evt-1": {
            "id": "evt-1",
            "match_id": 12345,
            "period": 1,
            "index": 1,
            "type": {"id": 35, "name": "Starting XI"},
            "team": {"id": 1, "name": "Team A"},
        },
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = Path(tmpdir) / "12345.json"
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f)

        df = cargar_eventos_partido(file_path)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        # Check ordering by period and index
        assert list(df["index"]) == [1, 2]
        assert list(df["id"]) == ["evt-1", "evt-2"]


def test_validar_coincidencia_id_evento_falla():
    data = {
        "mismatched-key": {
            "id": "evt-1",
            "match_id": 12345,
            "period": 1,
            "index": 1,
        }
    }
    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = Path(tmpdir) / "12345.json"
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f)

        with pytest.raises(ValueError, match="id"):
            cargar_eventos_partido(file_path)


def test_validar_coincidencia_match_id_falla():
    data = {
        "evt-1": {
            "id": "evt-1",
            "match_id": 99999,  # Mismatch with filename 12345.json
            "period": 1,
            "index": 1,
        }
    }
    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = Path(tmpdir) / "12345.json"
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f)

        with pytest.raises(ValueError, match="match_id"):
            cargar_eventos_partido(file_path)
