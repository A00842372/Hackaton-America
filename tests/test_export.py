import json
import tempfile
from pathlib import Path
import pandas as pd
import pytest

from src.data_preparation.export import exportar_jugadas_partido, leer_jugada


@pytest.fixture
def partido_sample():
    # Columns in a specific order with nested dicts and nulls
    columns = [
        "id", "index", "period", "timestamp", "type", "possession",
        "possession_team", "team", "pass", "location"
    ]
    records = [
        # Evento administrativo
        {
            "id": "adm-1", "index": 1, "period": 1, "timestamp": "00:00:00.000",
            "type": {"id": 35, "name": "Starting XI"}, "possession": 1,
            "possession_team": {"id": 1, "name": "Team A"},
            "team": {"id": 1, "name": "Team A"}, "pass": None, "location": None
        },
        # Posesión 2: Secuencia elegible (3 pases completados)
        {
            "id": "e-1", "index": 2, "period": 1, "timestamp": "00:00:05.100",
            "type": {"id": 30, "name": "Pass"}, "possession": 2,
            "possession_team": {"id": 1, "name": "Team A"},
            "team": {"id": 1, "name": "Team A"}, "pass": {"recipient": {"id": 10}}, "location": [10.0, 20.0]
        },
        {
            "id": "e-2", "index": 3, "period": 1, "timestamp": "00:00:07.200",
            "type": {"id": 30, "name": "Pass"}, "possession": 2,
            "possession_team": {"id": 1, "name": "Team A"},
            "team": {"id": 1, "name": "Team A"}, "pass": {}, "location": [25.0, 30.0]
        },
        {
            "id": "e-3", "index": 4, "period": 1, "timestamp": "00:00:09.500",
            "type": {"id": 30, "name": "Pass"}, "possession": 2,
            "possession_team": {"id": 1, "name": "Team A"},
            "team": {"id": 1, "name": "Team A"}, "pass": None, "location": [40.0, 50.0]
        },
        # Posesión 3: Secuencia candidata descartada (solo 1 pase completado)
        {
            "id": "e-4", "index": 5, "period": 1, "timestamp": "00:01:00.000",
            "type": {"id": 30, "name": "Pass"}, "possession": 3,
            "possession_team": {"id": 2, "name": "Team B"},
            "team": {"id": 2, "name": "Team B"}, "pass": None, "location": [50.0, 50.0]
        },
        {
            "id": "e-5", "index": 6, "period": 1, "timestamp": "00:01:02.000",
            "type": {"id": 43, "name": "Carry"}, "possession": 3,
            "possession_team": {"id": 2, "name": "Team B"},
            "team": {"id": 2, "name": "Team B"}, "pass": None, "location": [55.0, 52.0]
        },
    ]
    df = pd.DataFrame(records, columns=columns)
    return df


def test_exportar_jugadas_partido_genera_archivos_y_manifiesto(partido_sample):
    with tempfile.TemporaryDirectory() as tmpdir:
        base_salida = Path(tmpdir)
        metadatos = {
            "competicion_id": 73,
            "temporada_id": 108,
            "partido_id": 12345,
            "temporada_nombre": "2025/2026",
        }

        resumen = exportar_jugadas_partido(
            partido_sample,
            ruta_salida=base_salida,
            metadatos_partido=metadatos,
            min_pases=3,
        )

        dir_partido = base_salida / "competicion_73" / "temporada_108" / "partido_12345"
        assert dir_partido.is_dir()

        # Debe existir jugada_0001.json y manifiesto.json, NO jugada_0002.json
        assert (dir_partido / "jugada_0001.json").is_file()
        assert not (dir_partido / "jugada_0002.json").exists()
        assert (dir_partido / "manifiesto.json").is_file()

        # Validar contenido del manifiesto
        with open(dir_partido / "manifiesto.json", "r", encoding="utf-8") as f:
            manifiesto = json.load(f)

        assert manifiesto["partido_id"] == 12345
        assert manifiesto["total_eventos_entrada"] == 6
        assert manifiesto["total_eventos_exportados"] == 3
        assert manifiesto["total_candidatas_descartadas"] == 1
        assert manifiesto["total_jugadas_elegibles"] == 1
        assert len(manifiesto["jugadas"]) == 1
        assert manifiesto["jugadas"][0]["play_id"] == "c73_t108_m12345_p0001"
        assert manifiesto["jugadas"][0]["pases_completados"] == 3

        # Conciliación: exportados + no exportados == total entrada
        assert manifiesto["total_eventos_exportados"] + len(manifiesto["eventos_no_exportados"]) == manifiesto["total_eventos_entrada"]


def test_releer_jugada_preserva_columnas_orden_y_anidados(partido_sample):
    with tempfile.TemporaryDirectory() as tmpdir:
        base_salida = Path(tmpdir)
        metadatos = {
            "competicion_id": 73,
            "temporada_id": 108,
            "partido_id": 12345,
        }
        exportar_jugadas_partido(
            partido_sample,
            ruta_salida=base_salida,
            metadatos_partido=metadatos,
            min_pases=3,
        )

        ruta_jugada = base_salida / "competicion_73" / "temporada_108" / "partido_12345" / "jugada_0001.json"
        df_releido = leer_jugada(ruta_jugada)

        # Mismo orden y nombres de columnas que el original
        assert list(df_releido.columns) == list(partido_sample.columns)
        assert len(df_releido) == 3

        # Los eventos mantienen el orden
        assert list(df_releido["id"]) == ["e-1", "e-2", "e-3"]

        # Valores anidados se recuperan como diccionarios/listas intactos
        primera_fila = df_releido.iloc[0]
        assert isinstance(primera_fila["type"], dict)
        assert primera_fila["type"]["name"] == "Pass"
        assert primera_fila["location"] == [10.0, 20.0]
        assert isinstance(primera_fila["pass"], dict)

        # Valores faltantes conservados como null / None / NaN
        tercera_fila = df_releido.iloc[2]
        assert pd.isna(tercera_fila["pass"]) or tercera_fila["pass"] is None


def test_no_sobrescribir_si_directorio_existe(partido_sample):
    with tempfile.TemporaryDirectory() as tmpdir:
        base_salida = Path(tmpdir)
        metadatos = {
            "competicion_id": 73,
            "temporada_id": 108,
            "partido_id": 12345,
        }
        exportar_jugadas_partido(
            partido_sample,
            ruta_salida=base_salida,
            metadatos_partido=metadatos,
            sobrescribir=False,
        )

        # Segunda llamada con sobrescribir=False debe fallar antes de escribir
        with pytest.raises(FileExistsError, match="Ya existe"):
            exportar_jugadas_partido(
                partido_sample,
                ruta_salida=base_salida,
                metadatos_partido=metadatos,
                sobrescribir=False,
            )

        # Con sobrescribir=True debe permitir la regeneración
        resumen = exportar_jugadas_partido(
            partido_sample,
            ruta_salida=base_salida,
            metadatos_partido=metadatos,
            sobrescribir=True,
        )
        assert resumen["total_jugadas_elegibles"] == 1
