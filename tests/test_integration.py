import json
import tempfile
from pathlib import Path
import pytest
import pandas as pd

from src.data_preparation.events import cargar_eventos_partido
from src.data_preparation.export import exportar_jugadas_partido, leer_jugada


def test_caso_reproducible_un_partido_local():
    # Buscar un archivo real de eventos en el repositorio
    event_files = list(Path("datos_statsbomb").glob("**/eventos/*.json"))
    assert len(event_files) > 0, "Debe existir al menos un archivo de eventos local."

    match_file = event_files[0]
    match_id = int(match_file.stem)

    # 1. Cargar eventos locales sin credenciales
    df_eventos = cargar_eventos_partido(match_file)
    assert not df_eventos.empty
    assert "period" in df_eventos.columns
    assert "index" in df_eventos.columns
    assert "id" in df_eventos.columns

    # 2. Exportar a directorio temporal
    with tempfile.TemporaryDirectory() as tmpdir:
        meta = {
            "competicion_id": 73,
            "temporada_id": 108,
            "partido_id": match_id,
            "temporada_nombre": "Test Season",
        }
        resumen = exportar_jugadas_partido(
            df_eventos,
            ruta_salida=tmpdir,
            metadatos_partido=meta,
            min_pases=3,
        )

        dir_partido = Path(tmpdir) / "competicion_73" / "temporada_108" / f"partido_{match_id}"
        assert dir_partido.is_dir()
        assert (dir_partido / "manifiesto.json").is_file()

        # Verificar manifiesto
        with open(dir_partido / "manifiesto.json", "r", encoding="utf-8") as f:
            manifiesto = json.load(f)

        assert manifiesto["partido_id"] == match_id
        assert manifiesto["total_eventos_entrada"] == len(df_eventos)
        assert (
            manifiesto["total_eventos_exportados"]
            + len(manifiesto["eventos_no_exportados"])
            == manifiesto["total_eventos_entrada"]
        )

        # Si hay jugadas elegibles, verificar la primera jugada
        if manifiesto["total_jugadas_elegibles"] > 0:
            primera_jugada = dir_partido / "jugada_0001.json"
            assert primera_jugada.is_file()

            df_jugada = leer_jugada(primera_jugada)
            assert list(df_jugada.columns) == list(df_eventos.columns)
            assert len(df_jugada) == manifiesto["jugadas"][0]["total_eventos"]
            assert manifiesto["jugadas"][0]["pases_completados"] >= 3
