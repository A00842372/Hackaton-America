import json
import pytest

from src.data_preparation.corpus import procesar_corpus


def guardar_partido(raiz, temporada, partido, *, valido=True):
    ruta = raiz / "competicion_73" / f"temporada_{temporada}" / "eventos" / f"{partido}.json"
    ruta.parent.mkdir(parents=True, exist_ok=True)
    eventos = {
        f"{temporada}-{i}": {
            "id": f"{temporada}-{i}" if valido else "otro-id",
            "match_id": partido,
            "period": 1,
            "index": i,
            "timestamp": f"00:00:0{i}.000",
            "type": {"name": "Pass"},
            "team": {"name": "A"},
            "possession_team": {"name": "A"},
            "possession": 1,
            "pass": {},
            "location": [float(i), 40.0],
        }
        for i in range(1, 4)
    }
    ruta.write_text(json.dumps(eventos), encoding="utf-8")
    return ruta


@pytest.mark.parametrize("trabajadores", [1, 2])
def test_procesa_todos_los_eventos_sin_depender_de_360_ni_partidos_json(tmp_path, trabajadores):
    entrada = tmp_path / "entrada"
    salida = tmp_path / "salida"
    guardar_partido(entrada, 108, 42)
    guardar_partido(entrada, 235, 42)

    resumen = procesar_corpus(entrada, salida, trabajadores=trabajadores)

    assert resumen["estado"] == "completo"
    assert resumen["partidos_descubiertos"] == 2
    assert resumen["partidos_procesados"] == 2
    assert resumen["errores"] == []
    assert resumen["alcance_totales"] == "partidos_procesados"
    assert resumen["total_eventos_entrada"] == 6
    assert resumen["total_eventos_exportados"] == 6
    assert resumen["total_eventos_no_exportados"] == 0
    assert resumen["total_jugadas_elegibles"] == 2
    for temporada in (108, 235):
        carpeta = salida / "competicion_73" / f"temporada_{temporada}" / "partido_42"
        manifiesto = json.loads((carpeta / "manifiesto.json").read_text(encoding="utf-8"))
        jugada = json.loads((carpeta / "jugada_0001.json").read_text(encoding="utf-8"))
        assert manifiesto["jugadas"][0]["play_id"] == f"c73_t{temporada}_m42_p0001"
        assert jugada["columns"] == list(next(iter(json.loads(
            (entrada / "competicion_73" / f"temporada_{temporada}" / "eventos" / "42.json").read_text()
        ).values())).keys())
    assert json.loads((salida / "resumen_corpus.json").read_text()) == resumen


def test_error_de_un_partido_no_oculta_los_demas_y_marca_lote_incompleto(tmp_path):
    entrada, salida = tmp_path / "entrada", tmp_path / "salida"
    guardar_partido(entrada, 108, 42)
    guardar_partido(entrada, 108, 43, valido=False)

    resumen = procesar_corpus(entrada, salida)

    assert resumen["estado"] == "incompleto"
    assert resumen["partidos_descubiertos"] == 2
    assert resumen["partidos_procesados"] == 1
    assert resumen["partidos_fallidos"] == 1
    assert "43.json" in resumen["errores"][0]["archivo_eventos"]
    assert "id" in resumen["errores"][0]["error"]
    assert not (salida / "competicion_73" / "temporada_108" / "partido_43").exists()
    assert (salida / "competicion_73" / "temporada_108" / "partido_42" / "manifiesto.json").exists()
    assert resumen["total_eventos_entrada"] == 3


def test_reejecucion_no_sobrescribe_y_reemplazo_explicito_es_estable(tmp_path):
    entrada, salida = tmp_path / "entrada", tmp_path / "salida"
    guardar_partido(entrada, 108, 42)
    primera = procesar_corpus(entrada, salida)
    carpeta = salida / "competicion_73" / "temporada_108" / "partido_42"
    jugada_inicial = (carpeta / "jugada_0001.json").read_bytes()

    segunda = procesar_corpus(entrada, salida)

    assert segunda["estado"] == "incompleto"
    assert segunda["partidos_procesados"] == 0
    assert segunda["partidos_fallidos"] == 1
    assert (carpeta / "jugada_0001.json").read_bytes() == jugada_inicial

    tercera = procesar_corpus(entrada, salida, sobrescribir=True)

    assert tercera["estado"] == "completo"
    assert tercera["partidos_procesados"] == 1
    assert tercera["partidos"][0]["total_jugadas_elegibles"] == primera["partidos"][0]["total_jugadas_elegibles"]
    assert (carpeta / "jugada_0001.json").read_bytes() == jugada_inicial


def test_metadatos_opcionales_invalidos_no_bloquean_eventos(tmp_path):
    entrada, salida = tmp_path / "entrada", tmp_path / "salida"
    ruta = guardar_partido(entrada, 108, 42)
    (ruta.parents[1] / "partidos.json").write_text("JSON inválido", encoding="utf-8")

    resumen = procesar_corpus(entrada, salida)

    assert resumen["estado"] == "completo"
    manifiesto = json.loads((salida / "competicion_73" / "temporada_108" / "partido_42" / "manifiesto.json").read_text())
    assert manifiesto["temporada_nombre"] is None
