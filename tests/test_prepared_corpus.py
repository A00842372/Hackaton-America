"""Actualización observable de un corpus preparado en su ubicación original."""

import json
import pytest

from src.data_preparation.corpus import procesar_corpus
from src.data_preparation.export import leer_jugada
from src.data_preparation.prepared_corpus import actualizar_corpus_preparado


def _preparar(tmp_path):
    entrada = tmp_path / "entrada"
    salida = tmp_path / "preparado"
    for partido, patrones in (
        (42, ["From Corner"] * 3 + ["Regular Play"] * 3),
        (43, ["From Free Kick"] * 2 + ["Regular Play"] * 3),
    ):
        ruta = entrada / "competicion_73" / "temporada_108" / "eventos" / f"{partido}.json"
        ruta.parent.mkdir(parents=True, exist_ok=True)
        eventos = {}
        for i, patron in enumerate(patrones, 1):
            id_evento = f"{partido}-{i}"
            eventos[id_evento] = {
                "id": id_evento, "index": i, "period": 1, "match_id": partido,
                "timestamp": f"00:00:{i:02d}.000", "type": {"name": "Pass"},
                "team": {"name": "A"}, "possession_team": {"name": "A"},
                "possession": 1, "play_pattern": {"name": patron},
                "pass": {}, "nested": {"values": [i, None]},
            }
        ruta.write_text(json.dumps(eventos), encoding="utf-8")
    resumen = procesar_corpus(entrada, salida)
    assert resumen["estado"] == "completo"
    return entrada, salida


def _carpeta(salida, partido):
    return salida / "competicion_73" / "temporada_108" / f"partido_{partido}"


@pytest.mark.parametrize("trabajadores", [1, 2])
def test_actualiza_en_sitio_y_repetir_no_cambia_ids_ni_archivos(tmp_path, trabajadores):
    entrada, salida = _preparar(tmp_path)
    carpeta_corta = _carpeta(salida, 43)
    bytes_sin_dividir = (carpeta_corta / "jugada_0001.json").read_bytes()
    for ruta in entrada.glob("competicion_*/temporada_*/eventos/*.json"):
        ruta.unlink()  # La actualización no debe volver a leer eventos originales.

    resumen = actualizar_corpus_preparado(salida, trabajadores=trabajadores)

    assert resumen["estado"] == "completo"
    assert resumen["actualizacion_fases"]["partidos_actualizados"] == 2
    assert resumen["total_jugadas_elegibles"] == 3
    assert resumen["total_cortes_clasificacion"] == 1
    assert resumen["total_eventos_exportados"] == 11
    carpeta_dividida = _carpeta(salida, 42)
    assert not (carpeta_dividida / "jugada_0001.json").exists()
    manifiesto = json.loads((carpeta_dividida / "manifiesto.json").read_text())
    assert [j["play_id"] for j in manifiesto["jugadas"]] == [
        "c73_t108_m42_p0001_f01", "c73_t108_m42_p0001_f02",
    ]
    assert [j["categoria_clasificacion"] for j in manifiesto["jugadas"]] == [
        "balon_parado", "juego_abierto",
    ]
    assert [j["pases_completados"] for j in manifiesto["jugadas"]] == [3, 3]
    partes = [leer_jugada(carpeta_dividida / j["archivo"]) for j in manifiesto["jugadas"]]
    assert [id for parte in partes for id in parte["id"]] == [f"42-{i}" for i in range(1, 7)]
    assert partes[0].iloc[0]["nested"] == {"values": [1, None]}
    assert (carpeta_corta / "jugada_0001.json").read_bytes() == bytes_sin_dividir
    manifiesto_corto = json.loads((carpeta_corta / "manifiesto.json").read_text())
    assert manifiesto_corto["jugadas"][0]["play_id"] == "c73_t108_m43_p0001"
    assert manifiesto_corto["jugadas"][0]["categoria_clasificacion"] == "juego_abierto"

    antes = {
        ruta.relative_to(salida): ruta.read_bytes()
        for ruta in salida.glob("competicion_*/temporada_*/partido_*/*.json")
    }
    repetido = actualizar_corpus_preparado(salida)
    despues = {
        ruta.relative_to(salida): ruta.read_bytes()
        for ruta in salida.glob("competicion_*/temporada_*/partido_*/*.json")
    }
    assert repetido["estado"] == "completo"
    assert repetido["actualizacion_fases"]["partidos_sin_cambios"] == 2
    assert despues == antes


def test_partido_corrupto_se_informa_y_no_se_publica_como_actualizado(tmp_path):
    _, salida = _preparar(tmp_path)
    carpeta_fallida = _carpeta(salida, 43)
    archivo = carpeta_fallida / "jugada_0001.json"
    contenido = json.loads(archivo.read_text())
    contenido["data"].pop()
    archivo.write_text(json.dumps(contenido), encoding="utf-8")
    bytes_corruptos = archivo.read_bytes()
    bytes_manifiesto = (carpeta_fallida / "manifiesto.json").read_bytes()

    resumen = actualizar_corpus_preparado(salida)

    assert resumen["estado"] == "incompleto"
    assert resumen["actualizacion_fases"]["partidos_actualizados"] == 1
    assert resumen["actualizacion_fases"]["partidos_fallidos"] == 1
    assert "partido_43" in resumen["actualizacion_fases"]["errores"][0]["manifiesto"]
    assert archivo.read_bytes() == bytes_corruptos
    assert (carpeta_fallida / "manifiesto.json").read_bytes() == bytes_manifiesto
    assert not list(carpeta_fallida.parent.glob(".actualizacion-*"))
    assert json.loads((salida / "resumen_corpus.json").read_text())["estado"] == "incompleto"


@pytest.mark.parametrize("alteracion", ["total_cortes", "eventos_ids"])
def test_reejecucion_detecta_manifiesto_actualizado_inconsistente(tmp_path, alteracion):
    _, salida = _preparar(tmp_path)
    assert actualizar_corpus_preparado(salida)["estado"] == "completo"
    carpeta = _carpeta(salida, 42)
    ruta = carpeta / "manifiesto.json"
    manifiesto = json.loads(ruta.read_text())
    if alteracion == "total_cortes":
        manifiesto["total_cortes_clasificacion"] = 999
    else:
        del manifiesto["jugadas"][0]["eventos_ids"]
    ruta.write_text(json.dumps(manifiesto), encoding="utf-8")
    bytes_alterados = ruta.read_bytes()

    resumen = actualizar_corpus_preparado(salida)

    assert resumen["estado"] == "incompleto"
    assert resumen["actualizacion_fases"]["partidos_fallidos"] == 1
    assert ruta.read_bytes() == bytes_alterados


def test_primera_ejecucion_rechaza_play_id_que_no_coincide_con_archivo(tmp_path):
    _, salida = _preparar(tmp_path)
    ruta = _carpeta(salida, 42) / "manifiesto.json"
    manifiesto = json.loads(ruta.read_text())
    manifiesto["jugadas"][0]["play_id"] = "c73_t108_m42_p0002"
    ruta.write_text(json.dumps(manifiesto), encoding="utf-8")
    bytes_alterados = ruta.read_bytes()

    resumen = actualizar_corpus_preparado(salida)

    assert resumen["estado"] == "incompleto"
    assert resumen["actualizacion_fases"]["partidos_fallidos"] == 1
    assert "play_id y archivo" in resumen["actualizacion_fases"]["errores"][0]["error"]
    assert ruta.read_bytes() == bytes_alterados
