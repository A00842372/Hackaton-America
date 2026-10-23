"""Comparación espacial observable sobre jugadas reflejadas entre bandas."""

import json

import pandas as pd
import pytest

from src.classification.exploratory_nmf import comparar_representaciones
from src.data_preparation.export import exportar_jugadas_partido


def _partido(corpus, partido_id, banda, direccion, *, orientacion_invertida=False):
    eventos = []
    for indice in range(1, 9):
        patron = "From Corner" if indice <= 4 else "Regular Play"
        tipo = "Pressure" if indice == 4 else "Shot" if indice == 8 else "Pass"
        x = (30.0 + indice * 10) if partido_id <= 42 else (20.0 + indice * 8)
        if indice == 4:
            x = 20.0
        if tipo == "Shot" and orientacion_invertida and direccion == "right_to_left":
            x = 80.0
        avance = 12 if partido_id <= 42 else 35
        destino = [x + avance, banda + (5 if banda < 40 else -5)]
        origen_y = 40.0 if partido_id >= 43 and tipo != "Pressure" else banda
        eventos.append({
            "id": f"{partido_id}-{indice}", "index": indice, "period": 1,
            "match_id": partido_id, "timestamp": f"00:00:{indice:02d}.000",
            "type": {"name": tipo}, "team": {"name": "B" if indice == 4 else "A"},
            "possession_team": {"name": "A"}, "possession": 1,
            "play_pattern": {"name": patron},
            "attacking_direction": direccion,
            "location": [x, origen_y],
            "pass": ({"end_location": destino} if indice != 7 else {}) if tipo == "Pass" else None,
            "shot": {"end_location": [0.0 if orientacion_invertida and direccion == "right_to_left"
                                      else 120.0, 40.0]} if tipo == "Shot" else None,
        })
    exportar_jugadas_partido(
        pd.DataFrame(eventos), corpus,
        {"competicion_id": 73, "temporada_id": 108, "partido_id": partido_id},
        dividir_fases_clasificables=True,
    )


def test_compara_zonas_reflejadas_sin_invertir_x_ni_modificar_eventos(tmp_path):
    corpus = tmp_path / "preparado"
    for partido_id, banda, direccion in (
        (41, 20.0, "left_to_right"), (42, 60.0, "right_to_left"),
        (43, 10.0, "left_to_right"), (44, 70.0, "right_to_left"),
    ):
        _partido(corpus, partido_id, banda, direccion)
    originales = {p: p.read_bytes() for p in corpus.rglob("jugada_*.json")}

    comparacion = comparar_representaciones(
        corpus, tmp_path / "comparacion", n_componentes=1, semilla=17,
        partidos_reservados=[44], zonas_x=6, zonas_y=4, peso_espacial=1.5,
    )

    salida = tmp_path / "comparacion"
    base = json.loads((salida / "base" / "resultados.json").read_text())
    espacial = json.loads((salida / "espacial" / "resultados.json").read_text())
    assert {j["play_id"] for j in base["jugadas"]} == {j["play_id"] for j in espacial["jugadas"]}
    assert len(espacial["jugadas"]) == 8
    assert comparacion["partidos_evaluacion"] == [44]
    assert set(comparacion["variantes"]) == {"base", "espacial"}
    assert comparacion["jugadas"] == 8
    assert espacial["parametros"]["cuadricula"] == {"zonas_x": 6, "zonas_y": 4}
    assert espacial["parametros"]["peso_espacial"] == 1.5
    assert espacial["verificacion_orientacion"]["convencion"] == "x_hacia_120_por_equipo"
    assert set(espacial["verificacion_orientacion"]["tiros_por_direccion"]) == {
        "left_to_right", "right_to_left",
    }
    assert any("destino=sin_dato" in ngrama
               for modelo in espacial["modelos"].values() for ngrama in modelo["vocabulario"])
    assert any("rival:Pressure:origen=x5_y3" in ngrama
               for ngrama in espacial["modelos"]["balon_parado"]["vocabulario"])
    for categoria in ("balon_parado", "juego_abierto"):
        assert set(base["modelos"][categoria]["vocabulario"]) == {
            rasgo for rasgo in espacial["modelos"][categoria]["vocabulario"]
            if not rasgo.startswith("espacio:")
        }

    filas = {(j["partido_id"], j["categoria"]): j for j in espacial["jugadas"]}
    for categoria in ("balon_parado", "juego_abierto"):
        izquierda, derecha = filas[41, categoria], filas[42, categoria]
        assert izquierda["banda_original"] == "y_baja"
        assert derecha["banda_original"] == "y_alta"
        assert izquierda["reflejada"] is False
        assert derecha["reflejada"] is True
        assert izquierda["pesos"] == pytest.approx(derecha["pesos"])
        assert izquierda["error_reconstruccion"] == pytest.approx(derecha["error_reconstruccion"])
        centro_izquierda, centro_derecha = filas[43, categoria], filas[44, categoria]
        assert centro_izquierda["banda_original"] == "y_baja"
        assert centro_derecha["banda_original"] == "y_alta"
        assert centro_izquierda["pesos"] == pytest.approx(centro_derecha["pesos"])
    assert {p: p.read_bytes() for p in originales} == originales


def test_rechaza_orientacion_fisica_sin_convencion_confirmada(tmp_path):
    corpus = tmp_path / "preparado"
    for partido_id, banda, direccion in (
        (41, 20.0, "left_to_right"), (42, 60.0, "right_to_left"),
        (43, 10.0, "left_to_right"), (44, 70.0, "right_to_left"),
    ):
        _partido(corpus, partido_id, banda, direccion, orientacion_invertida=True)

    salida = tmp_path / "comparacion"
    with pytest.raises(ValueError, match="No se confirmó"):
        comparar_representaciones(corpus, salida, n_componentes=1, partidos_reservados=[44])
    assert not salida.exists()
