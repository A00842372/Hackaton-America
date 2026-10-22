"""La exportación con corte opcional conserva trazabilidad y cobertura."""

import json

import pandas as pd

from src.data_preparation.export import exportar_jugadas_partido, leer_jugada
from src.data_preparation.corpus import procesar_corpus


META = {"competicion_id": 73, "temporada_id": 108, "partido_id": 42}


def _eventos(patrones, *, desconocido=None, rival=None):
    filas = []
    for i, patron in enumerate(patrones, 1):
        nombre = "Unknown Phase" if i == desconocido else patron
        equipo = "B" if i == rival else "A"
        filas.append({
            "id": f"e{i}", "index": i, "period": 1,
            "match_id": 42, "timestamp": f"00:00:{i:02d}.000",
            "type": {"name": "Pass" if i != rival else "Pressure"},
            "team": {"name": equipo}, "possession_team": {"name": "A"},
            "possession": 1,
            "play_pattern": {"name": nombre},
            "pass": {} if i != rival else None,
            "location": [float(i), 40.0],
            "nested": {"values": [i, None]},
        })
    return pd.DataFrame(filas)


def _exportar(tmp_path, eventos):
    manifiesto = exportar_jugadas_partido(
        eventos, tmp_path, META, dividir_fases_clasificables=True,
    )
    carpeta = tmp_path / "competicion_73" / "temporada_108" / "partido_42"
    return manifiesto, carpeta


def test_tres_fases_validas_conservan_eventos_y_procedencia(tmp_path):
    eventos = _eventos(
        ["From Corner"] * 3 + ["Regular Play", "From Free Kick",
                               "Regular Play", "Regular Play"] + ["From Throw In"] * 3,
        rival=5,
    )
    manifiesto, carpeta = _exportar(tmp_path, eventos)

    assert [j["play_id"] for j in manifiesto["jugadas"]] == [
        "c73_t108_m42_p0001_f01", "c73_t108_m42_p0001_f02",
        "c73_t108_m42_p0001_f03",
    ]
    assert [j["categoria_clasificacion"] for j in manifiesto["jugadas"]] == [
        "balon_parado", "juego_abierto", "balon_parado",
    ]
    assert [j["pases_completados"] for j in manifiesto["jugadas"]] == [3, 3, 3]
    assert manifiesto["total_jugadas_origen_divididas"] == 1
    assert manifiesto["total_cortes_clasificacion"] == 2
    assert [j["ordinal_tramo"] for j in manifiesto["jugadas"]] == [1, 2, 3]
    assert all(j["play_id_origen"] == "c73_t108_m42_p0001" for j in manifiesto["jugadas"])
    assert manifiesto["jugadas"][1]["corte_antes"] == {
        "evento_anterior_id": "e3", "evento_siguiente_id": "e4",
    }
    partes = [leer_jugada(carpeta / j["archivo"]) for j in manifiesto["jugadas"]]
    assert [id for parte in partes for id in parte["id"]] == list(eventos["id"])
    assert all(list(parte.columns) == list(eventos.columns) for parte in partes)
    assert partes[1].iloc[0]["nested"] == {"values": [4, None]}
    assert manifiesto["total_eventos_exportados"] == manifiesto["total_eventos_entrada"] == 10
    assert json.loads((carpeta / "manifiesto.json").read_text()) == manifiesto


def test_tramo_corto_conserva_id_y_elige_fase_mayoritaria_sin_perder_desconocidos(tmp_path):
    eventos = _eventos(["From Free Kick"] * 2 + ["Regular Play"] * 5, desconocido=4)
    manifiesto, carpeta = _exportar(tmp_path, eventos)

    jugada = manifiesto["jugadas"][0]
    assert manifiesto["total_jugadas_elegibles"] == 1
    assert jugada["play_id"] == "c73_t108_m42_p0001"
    assert jugada["archivo"] == "jugada_0001.json"
    assert jugada["categoria_clasificacion"] == "juego_abierto"
    assert jugada["motivo_corte"] == "tramo_con_menos_de_min_pases"
    assert list(leer_jugada(carpeta / jugada["archivo"])["id"]) == list(eventos["id"])
    assert manifiesto["patrones_desconocidos"] == [{
        "id": "e4", "play_pattern": "Unknown Phase", "categoria_asignada": "juego_abierto",
    }]


def test_empate_de_acciones_favorece_primera_fase(tmp_path):
    eventos = _eventos(["From Corner"] * 2 + ["Regular Play"] * 2)
    manifiesto, _ = _exportar(tmp_path, eventos)
    assert manifiesto["jugadas"][0]["categoria_clasificacion"] == "balon_parado"


def test_cambio_entre_patrones_de_una_misma_categoria_no_corta(tmp_path):
    eventos = _eventos(["From Corner"] * 3 + ["From Free Kick"] * 3)
    manifiesto, _ = _exportar(tmp_path, eventos)
    assert [j["play_id"] for j in manifiesto["jugadas"]] == ["c73_t108_m42_p0001"]
    assert manifiesto["jugadas"][0]["categoria_clasificacion"] == "balon_parado"


def test_preparacion_futura_registra_opcion_y_respeta_minimo_tres(tmp_path):
    eventos = _eventos(["From Corner"] * 2 + ["Regular Play"] * 2)
    origen = tmp_path / "origen" / "competicion_73" / "temporada_108" / "eventos" / "42.json"
    origen.parent.mkdir(parents=True)
    origen.write_text(json.dumps({fila["id"]: fila for fila in eventos.to_dict("records")}))

    resumen = procesar_corpus(
        tmp_path / "origen", tmp_path / "salida", min_pases=2,
        dividir_fases_clasificables=True,
    )

    assert resumen["estado"] == "completo"
    assert resumen["total_jugadas_elegibles"] == 1
    assert resumen["total_cortes_clasificacion"] == 0
    assert resumen["parametros"]["dividir_fases_clasificables"] is True
    manifiesto = json.loads((tmp_path / "salida" / "competicion_73" / "temporada_108"
                            / "partido_42" / "manifiesto.json").read_text())
    assert manifiesto["parametros"]["min_pases_por_fase"] == 3
    assert manifiesto["jugadas"][0]["play_id"] == "c73_t108_m42_p0001"


def test_minimo_general_mayor_no_eleva_el_umbral_de_cada_fase(tmp_path):
    eventos = _eventos(["From Corner"] * 3 + ["Regular Play"] * 3)
    manifiesto = exportar_jugadas_partido(
        eventos, tmp_path, META, min_pases=5, dividir_fases_clasificables=True,
    )
    assert [j["pases_completados"] for j in manifiesto["jugadas"]] == [3, 3]
    assert manifiesto["parametros"]["min_pases_por_fase"] == 3
