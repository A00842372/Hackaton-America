"""Comparación de posición observada en la salida del piloto NMF."""

import json

import pandas as pd

from src.classification.exploratory_nmf import comparar_cronologia_y_ritmo, comparar_posiciones
from src.data_preparation.export import exportar_jugadas_partido


def test_compara_ngramas_por_tercios_con_la_misma_cobertura_y_particion(tmp_path):
    corpus = tmp_path / "corpus"
    for partido_id in (41, 42, 43, 44):
        eventos = []
        for indice in range(1, 19):
            categoria = "From Corner" if indice <= 9 else "Regular Play"
            eventos.append({
                "id": f"{partido_id}-{indice}", "index": indice, "period": 1,
                "match_id": partido_id, "timestamp": f"00:00:{indice:02d}.000",
                "type": {"name": "Pass"}, "team": {"name": "A"},
                "possession_team": {"name": "A"}, "possession": 1,
                "play_pattern": {"name": categoria},
                "location": [float(indice * 3), 20.0],
                "pass": {"end_location": [float(indice * 3 + 10), 20.0]},
            })
        exportar_jugadas_partido(
            pd.DataFrame(eventos), corpus,
            {"competicion_id": 73, "temporada_id": 108, "partido_id": partido_id},
            dividir_fases_clasificables=True,
        )

    comparacion = comparar_posiciones(
        corpus, tmp_path / "comparacion", n_componentes=2,
        partidos_reservados=[44], semilla=17,
    )
    assert comparacion["jugadas"] == 8
    assert comparacion["partidos_evaluacion"] == [44]
    assert set(comparacion["variantes"]) == {"base", "posicional"}
    base = json.loads((tmp_path / "comparacion" / "base" / "resultados.json").read_text())
    posicional = json.loads((tmp_path / "comparacion" / "posicional" / "resultados.json").read_text())
    assert [(j["play_id"], j["particion"]) for j in base["jugadas"]] == [
        (j["play_id"], j["particion"]) for j in posicional["jugadas"]
    ]
    for categoria in ("balon_parado", "juego_abierto"):
        vocabulario = posicional["modelos"][categoria]["vocabulario"]
        for tercio in ("inicio", "medio", "final"):
            assert any(f"posicion:{tercio}:n2:" in rasgo for rasgo in vocabulario)
            assert any(f"posicion:{tercio}:n3:" in rasgo for rasgo in vocabulario)
        assert set(base["modelos"][categoria]["vocabulario"]) < set(vocabulario)
    assert (tmp_path / "comparacion" / "comparacion.json").is_file()


def test_compara_ritmo_sin_inventar_intervalos_para_marcas_ausentes_o_compartidas(tmp_path):
    corpus = tmp_path / "corpus"
    for partido_id in (41, 42, 43, 44):
        eventos = []
        for indice in range(1, 19):
            eventos.append({
                "id": f"{partido_id}-{indice}", "index": indice, "period": 1,
                "match_id": partido_id, "timestamp": f"00:00:{indice:02d}.000",
                "type": {"name": "Pass"}, "team": {"name": "A"},
                "possession_team": {"name": "A"}, "possession": 1,
                "play_pattern": {"name": "From Corner" if indice <= 9 else "Regular Play"},
                "pass": {},
            })
        exportar_jugadas_partido(
            pd.DataFrame(eventos), corpus,
            {"competicion_id": 73, "temporada_id": 108, "partido_id": partido_id},
            dividir_fases_clasificables=True,
        )
    for partido_id in (41, 42, 43):
        ruta = next((corpus / "competicion_73" / "temporada_108" /
                     f"partido_{partido_id}").glob("jugada_*_f01.json"))
        jugada = json.loads(ruta.read_text(encoding="utf-8"))
        columna = jugada["columns"].index("timestamp")
        for fila in jugada["data"]:
            fila[columna] = None if partido_id == 41 else "00:00:01.000"
        ruta.write_text(json.dumps(jugada), encoding="utf-8")

    comparacion = comparar_cronologia_y_ritmo(
        corpus, tmp_path / "comparacion", n_componentes=2,
        partidos_reservados=[44], semilla=17,
    )
    assert comparacion["jugadas"] == 8
    assert set(comparacion["variantes"]) == {"base", "posicional", "temporal"}
    resultados = {
        nombre: json.loads((tmp_path / "comparacion" / nombre / "resultados.json").read_text())
        for nombre in comparacion["variantes"]
    }
    referencia = [(fila["play_id"], fila["particion"]) for fila in resultados["base"]["jugadas"]]
    assert all(referencia == [(fila["play_id"], fila["particion"]) for fila in salida["jugadas"]]
               for salida in resultados.values())
    for salida in resultados.values():
        for fila in salida["jugadas"]:
            assert fila["componente_dominante"] in fila["pesos"]
            assert (corpus / fila["manifiesto"]).parent.joinpath(fila["archivo"]).is_file()
    for categoria in ("balon_parado", "juego_abierto"):
        base = set(resultados["base"]["modelos"][categoria]["vocabulario"])
        temporal = set(resultados["temporal"]["modelos"][categoria]["vocabulario"])
        if categoria == "balon_parado":
            assert temporal == base
        else:
            assert base < temporal
            assert any(rasgo.startswith("ritmo:n3:") for rasgo in temporal)
    assert resultados["temporal"]["parametros"]["umbrales_ritmo_segundos"] == {
        "rapido_hasta": 1, "medio_hasta": 3,
    }
