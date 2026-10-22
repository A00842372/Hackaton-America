"""Salida observable de una clasificación NMF exploratoria sobre jugadas preparadas."""

import json

import pandas as pd

from src.data_preparation.export import exportar_jugadas_partido
from src.classification.exploratory_nmf import clasificar_muestra


def _preparar_partido(corpus, partido_id):
    eventos = []
    for indice in range(1, 8):
        balon_parado = indice <= 4
        auxiliar = indice == 2
        retroceso = partido_id == 44 and indice in (5, 6)
        inicio = float(indice * 10)
        eventos.append({
            "id": f"{partido_id}-{indice}", "index": indice, "period": 1,
            "match_id": partido_id, "timestamp": f"00:00:{indice:02d}.000",
            "type": {"name": "Ball Receipt*" if auxiliar else "Pass"},
            "team": {"name": "A"}, "possession_team": {"name": "A"},
            "possession": 1,
            "play_pattern": {"name": "From Corner" if balon_parado else "Regular Play"},
            "pass": None if auxiliar else (
                {} if indice == 7 else {
                    "end_location": [inicio - 20 if retroceso else inicio + 12, 40.0],
                }
            ),
            "location": [inicio, 40.0],
            "related_events": [f"{partido_id}-1"] if auxiliar else [],
        })
    exportar_jugadas_partido(
        pd.DataFrame(eventos), corpus,
        {"competicion_id": 73, "temporada_id": 108, "partido_id": partido_id},
        dividir_fases_clasificables=True,
    )


def test_clasifica_dos_categorias_con_partido_reservado_y_resultado_reproducible(tmp_path):
    corpus = tmp_path / "preparado"
    for partido_id in (41, 42, 43, 44):
        _preparar_partido(corpus, partido_id)

    salidas = []
    for nombre in ("corrida_1", "corrida_2"):
        salida = tmp_path / nombre
        resultado = clasificar_muestra(
            corpus, salida, n_componentes=2, semilla=17,
            partidos_reservados=[44],
        )
        salidas.append((salida / "resultados.json").read_bytes())
        assert resultado["estado"] == "exploratorio"
        assert resultado["partidos_entrenamiento"] == [41, 42, 43]
        assert resultado["partidos_evaluacion"] == [44]
        assert len(resultado["jugadas"]) == 8
        assert len({fila["play_id"] for fila in resultado["jugadas"]}) == 8
        assert {fila["categoria"] for fila in resultado["jugadas"]} == {
            "balon_parado", "juego_abierto",
        }
        assert {fila["particion"] for fila in resultado["jugadas"] if fila["partido_id"] == 44} == {"evaluacion"}
        sin_cobertura = [fila for fila in resultado["jugadas"] if fila["estado_proyeccion"] == "sin_rasgos_conocidos"]
        assert len(sin_cobertura) == 1
        assert sin_cobertura[0]["partido_id"] == 44
        assert sin_cobertura[0]["categoria"] == "juego_abierto"
        assert sin_cobertura[0]["componente_dominante"] is None
        assert sin_cobertura[0]["error_reconstruccion"] is None
        assert sin_cobertura[0]["cobertura_vocabulario"] == 0
        assert all(fila["componente_dominante"] in fila["pesos"]
                   for fila in resultado["jugadas"] if fila not in sin_cobertura)
        assert all(len(fila["pesos"]) == 2 for fila in resultado["jugadas"])
        assert all(fila["error_reconstruccion"] >= 0 for fila in resultado["jugadas"]
                   if fila not in sin_cobertura)
        assert all((corpus / fila["manifiesto"]).is_file() for fila in resultado["jugadas"])
        for categoria, modelo in resultado["modelos"].items():
            assert modelo["n_componentes"] == 2
            assert all(id_.startswith(categoria + "_") for id_ in modelo["componentes"])
            vocabulario = modelo["vocabulario"]
            assert "Ball_Receipt" not in " ".join(vocabulario)
            assert "retroceso" not in " ".join(vocabulario)
            assert "lateral" not in " ".join(vocabulario)
            if categoria == "juego_abierto":
                assert any(ngram.endswith("→ propio:Pass") for ngram in vocabulario)
            assert (salida / modelo["archivo_modelo"]).is_file()
        assert json.loads((salida / "resultados.json").read_text()) == resultado

    assert salidas[0] == salidas[1]
