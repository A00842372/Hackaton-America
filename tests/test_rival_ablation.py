"""Salida de la ablación rival sobre jugadas exportadas con lances vinculados."""

import json

import pandas as pd

from src.classification.rival_ablation import comparar_rivales
from src.classification.features import ConfiguracionEspacial, representar_jugada
from src.data_preparation.export import exportar_jugadas_partido


def _partido(corpus, partido_id):
    eventos = []

    def agregar(tipo, equipo, poseedor, posesion, segundo, patron, relacionados=None):
        indice = len(eventos) + 1
        x = 20.0 + indice * 4
        evento = {
            "id": f"{partido_id}-{indice}", "index": indice, "period": 1,
            "match_id": partido_id, "timestamp": f"00:00:{segundo:06.3f}",
            "type": {"name": tipo}, "team": {"name": equipo},
            "possession_team": {"name": poseedor}, "possession": posesion,
            "play_pattern": {"name": patron}, "location": [x, 20.0],
            "related_events": relacionados or [],
            "pass": {"end_location": [x + (12 if partido_id % 2 else 34), 20.0]}
            if tipo == "Pass" else None,
        }
        eventos.append(evento)
        return evento["id"]

    for desfase, patron, base in ((0, "From Corner", 1), (20, "Regular Play", 4)):
        for segundo in (1.0, 2.0, 3.0):
            agregar("Pass", "A", "A", base, desfase + segundo, patron)
        agregar("Pressure", "B", "A", base, desfase + 3.2, patron)
        duelo_id = f"{partido_id}-{len(eventos) + 1}"
        recuperacion_id = f"{partido_id}-{len(eventos) + 2}"
        agregar("Duel", "B", "B", base + 1, desfase + 3.5, patron, [recuperacion_id])
        agregar("Ball Recovery", "B", "B", base + 1, desfase + 3.5, patron, [duelo_id])
        agregar("Pressure", "B", "B", base + 1, desfase + 3.5, patron, [recuperacion_id])
        agregar("Carry", "B", "B", base + 1, desfase + 3.7, patron, [recuperacion_id])
        for segundo in (4.0, 5.0, 6.0):
            agregar("Pass", "A", "A", base + 2, desfase + segundo, patron)
    exportar_jugadas_partido(
        pd.DataFrame(eventos), corpus,
        {"competicion_id": 73, "temporada_id": 108, "partido_id": partido_id},
        dividir_fases_clasificables=True,
    )


def test_compara_mismas_jugadas_y_publica_influencia_y_artefactos(tmp_path):
    corpus = tmp_path / "corpus"
    for partido_id in (41, 42, 43, 44):
        _partido(corpus, partido_id)
    salida = tmp_path / "ablacion"
    comparacion = comparar_rivales(
        corpus, salida, n_componentes=2, semilla=17, partidos_reservados=[44],
    )
    con = json.loads((salida / "con_rivales" / "resultados.json").read_text())
    sin = json.loads((salida / "sin_rivales" / "resultados.json").read_text())
    assert comparacion == json.loads((salida / "comparacion.json").read_text())
    assert comparacion["jugadas"] == len(con["jugadas"]) == len(sin["jugadas"]) == 8
    assert [(j["play_id"], j["particion"]) for j in con["jugadas"]] == [
        (j["play_id"], j["particion"]) for j in sin["jugadas"]
    ]
    assert comparacion["partidos_evaluacion"] == [44]
    for categoria in ("balon_parado", "juego_abierto"):
        vocabulario = con["modelos"][categoria]["vocabulario"]
        assert any("rival_interrupcion:Pressure" in ng for ng in vocabulario)
        assert any("rival_recuperacion:Ball_Recovery" in ng for ng in vocabulario)
        assert any("rival_control:Carry" in ng for ng in vocabulario)
        assert not any("rival_control:Duel" in ng or "rival_recuperacion:Duel" in ng
                       for ng in vocabulario)
        assert not any("rival_control:Pressure" in ng for ng in vocabulario)
        assert not any("rival_" in ng for ng in sin["modelos"][categoria]["vocabulario"])
        diagnostico = comparacion["por_categoria"][categoria]
        assert diagnostico["ngramas_no_rivales_compartidos"] > 0
        assert all(f["ngramas_destacados"] and f["ejemplos"] for f in diagnostico["con_rivales"])
        assert all("posible_dominio_interrupcion" in f and "posible_artefacto_registro" in f
                   for f in diagnostico["con_rivales"])
        assert all(par["comparable"] and "ngramas_nuevos_destacados" in par
                   and "ejemplos_nuevos" in par
                   for par in diagnostico["componentes_emparejados"])
    assert {c["play_id"] for c in comparacion["cambios_contribuciones"]} == {
        j["play_id"] for j in con["jugadas"]
    }
    assert all(c["componentes"] and all(p["cambio"] is not None for p in c["componentes"])
               for c in comparacion["cambios_contribuciones"])


def test_recuperacion_rival_conserva_marco_protagonista_y_un_solo_lance():
    def evento(id_, tipo, equipo, poseedor, x, y, relacionados=()):
        return {
            "id": id_, "type": {"name": tipo}, "team": {"name": equipo},
            "possession_team": {"name": poseedor}, "location": [x, y],
            "timestamp": "00:00:01.000" if id_ in ("duelo", "recuperacion")
            else "00:00:02.000",
            "related_events": list(relacionados),
        }

    eventos = [
        evento("a", "Pass", "A", "A", 20, 20),
        evento("presion", "Pressure", "B", "A", 30, 60),
        evento("duelo", "Duel", "B", "B", 40, 60, ("recuperacion",)),
        evento("recuperacion", "Ball Recovery", "B", "B", 40, 60, ("duelo",)),
        evento("b", "Pass", "A", "A", 50, 20),
    ]
    con, _ = representar_jugada(eventos, "A", ConfiguracionEspacial())
    sin, _ = representar_jugada(eventos, "A", ConfiguracionEspacial(), modo_rivales="sin_rivales")
    assert any("rival_interrupcion:Pressure:origen=x4_y1" in ng for ng in con)
    assert any("rival_recuperacion:Ball_Recovery:origen=x4_y1" in ng for ng in con)
    assert not any("Duel" in ng for ng in con)
    assert not any("rival_" in ng for ng in sin)


def test_desposesion_y_duelo_entre_equipos_son_una_recuperacion():
    eventos = [
        {"id": "p1", "type": {"name": "Pass"}, "team": {"name": "A"},
         "possession_team": {"name": "A"}},
        {"id": "d", "type": {"name": "Dispossessed"}, "team": {"name": "A"},
         "possession_team": {"name": "B"}, "timestamp": "00:00:01.000",
         "related_events": ["u"]},
        {"id": "u", "type": {"name": "Duel"}, "team": {"name": "B"},
         "possession_team": {"name": "B"}, "timestamp": "00:00:01.000",
         "related_events": ["d"]},
        {"id": "p2", "type": {"name": "Pass"}, "team": {"name": "A"},
         "possession_team": {"name": "A"}},
    ]
    ngramas, _ = representar_jugada(eventos, "A")
    assert ngramas == {
        "n2:propio:Pass → rival_recuperacion:Duel": 1,
        "n2:rival_recuperacion:Duel → propio:Pass": 1,
        "n3:propio:Pass → rival_recuperacion:Duel → propio:Pass": 1,
    }


def test_primera_accion_rival_es_recuperacion_aunque_registro_propio_anticipo_posesion():
    eventos = [
        {"type": {"name": "Pass"}, "team": {"name": "A"},
         "possession_team": {"name": "A"}},
        {"type": {"name": "Miscontrol"}, "team": {"name": "A"},
         "possession_team": {"name": "B"}},
        {"type": {"name": "Ball Recovery"}, "team": {"name": "B"},
         "possession_team": {"name": "B"}},
        {"type": {"name": "Pass"}, "team": {"name": "A"},
         "possession_team": {"name": "A"}},
    ]
    ngramas, _ = representar_jugada(eventos, "A")
    assert any("propio:Miscontrol → rival_recuperacion:Ball_Recovery" in ng for ng in ngramas)


def test_dribbled_past_vinculado_al_regate_del_otro_equipo_no_duplica_lance():
    eventos = [
        {"id": "p1", "type": {"name": "Pass"}, "team": {"name": "A"},
         "possession_team": {"name": "A"}},
        {"id": "d", "type": {"name": "Dribble"}, "team": {"name": "A"},
         "possession_team": {"name": "A"}, "timestamp": "00:00:01.000",
         "related_events": ["dp"]},
        {"id": "dp", "type": {"name": "Dribbled Past"}, "team": {"name": "B"},
         "possession_team": {"name": "A"}, "timestamp": "00:00:01.000",
         "related_events": ["d"]},
        {"id": "p2", "type": {"name": "Pass"}, "team": {"name": "A"},
         "possession_team": {"name": "A"}},
    ]
    ngramas, _ = representar_jugada(eventos, "A")
    assert ngramas == {
        "n2:propio:Pass → propio:Dribble": 1,
        "n2:propio:Dribble → propio:Pass": 1,
        "n3:propio:Pass → propio:Dribble → propio:Pass": 1,
    }


def test_contactos_vinculados_en_cadena_forman_un_lance_independiente_del_orden():
    eventos = [
        {"id": "p1", "type": {"name": "Pass"}, "team": {"name": "A"},
         "possession_team": {"name": "A"}},
        {"id": "a", "type": {"name": "Duel"}, "team": {"name": "B"},
         "possession_team": {"name": "B"}, "timestamp": "00:00:01.000",
         "related_events": ["b"]},
        {"id": "c", "type": {"name": "Dispossessed"}, "team": {"name": "A"},
         "possession_team": {"name": "B"}, "timestamp": "00:00:01.000",
         "related_events": []},
        {"id": "b", "type": {"name": "Ball Recovery"}, "team": {"name": "B"},
         "possession_team": {"name": "B"}, "timestamp": "00:00:01.000",
         "related_events": ["c"]},
        {"id": "p2", "type": {"name": "Pass"}, "team": {"name": "A"},
         "possession_team": {"name": "A"}},
    ]
    ngramas, _ = representar_jugada(eventos, "A")
    assert ngramas == {
        "n2:propio:Pass → rival_recuperacion:Ball_Recovery": 1,
        "n2:rival_recuperacion:Ball_Recovery → propio:Pass": 1,
        "n3:propio:Pass → rival_recuperacion:Ball_Recovery → propio:Pass": 1,
    }
