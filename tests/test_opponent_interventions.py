import pandas as pd
import pytest

from src.data_preparation.export import exportar_jugadas_partido, leer_jugada
from src.data_preparation.plays import extraer_secuencias_candidatas


def evento(index, segundo, tipo, poseedor, posesion, *, duracion=0, relacionados=None, **campos):
    return {
        "id": f"e{index}",
        "match_id": 42,
        "period": 1,
        "index": index,
        "timestamp": f"00:00:{segundo:06.3f}",
        "duration": duracion,
        "type": {"name": tipo},
        "team": {"name": poseedor},
        "possession_team": {"name": poseedor},
        "possession": posesion,
        "related_events": relacionados or [],
        **campos,
    }


def ids(secuencias):
    return [list(secuencia["id"]) for secuencia in secuencias]


def test_limites_inclusivos_absorben_lance_relacionado_y_exportan_una_sola_jugada(tmp_path):
    eventos = pd.DataFrame([
        evento(1, 0, "Pass", "A", 1, **{"pass": {}}),
        evento(2, 1, "Pass", "A", 1, duracion=1, **{"pass": {}}),
        evento(3, 2, "Duel", "B", 2, relacionados=["e4"]),
        evento(4, 2, "Ball Recovery", "B", 2, relacionados=["e3"]),
        evento(5, 3, "Pass", "B", 2, duracion=1, relacionados=["e6"], **{"pass": {}}),
        evento(6, 4, "Ball Receipt*", "B", 2, relacionados=["e5"]),
        evento(7, 4, "Carry", "B", 2, duracion=1),
        evento(8, 5, "Pass", "A", 3, **{"pass": {}}),
    ])

    manifiesto = exportar_jugadas_partido(
        eventos, tmp_path, {"competicion_id": 73, "temporada_id": 108, "partido_id": 42}
    )
    carpeta = tmp_path / "competicion_73" / "temporada_108" / "partido_42"

    assert manifiesto["total_jugadas_elegibles"] == 1
    assert manifiesto["total_candidatas_descartadas"] == 0
    assert manifiesto["total_eventos_exportados"] == 8
    assert manifiesto["eventos_no_exportados"] == []
    assert manifiesto["jugadas"][0]["pases_completados"] == 3
    assert manifiesto["jugadas"][0]["posesiones_origen"] == [1, 2, 3]
    assert manifiesto["jugadas"][0]["intervenciones_rivales_absorbidas"] == [
        "e3", "e4", "e5", "e6", "e7"
    ]
    assert ids([leer_jugada(carpeta / "jugada_0001.json")]) == [[f"e{i}" for i in range(1, 9)]]
    assert not (carpeta / "jugada_0002.json").exists()


@pytest.mark.parametrize("retorno", [4.0, 4.001])
def test_limite_temporal_exactamente_tres_segundos_es_inclusivo(retorno):
    eventos = pd.DataFrame([
        evento(1, 0, "Pass", "A", 1, duracion=1),
        evento(2, 1, "Ball Recovery", "B", 2),
        evento(3, retorno, "Ball Recovery", "A", 3),
    ])

    secuencias, _ = extraer_secuencias_candidatas(eventos)

    assert ids(secuencias) == (
        [["e1", "e2", "e3"]] if retorno == 4.0 else [["e1"], ["e2"], ["e3"]]
    )


def test_cuatro_acciones_rivales_impiden_union_aunque_sean_rapidas():
    eventos = pd.DataFrame([
        evento(1, 0, "Pass", "A", 1, duracion=1),
        evento(2, 1, "Ball Recovery", "B", 2),
        evento(3, 1.5, "Pass", "B", 2),
        evento(4, 2, "Carry", "B", 2),
        evento(5, 2.5, "Dribble", "B", 2),
        evento(6, 3, "Ball Recovery", "A", 3),
    ])

    secuencias, _ = extraer_secuencias_candidatas(eventos)

    assert ids(secuencias) == [["e1"], ["e2", "e3", "e4", "e5"], ["e6"]]


def test_duracion_rival_may_superar_limite_aunque_timestamps_no_lo_hagan():
    eventos = pd.DataFrame([
        evento(1, 0, "Pass", "A", 1, duracion=1),
        evento(2, 1, "Pass", "B", 2, duracion=3.1),
        evento(3, 3, "Ball Recovery", "A", 3),
    ])

    secuencias, _ = extraer_secuencias_candidatas(eventos)

    assert ids(secuencias) == [["e1"], ["e2"], ["e3"]]


@pytest.mark.parametrize("cierre", ["salida", "detencion", "periodo"])
def test_no_absorbe_si_hay_salida_detencion_o_cambio_de_periodo(cierre):
    rival = evento(2, 1, "Pass", "B", 2)
    retorno = evento(3, 2, "Ball Recovery", "A", 3)
    if cierre == "salida":
        rival["out"] = True
    elif cierre == "detencion":
        rival["type"] = {"name": "Foul Committed"}
        rival["foul_committed"] = {}
    else:
        retorno["period"] = 2
    eventos = pd.DataFrame([evento(1, 0, "Pass", "A", 1, duracion=1), rival, retorno])

    secuencias, _ = extraer_secuencias_candidatas(eventos)

    assert ids(secuencias) == [["e1"], ["e2"], ["e3"]]


def test_falta_emparejada_con_cambio_de_poseedor_sigue_siendo_barrera():
    eventos = pd.DataFrame([
        evento(1, 0, "Pass", "A", 1, duracion=1),
        evento(2, 1, "Foul Committed", "B", 2),
        evento(3, 1, "Foul Won", "A", 3),
        evento(4, 2, "Pass", "A", 3, **{"pass": {"type": {"name": "Free Kick"}}}),
    ])

    secuencias, _ = extraer_secuencias_candidatas(eventos)

    assert ids(secuencias) == [["e1"], ["e2"], ["e3"], ["e4"]]


def test_intervenciones_consecutivas_se_absorben_una_sola_vez(tmp_path):
    eventos = pd.DataFrame([
        evento(1, 0, "Pass", "A", 1, **{"pass": {}}),
        evento(2, 1, "Ball Recovery", "B", 2),
        evento(3, 2, "Pass", "A", 3, **{"pass": {}}),
        evento(4, 3, "Ball Recovery", "B", 4),
        evento(5, 4, "Pass", "A", 5, **{"pass": {}}),
    ])

    manifiesto = exportar_jugadas_partido(
        eventos, tmp_path, {"competicion_id": 73, "temporada_id": 108, "partido_id": 42}
    )
    carpeta = tmp_path / "competicion_73" / "temporada_108" / "partido_42"

    assert manifiesto["total_jugadas_elegibles"] == 1
    assert manifiesto["jugadas"][0]["intervenciones_rivales_absorbidas"] == ["e2", "e4"]
    assert ids([leer_jugada(carpeta / "jugada_0001.json")]) == [["e1", "e2", "e3", "e4", "e5"]]
    assert manifiesto["total_eventos_exportados"] == 5
    assert not (carpeta / "jugada_0002.json").exists()


def test_intervencion_larga_conserva_jugada_rival_elegible(tmp_path):
    eventos = pd.DataFrame([
        evento(1, 0, "Pass", "A", 1, **{"pass": {}}),
        evento(2, 1, "Pass", "A", 1, **{"pass": {}}),
        evento(3, 2, "Pass", "A", 1, duracion=1, **{"pass": {}}),
        evento(4, 3, "Pass", "B", 2, **{"pass": {}}),
        evento(5, 5, "Pass", "B", 2, **{"pass": {}}),
        evento(6, 7, "Pass", "B", 2, **{"pass": {}}),
        evento(7, 9, "Pass", "A", 3, **{"pass": {}}),
        evento(8, 10, "Pass", "A", 3, **{"pass": {}}),
        evento(9, 11, "Pass", "A", 3, **{"pass": {}}),
    ])

    manifiesto = exportar_jugadas_partido(
        eventos, tmp_path, {"competicion_id": 73, "temporada_id": 108, "partido_id": 42}
    )
    carpeta = tmp_path / "competicion_73" / "temporada_108" / "partido_42"

    assert manifiesto["total_jugadas_elegibles"] == 3
    assert [j["equipo_protagonista"] for j in manifiesto["jugadas"]] == ["A", "B", "A"]
    assert [j["intervenciones_rivales_absorbidas"] for j in manifiesto["jugadas"]] == [[], [], []]
    assert [list(leer_jugada(carpeta / j["archivo"])["id"]) for j in manifiesto["jugadas"]] == [
        ["e1", "e2", "e3"], ["e4", "e5", "e6"], ["e7", "e8", "e9"]
    ]
