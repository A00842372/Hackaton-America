import pandas as pd

from src.data_preparation.export import exportar_jugadas_partido, leer_jugada
from src.data_preparation.plays import extraer_secuencias_candidatas


def evento(index, tipo, posesion, equipo="A", periodo=1, **campos):
    return {
        "id": f"e{index}",
        "match_id": 42,
        "index": index,
        "period": periodo,
        "timestamp": f"00:00:{index:02d}.000",
        "type": {"name": tipo},
        "team": {"name": equipo},
        "possession_team": {"name": "A" if posesion < 90 else "B"},
        "possession": posesion,
        **campos,
    }


def ids(secuencias):
    return [list(secuencia["id"]) for secuencia in secuencias]


def test_presion_tiro_rebote_y_ventaja_conservan_continuidad():
    eventos = pd.DataFrame([
        evento(1, "Ball Recovery", 1),
        evento(2, "Pass", 1),
        evento(3, "Pressure", 1, equipo="B"),
        evento(4, "Shot", 1, shot={"outcome": {"name": "Saved"}}),
        evento(5, "Goal Keeper", 1, equipo="B"),
        evento(6, "Ball Recovery", 2),
        evento(7, "Foul Committed", 2, equipo="B", foul_committed={"advantage": True}),
        evento(8, "Foul Won", 2, foul_won={"advantage": True}),
        evento(9, "Pass", 2),
    ])

    secuencias, excluidos = extraer_secuencias_candidatas(eventos)

    assert ids(secuencias) == [[f"e{i}" for i in range(1, 10)]]
    assert excluidos == []


def test_reanudaciones_salidas_detenciones_y_periodo_separan_secuencias():
    eventos = pd.DataFrame([
        evento(1, "Ball Recovery", 1),
        evento(2, "Pass", 1, out=True, **{"pass": {"outcome": {"name": "Out"}}}),
        evento(3, "Pass", 1, **{"pass": {"type": {"name": "Throw-in"}}}),
        evento(4, "Pass", 1),
        evento(5, "Foul Committed", 1, equipo="B"),
        evento(6, "Foul Won", 1, timestamp="00:00:05.000"),
        evento(7, "Pass", 1, **{"pass": {"type": {"name": "Free Kick"}}}),
        evento(8, "Pass", 1),
        evento(9, "Injury Stoppage", 1),
        evento(10, "Pass", 1, **{"pass": {"type": {"name": "Corner"}}}),
        evento(11, "Pass", 1, out=True),
        evento(12, "Pass", 1, **{"pass": {"type": {"name": "Goal Kick"}}}),
        evento(13, "Half End", 1),
        evento(14, "Half Start", 1, periodo=2),
        evento(15, "Pass", 1, periodo=2),
    ])

    secuencias, excluidos = extraer_secuencias_candidatas(eventos)

    assert ids(secuencias) == [
        ["e1", "e2"], ["e3", "e4", "e5", "e6"], ["e7", "e8"],
        ["e10", "e11"], ["e12"], ["e15"],
    ]
    assert {item["id"] for item in excluidos} == {"e9", "e13", "e14"}


def test_cierre_antes_de_filtrar_pases_y_exportacion_solo_de_elegibles(tmp_path):
    eventos = pd.DataFrame([
        evento(1, "Ball Recovery", 1),
        evento(2, "Pass", 1, **{"pass": {}}),
        evento(3, "Pass", 1, **{"pass": {}}),
        evento(4, "Shot", 1, shot={"outcome": {"name": "Saved"}}),
        evento(5, "Ball Recovery", 2),
        evento(6, "Pass", 2, **{"pass": {}}),
        evento(7, "Pass", 2, out=True, **{"pass": {"outcome": {"name": "Out"}}}),
        evento(8, "Pass", 3, **{"pass": {"type": {"name": "Throw-in"}}}),
    ])

    manifiesto = exportar_jugadas_partido(
        eventos, tmp_path, {"competicion_id": 73, "temporada_id": 108, "partido_id": 42}
    )
    carpeta = tmp_path / "competicion_73" / "temporada_108" / "partido_42"

    assert manifiesto["total_jugadas_elegibles"] == 1
    assert manifiesto["total_candidatas_descartadas"] == 1
    assert manifiesto["jugadas"][0]["pases_completados"] == 3
    assert manifiesto["jugadas"][0]["posesiones_origen"] == [1, 2]
    assert list(leer_jugada(carpeta / "jugada_0001.json")["id"]) == [f"e{i}" for i in range(1, 8)]
    assert not (carpeta / "jugada_0002.json").exists()
    assert manifiesto["eventos_no_exportados"] == [
        {"id": "e8", "motivo": "secuencia_descartada_menos_3_pases"}
    ]


def test_partidos_distintos_y_tiro_fuera_no_se_unen():
    eventos = pd.DataFrame([
        evento(1, "Pass", 1),
        evento(2, "Shot", 1, shot={"outcome": {"name": "Off T"}}),
        evento(3, "Ball Recovery", 2),
        evento(4, "Pass", 2, match_id=43),
    ])

    secuencias, _ = extraer_secuencias_candidatas(eventos)

    assert ids(secuencias) == [["e1", "e2"], ["e3"], ["e4"]]
