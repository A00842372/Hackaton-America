import pandas as pd
import pytest

from src.data_preparation.plays import (
    contar_pases_completados,
    es_jugada_elegible,
    extraer_secuencias_posesion,
)


def test_contar_pases_completados_solo_protagonista_y_sin_outcome():
    events = [
        # 1. Pase completado del protagonista (pass dict sin outcome)
        {
            "id": "e1",
            "type": {"id": 30, "name": "Pass"},
            "team": {"id": 1, "name": "Protagonista"},
            "pass": {"recipient": {"id": 10, "name": "Player 10"}},
        },
        # 2. Pase incompleto del protagonista (con outcome)
        {
            "id": "e2",
            "type": {"id": 30, "name": "Pass"},
            "team": {"id": 1, "name": "Protagonista"},
            "pass": {"outcome": {"id": 9, "name": "Incomplete"}},
        },
        # 3. Pase del rival (no debe contar)
        {
            "id": "e3",
            "type": {"id": 30, "name": "Pass"},
            "team": {"id": 2, "name": "Rival"},
            "pass": {},
        },
        # 4. Acción que no es pase (Conducción / Carry)
        {
            "id": "e4",
            "type": {"id": 43, "name": "Carry"},
            "team": {"id": 1, "name": "Protagonista"},
        },
        # 5. Otro pase completado del protagonista (sin campo pass o pass None/dict vacío)
        {
            "id": "e5",
            "type": {"id": 30, "name": "Pass"},
            "team": {"id": 1, "name": "Protagonista"},
        },
    ]
    df = pd.DataFrame(events)

    count = contar_pases_completados(df, equipo_protagonista="Protagonista")
    assert count == 2


def test_es_jugada_elegible_umbral_tres_pases():
    # 2 pases completados -> ineligible
    df_ineligible = pd.DataFrame([
        {
            "id": f"p{i}",
            "type": {"id": 30, "name": "Pass"},
            "team": {"id": 1, "name": "Team A"},
            "possession_team": {"id": 1, "name": "Team A"},
        }
        for i in range(2)
    ])
    es_eleg, pases = es_jugada_elegible(df_ineligible, min_pases=3)
    assert not es_eleg
    assert pases == 2

    # 3 pases completados -> eligible
    df_eligible = pd.DataFrame([
        {
            "id": f"p{i}",
            "type": {"id": 30, "name": "Pass"},
            "team": {"id": 1, "name": "Team A"},
            "possession_team": {"id": 1, "name": "Team A"},
        }
        for i in range(3)
    ])
    es_eleg, pases = es_jugada_elegible(df_eligible, min_pases=3)
    assert es_eleg
    assert pases == 3


def test_extraer_secuencias_posesion_separa_por_periodo_y_posesion():
    events = [
        # Administrativo: Starting XI
        {
            "id": "adm-1",
            "period": 1,
            "index": 1,
            "possession": 1,
            "possession_team": {"id": 1, "name": "Team A"},
            "type": {"id": 35, "name": "Starting XI"},
            "team": {"id": 1, "name": "Team A"},
        },
        # Posesión 2, Periodo 1
        {
            "id": "e1",
            "period": 1,
            "index": 2,
            "possession": 2,
            "possession_team": {"id": 1, "name": "Team A"},
            "type": {"id": 30, "name": "Pass"},
            "team": {"id": 1, "name": "Team A"},
        },
        {
            "id": "e2",
            "period": 1,
            "index": 3,
            "possession": 2,
            "possession_team": {"id": 1, "name": "Team A"},
            "type": {"id": 43, "name": "Carry"},
            "team": {"id": 1, "name": "Team A"},
        },
        # Posesión 3, Periodo 1
        {
            "id": "e3",
            "period": 1,
            "index": 4,
            "possession": 3,
            "possession_team": {"id": 2, "name": "Team B"},
            "type": {"id": 30, "name": "Pass"},
            "team": {"id": 2, "name": "Team B"},
        },
    ]
    df = pd.DataFrame(events)
    secuencias, no_exportados = extraer_secuencias_posesion(df)

    assert len(secuencias) == 2
    assert len(no_exportados) == 1
    assert no_exportados[0]["id"] == "adm-1"
    assert no_exportados[0]["motivo"] == "evento_administrativo"

    # Primera secuencia: posesion 2
    assert len(secuencias[0]) == 2
    assert list(secuencias[0]["id"]) == ["e1", "e2"]

    # Segunda secuencia: posesion 3
    assert len(secuencias[1]) == 1
    assert list(secuencias[1]["id"]) == ["e3"]
