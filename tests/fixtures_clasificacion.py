"""Corpus mínimo compartido por pruebas de la salida de clasificación."""

import pandas as pd

from src.data_preparation.export import exportar_jugadas_partido


def preparar_partido(corpus, partido_id):
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
