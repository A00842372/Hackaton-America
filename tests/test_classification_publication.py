"""Publicación observable de la clasificación y del diagnóstico rechazado."""

import json
import shutil

import pandas as pd
import pytest

from src.classification.exploratory_nmf import clasificar_muestra
from src.classification.publication import aplicar_modelos_publicados, publicar_clasificacion
from src.classification.selection import evaluar_corridas
from src.data_preparation.export import exportar_jugadas_partido


def _corpus_controlado(corpus):
    for partido_id in (41, 42, 43):
        eventos = []
        banda_alta = partido_id == 42
        origen_y = 60.0 if banda_alta else 20.0
        destino_y = 58.0 if banda_alta else 22.0
        for periodo, total in ((1, 6), (2, 5)):
            for ordinal in range(1, total + 1):
                indice = (periodo - 1) * 10 + ordinal
                patron = (
                    "From Corner" if periodo == 1 and ordinal <= 3 else
                    "From Free Kick" if periodo == 2 and ordinal <= 2 else
                    "Regular Play"
                )
                origen_x = 10.0 + ordinal * 8
                eventos.append({
                    "id": f"{partido_id}-{indice}", "index": indice,
                    "period": periodo, "match_id": partido_id,
                    "timestamp": None if ordinal == 5 else f"00:00:{indice:02d}.000",
                    "type": {"name": "Pass"}, "team": {"name": "A"},
                    "possession_team": {"name": "A"},
                    "possession": 1 if periodo == 1 and ordinal <= 3 else
                    3 if periodo == 1 else 4,
                    "play_pattern": {"name": patron},
                    "pass": {"end_location": [
                        origen_x - 8 if partido_id == 43 else origen_x + 8, destino_y,
                    ]},
                    "location": [origen_x, origen_y],
                })
                if periodo == 1 and ordinal == 3:
                    eventos.append({
                        "id": f"{partido_id}-presion", "index": indice + 0.1,
                        "period": periodo, "match_id": partido_id,
                        "timestamp": f"00:00:{indice:02d}.500",
                        "type": {"name": "Pressure"}, "team": {"name": "B"},
                        "possession_team": {"name": "B"}, "possession": 2,
                        "play_pattern": {"name": patron}, "pass": None,
                        "location": [60.0, origen_y],
                    })
        exportar_jugadas_partido(
            pd.DataFrame(eventos), corpus,
            {"competicion_id": 73, "temporada_id": 108, "partido_id": partido_id},
            dividir_fases_clasificables=True,
        )


def _evaluacion(corpus, tmp_path, decision):
    corrida = tmp_path / "corrida"
    clasificar_muestra(corpus, corrida, partidos_reservados=[43], ritmo=True)
    salida = tmp_path / "evaluacion"
    evaluar_corridas(
        {"base": corrida}, salida, candidato="base", cuantil_error=0.9,
        decision=decision, justificacion="Ensayo controlado de la publicación",
    )
    return salida / "evaluacion.json"


def test_publica_una_fila_por_jugada_vigente_y_reproduce_la_salida(tmp_path):
    corpus = tmp_path / "corpus"
    _corpus_controlado(corpus)
    evaluacion = _evaluacion(corpus, tmp_path, "aceptado")

    primer = publicar_clasificacion(corpus, evaluacion, tmp_path / "publicacion_1")
    segundo = publicar_clasificacion(corpus, evaluacion, tmp_path / "publicacion_2")

    filas_1 = (tmp_path / "publicacion_1" / "clasificacion.jsonl").read_bytes()
    filas_2 = (tmp_path / "publicacion_2" / "clasificacion.jsonl").read_bytes()
    assert filas_1 == filas_2
    assert primer == segundo
    filas = [json.loads(linea) for linea in filas_1.splitlines()]
    manifiestos = [json.loads(ruta.read_text()) for ruta in corpus.glob(
        "competicion_*/temporada_*/partido_*/manifiesto.json"
    )]
    ids_vigentes = {jugada["play_id"] for m in manifiestos for jugada in m["jugadas"]}
    categorias = {jugada["play_id"]: jugada["categoria_clasificacion"]
                  for m in manifiestos for jugada in m["jugadas"]}
    assert primer["estado"] == "clasificacion_publicada"
    assert primer["total_jugadas"] == len(filas) == len(ids_vigentes)
    assert {fila["play_id"] for fila in filas} == ids_vigentes
    assert all(fila["categoria"] == fila["modelo"] == categorias[fila["play_id"]]
               for fila in filas)
    assert any(fila["play_id"].endswith("_f01") for fila in filas)
    assert any(fila["play_id"].endswith("_f02") for fila in filas)
    assert any(fila["motivo_corte"] == "tramo_con_menos_de_min_pases" for fila in filas)
    assert any(fila["intervenciones_rivales_absorbidas"] for fila in filas)
    assert all(fila["clase_principal"] and fila["pesos"]
               and (corpus / fila["manifiesto"]).parent.joinpath(fila["archivo"]).is_file()
               for fila in filas)
    assert any(fila["clase_principal"] == "Sin patrón claro"
               and fila["error_reconstruccion"] is None for fila in filas)
    reflejada = next(f for f in filas if f["play_id"] == "c73_t108_m42_p0001_f01")
    normal = next(f for f in filas if f["play_id"] == "c73_t108_m41_p0001_f01")
    assert reflejada["reflejada"] is True and normal["reflejada"] is False
    assert reflejada["pesos"] == normal["pesos"]
    assert (tmp_path / "publicacion_1" / "modelo_balon_parado.joblib").is_file()
    assert (tmp_path / "publicacion_1" / "modelo_juego_abierto.joblib").is_file()
    informe = json.loads((tmp_path / "publicacion_1" / "informe.json").read_text())
    assert informe["parametros"]["ritmo"] is True

    shutil.rmtree(tmp_path / "corrida")
    shutil.rmtree(tmp_path / "evaluacion")
    aplicar_modelos_publicados(corpus, tmp_path / "publicacion_1", tmp_path / "futuras")
    assert (tmp_path / "futuras" / "clasificacion.jsonl").read_bytes() == filas_1

    archivo_roto = (corpus / "competicion_73" / "temporada_108" / "partido_43"
                   / "jugada_0002.json")
    contenido = json.loads(archivo_roto.read_text())
    contenido["data"].pop()
    archivo_roto.write_text(json.dumps(contenido))
    with pytest.raises(ValueError, match="no concilian"):
        aplicar_modelos_publicados(corpus, tmp_path / "publicacion_1", tmp_path / "fallida")
    assert not (tmp_path / "fallida").exists()


def test_rechazo_publica_diagnostico_sin_tabla_de_etiquetas(tmp_path):
    corpus = tmp_path / "corpus"
    _corpus_controlado(corpus)
    evaluacion = _evaluacion(corpus, tmp_path, "rechazado")

    resumen = publicar_clasificacion(corpus, evaluacion, tmp_path / "publicacion")

    carpeta = tmp_path / "publicacion"
    assert resumen["estado"] == "diagnostico_publicado"
    assert resumen["total_jugadas"] == 9
    assert not (carpeta / "clasificacion.jsonl").exists()
    informe = json.loads((carpeta / "informe.json").read_text())
    assert informe["decision"] == "rechazado"
    assert "jugadas" not in informe
    assert informe["diagnosticos"]["base"]["por_categoria"]["balon_parado"]["componentes"]
    assert (carpeta / "modelo_balon_parado.joblib").is_file()
    assert (carpeta / "modelo_juego_abierto.joblib").is_file()
    with pytest.raises(ValueError, match="rechazado"):
        aplicar_modelos_publicados(corpus, carpeta, tmp_path / "invalida")
    assert not (tmp_path / "invalida").exists()
