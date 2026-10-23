"""Clasificación observable después de seleccionar y evaluar componentes NMF."""

import json

from src.classification.exploratory_nmf import clasificar_muestra
from src.classification.selection import evaluar_corridas
from tests.fixtures_clasificacion import preparar_partido


def test_conserva_pesos_y_deja_sin_patron_las_jugadas_mal_ajustadas(tmp_path):
    corpus = tmp_path / "corpus"
    for partido_id in (41, 42, 43, 44):
        preparar_partido(corpus, partido_id)
    corrida = tmp_path / "base"
    clasificar_muestra(corpus, corrida, partidos_reservados=[44])

    evaluacion = evaluar_corridas(
        {"base": corrida}, tmp_path / "evaluacion", candidato="base",
        cuantil_error=0.5, decision="aceptado",
        justificacion="Piloto controlado para verificar la política de mal ajuste",
    )

    assert evaluacion == json.loads((tmp_path / "evaluacion" / "evaluacion.json").read_text())
    assert evaluacion["decision"] == "aceptado"
    assert len(evaluacion["jugadas"]) == 8
    assert len({fila["play_id"] for fila in evaluacion["jugadas"]}) == 8
    assert {fila["categoria"] for fila in evaluacion["jugadas"]} == {
        "balon_parado", "juego_abierto",
    }
    assert {fila["clase_principal"] for fila in evaluacion["jugadas"]} >= {"Sin patrón claro"}
    assert any(fila["clase_principal"] != "Sin patrón claro" for fila in evaluacion["jugadas"])
    assert all(fila["pesos"] and (corpus / fila["manifiesto"]).parent.joinpath(
        fila["archivo"]
    ).is_file() for fila in evaluacion["jugadas"])
    assert all(fila["umbral_error"] is not None for fila in evaluacion["jugadas"])
    assert all(abs(sum(fila["contribuciones_normalizadas"].values()) - 1) < 1e-9
               for fila in evaluacion["jugadas"] if fila["contribuciones_normalizadas"])
    assert any(fila["contribuciones_normalizadas"] is None
               for fila in evaluacion["jugadas"])
    diagnostico = evaluacion["corridas"]["base"]["por_categoria"]["juego_abierto"]
    assert diagnostico["componentes"]
    assert all(ficha["ngramas_destacados"] and ficha["ejemplos_representativos"]
               for ficha in diagnostico["componentes"])
    for ficha in diagnostico["componentes"]:
        candidatas = [fila["error_reconstruccion"] for fila in evaluacion["jugadas"]
                      if fila["particion"] == "entrenamiento"
                      and fila["componente_dominante"] == ficha["componente"]
                      and fila["error_reconstruccion"] is not None]
        if candidatas:
            assert ficha["criterio_ejemplos"] == "dominante_y_mejor_encaje"
            assert ficha["ejemplos_representativos"][0]["error"] == min(candidatas)
        else:
            assert ficha["criterio_ejemplos"] == "sin_dominantes_mayor_contribucion"
    assert diagnostico["peor_encaje"]
    assert diagnostico["por_reanudacion"]
    assert "correlacion_error_eventos" in diagnostico["por_longitud"]
    assert len(diagnostico["umbrales_experimentales"]) >= 3


def test_compara_frecuencia_y_tfidf_con_las_mismas_jugadas(tmp_path):
    corpus = tmp_path / "corpus"
    for partido_id in (41, 42, 43, 44):
        preparar_partido(corpus, partido_id)
    tfidf = tmp_path / "tfidf"
    frecuencia = tmp_path / "frecuencia"
    clasificar_muestra(corpus, tfidf, partidos_reservados=[44])
    clasificar_muestra(
        corpus, frecuencia, partidos_reservados=[44],
        ponderacion="frecuencia_l2",
    )

    evaluacion = evaluar_corridas(
        {"tfidf": tfidf, "frecuencia": frecuencia}, tmp_path / "evaluacion",
        candidato="frecuencia", cuantil_error=0.9,
        decision="rechazado", justificacion="Comparación controlada",
    )

    assert len(evaluacion["jugadas"]) == 8
    assert evaluacion["corridas"]["tfidf"]["parametros"]["ponderacion"] == "tfidf_l2"
    assert evaluacion["corridas"]["frecuencia"]["parametros"]["ponderacion"] == "frecuencia_l2"
    assert all(fila["clase_principal"] is None and fila["etiqueta_exploratoria"]
               for fila in evaluacion["jugadas"])
    assert evaluacion["estado"] == "diagnostico_exploratorio"
    comparacion = evaluacion["comparaciones"]["tfidf"]["juego_abierto"]
    assert comparacion["ngramas_compartidos"] > 0
    assert comparacion["pares_componentes"]
    assert 0 <= comparacion["acuerdo_dominante_entrenamiento"] <= 1
    assert comparacion["jugadas_comparables_evaluacion"] == 0
    assert comparacion["acuerdo_dominante_evaluacion"] is None
