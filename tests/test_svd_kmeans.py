"""Contratos observables de muestreo y publicación del clustering alternativo."""

import json

import numpy as np
import pytest

from src.classification.svd_kmeans import (
    _distancias_centroides, publicar_corpus_completo, seleccionar_partidos,
)


def test_muestra_por_partido_cubre_cuotas_y_no_filtra_encuentros():
    manifiestos = [
        {"temporada": temporada, "cantidad": 20, "ruta": f"{temporada}-{i}"}
        for temporada in ("temporada_1", "temporada_2") for i in range(8)
    ]
    train, test = seleccionar_partidos(manifiestos, entrenamiento=80, evaluacion=40)
    assert sum(cantidad for _, cantidad in train) == 80
    assert sum(cantidad for _, cantidad in test) == 40
    assert {item["ruta"] for item, _ in train}.isdisjoint(
        {item["ruta"] for item, _ in test})
    assert {item["temporada"] for item, _ in train} == {
        "temporada_1", "temporada_2"}
    assert {item["temporada"] for item, _ in test} == {
        "temporada_1", "temporada_2"}


def test_distancias_y_margen_solo_comparan_con_centroides():
    class Modelo:
        cluster_centers_ = np.array([[0., 0.], [2., 0.]])

    etiquetas, distancias, margen = _distancias_centroides(
        np.array([[0.2, 0.], [1.7, 0.]]), Modelo())
    assert etiquetas.tolist() == [0, 1]
    assert distancias == pytest.approx([0.2, 0.3])
    assert margen == pytest.approx([1.6, 1.4])


def test_publicacion_exige_revision_tactica_y_conciliacion(tmp_path):
    origen = tmp_path / "completo"
    origen.mkdir()
    for categoria in ("balon_parado", "juego_abierto"):
        (origen / f"modelo_{categoria}.joblib").write_bytes(b"modelo")
    (origen / "evaluacion_completa.json").write_text(json.dumps({
        "estado": "reentrenado_pendiente_revision",
        "total_jugadas": 2,
        "categorias": {categoria: {"filtros_cuantitativos_y_ejemplos": True}
                       for categoria in ("balon_parado", "juego_abierto")},
    }), encoding="utf-8")
    with (origen / "clasificacion_exploratoria.jsonl").open("w", encoding="utf-8") as archivo:
        for indice, categoria in enumerate(("balon_parado", "juego_abierto")):
            archivo.write(json.dumps({"play_id": f"p{indice}", "categoria": categoria,
                                      "etiqueta_exploratoria": "Sin patrón claro"}) + "\n")
    revision = tmp_path / "revision.json"
    revision.write_text(json.dumps({"aprobado": False}), encoding="utf-8")
    with pytest.raises(ValueError, match="aprobación táctica"):
        publicar_corpus_completo(origen, revision, tmp_path / "publicado")
    revision.write_text(json.dumps({
        "aprobado": True, "justificacion": "Ejemplos coherentes y distintivos",
        "categorias": {categoria: {"aprobado": True, "justificacion": "Tres ejemplos revisados"}
                       for categoria in ("balon_parado", "juego_abierto")},
    }), encoding="utf-8")
    resultado = publicar_corpus_completo(origen, revision, tmp_path / "publicado")
    assert resultado["total_jugadas"] == 2
    filas = [json.loads(linea) for linea in
             (tmp_path / "publicado" / "clasificacion.jsonl").read_text(
                 encoding="utf-8").splitlines()]
    assert all(fila["clase_principal"] == "Sin patrón claro" for fila in filas)
    assert all("etiqueta_exploratoria" not in fila for fila in filas)


def test_no_publica_como_validado_un_piloto_rechazado(tmp_path):
    origen = tmp_path / "exploratorio"
    origen.mkdir()
    (origen / "evaluacion_completa.json").write_text(json.dumps({
        "estado": "reentrenado_exploratorio_rechazado",
        "total_jugadas": 1,
        "categorias": {categoria: {"filtros_cuantitativos_y_ejemplos": True}
                       for categoria in ("balon_parado", "juego_abierto")},
    }), encoding="utf-8")
    revision = tmp_path / "revision.json"
    revision.write_text(json.dumps({
        "aprobado": True, "justificacion": "Intento de publicar",
        "categorias": {categoria: {"aprobado": True, "justificacion": "Intento"}
                       for categoria in ("balon_parado", "juego_abierto")},
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="piloto rechazado"):
        publicar_corpus_completo(origen, revision, tmp_path / "publicado")
