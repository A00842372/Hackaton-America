"""Publica clasificación por lotes o solo diagnósticos según la decisión NMF."""

import argparse
import hashlib
import json
import shutil
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from src.classification.exploratory_nmf import CATEGORIAS, proyectar_rasgos
from src.classification.features import ConfiguracionEspacial, representar_jugada
from src.data_preparation.prepared_corpus import _leer_jugadas_validas


def _leer_json(ruta: Path) -> dict[str, Any]:
    with ruta.open(encoding="utf-8") as archivo:
        valor = json.load(archivo)
    if not isinstance(valor, dict):
        raise ValueError(f"Se esperaba un objeto JSON en {ruta}")
    return valor


def _huella(ruta: Path) -> str:
    digest = hashlib.sha256()
    with ruta.open("rb") as archivo:
        for bloque in iter(lambda: archivo.read(1024 * 1024), b""):
            digest.update(bloque)
    return digest.hexdigest()


def _resumir_manifiestos(corpus: Path) -> tuple[list[Path], dict[str, Any]]:
    rutas = sorted(corpus.glob("competicion_*/temporada_*/partido_*/manifiesto.json"))
    if not rutas:
        raise ValueError(f"No hay partidos preparados en {corpus}")
    huella = hashlib.sha256()
    ids: set[str] = set()
    categorias: Counter[str] = Counter()
    total_eventos = 0
    for ruta in rutas:
        datos = ruta.read_bytes()
        huella.update(str(ruta.relative_to(corpus)).encode("utf-8"))
        huella.update(b"\0")
        huella.update(datos)
        manifiesto = json.loads(datos)
        if manifiesto.get("parametros", {}).get("dividir_fases_clasificables") is not True:
            raise ValueError(f"Partido sin fases de clasificación: {ruta}")
        jugadas = manifiesto["jugadas"]
        if len(jugadas) != manifiesto["total_jugadas_elegibles"]:
            raise ValueError(f"El total de jugadas no concilia: {ruta}")
        eventos = 0
        for detalle in jugadas:
            identificador = detalle["play_id"]
            if identificador in ids:
                raise ValueError(f"play_id duplicado: {identificador}")
            ids.add(identificador)
            categoria = detalle["categoria_clasificacion"]
            if categoria not in CATEGORIAS:
                raise ValueError(f"Categoría inválida en {ruta}: {categoria}")
            categorias[categoria] += 1
            nombre = detalle["archivo"]
            if Path(nombre).name != nombre or not nombre.endswith(".json"):
                raise ValueError(f"Archivo de jugada inválido en {ruta}: {nombre}")
            eventos += detalle["total_eventos"]
        if (eventos != manifiesto["total_eventos_exportados"]
                or eventos + len(manifiesto["eventos_no_exportados"])
                != manifiesto["total_eventos_entrada"]):
            raise ValueError(f"Los eventos no concilian: {ruta}")
        total_eventos += eventos
    resumen = {
        "total_partidos": len(rutas), "total_jugadas": len(ids),
        "total_por_categoria": dict(sorted(categorias.items())),
        "total_eventos_exportados": total_eventos,
        "huella_manifiestos_sha256": huella.hexdigest(),
    }
    ruta_resumen = corpus / "resumen_corpus.json"
    if ruta_resumen.exists():
        declarado = _leer_json(ruta_resumen)
        if (declarado.get("estado") != "completo"
                or declarado.get("partidos_procesados") != len(rutas)
                or declarado.get("total_jugadas_elegibles") != len(ids)
                or declarado.get("total_eventos_exportados") != total_eventos):
            raise ValueError("El resumen del corpus no concilia con los manifiestos vigentes")
        resumen["huella_resumen_sha256"] = _huella(ruta_resumen)
    return rutas, resumen


def _configuracion(parametros: dict[str, Any]) -> ConfiguracionEspacial | None:
    cuadricula = parametros["cuadricula"]
    if cuadricula is None:
        return None
    return ConfiguracionEspacial(
        cuadricula["zonas_x"], cuadricula["zonas_y"], parametros["peso_espacial"],
    )


def _copiar_modelos(
    origen: Path, destino: Path, modelos: dict[str, Any], *, verificar_huellas: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    artefactos = {}
    metadatos = {}
    nombres = set()
    for categoria in CATEGORIAS:
        metadata = modelos[categoria]
        nombre = metadata["archivo_modelo"]
        if Path(nombre).name != nombre or nombre in nombres:
            raise ValueError(f"Nombre de modelo inválido o duplicado: {nombre}")
        nombres.add(nombre)
        archivo = origen / nombre
        huella = _huella(archivo)
        if verificar_huellas and huella != metadata["sha256"]:
            raise ValueError(f"Cambió el modelo publicado de {categoria}")
        shutil.copy2(archivo, destino / nombre)
        artefacto = joblib.load(destino / nombre)
        if artefacto["componentes_ids"] != metadata["componentes"]:
            raise ValueError(f"Los IDs del modelo {categoria} no concilian")
        artefactos[categoria] = artefacto
        metadatos[categoria] = {**metadata, "sha256": huella}
    return artefactos, metadatos


def _proyectar_partido(
    carpeta: Path, manifiesto: dict[str, Any], artefactos: dict[str, Any],
    parametros: dict[str, Any], umbrales: dict[str, float], corpus: Path,
) -> list[dict[str, Any]]:
    jugadas = _leer_jugadas_validas(carpeta, manifiesto)
    por_categoria: dict[str, list[tuple[dict[str, Any], list[dict[str, Any]]]]] = {
        categoria: [] for categoria in CATEGORIAS
    }
    for detalle, bruto, _ in jugadas:
        categoria = detalle["categoria_clasificacion"]
        if categoria not in por_categoria:
            raise ValueError(f"Categoría inválida en {carpeta}: {categoria}")
        eventos = [dict(zip(bruto["columns"], fila)) for fila in bruto["data"]]
        por_categoria[categoria].append((detalle, eventos))

    filas = []
    for categoria, grupo in por_categoria.items():
        if not grupo:
            continue
        artefacto = artefactos[categoria]
        representaciones = [
            representar_jugada(
                eventos, detalle["equipo_protagonista"], _configuracion(parametros),
                modo_rivales=parametros["modo_rivales"],
                posicion=parametros["posicion"], ritmo=parametros["ritmo"],
            ) for detalle, eventos in grupo
        ]
        vectorizador = artefacto["vectorizador"]
        nmf = artefacto["nmf"]
        pesos, errores, coberturas = proyectar_rasgos(
            [rasgos for rasgos, _ in representaciones], vectorizador,
            artefacto["tfidf"], nmf,
        )
        normas = np.linalg.norm(nmf.components_, axis=1)
        nombres = artefacto["componentes_ids"]
        for (detalle, _), (_, metadatos), vector, error, cobertura in zip(
            grupo, representaciones, pesos, errores, coberturas,
        ):
            magnitudes = vector * normas
            total = float(magnitudes.sum())
            sin_rasgos = cobertura == 0 or total == 0
            dominante = None if sin_rasgos else nombres[int(np.argmax(magnitudes))]
            error_jugada = None if sin_rasgos else float(error)
            etiqueta = ("Sin patrón claro" if sin_rasgos or error_jugada > umbrales[categoria]
                        else dominante)
            filas.append({
                "play_id": detalle["play_id"], "play_id_origen": detalle["play_id_origen"],
                "competicion_id": manifiesto["competicion_id"],
                "temporada_id": manifiesto["temporada_id"],
                "partido_id": manifiesto["partido_id"],
                "categoria": categoria, "modelo": categoria,
                "clase_principal": etiqueta, "componente_dominante": dominante,
                "pesos": {nombre: float(valor) for nombre, valor in zip(nombres, vector)},
                "contribuciones_normalizadas": {
                    nombre: float(valor / total) for nombre, valor in zip(nombres, magnitudes)
                } if total else None,
                "error_reconstruccion": error_jugada,
                "cobertura_vocabulario": float(cobertura),
                "umbral_error": umbrales[categoria],
                "manifiesto": str((carpeta / "manifiesto.json").relative_to(corpus)),
                "archivo": detalle["archivo"],
                "primer_evento_id": detalle["primer_evento_id"],
                "ultimo_evento_id": detalle["ultimo_evento_id"],
                "total_eventos": detalle["total_eventos"],
                "pases_completados": detalle["pases_completados"],
                "motivo_corte": detalle["motivo_corte"],
                "intervenciones_rivales_absorbidas": detalle["intervenciones_rivales_absorbidas"],
                **metadatos,
            })
    return sorted(filas, key=lambda fila: fila["play_id"])


def _escribir_tabla(
    corpus: Path, rutas: list[Path], destino: Path, artefactos: dict[str, Any],
    parametros: dict[str, Any], umbrales: dict[str, float], esperado: int,
) -> None:
    total = 0
    with destino.open("w", encoding="utf-8") as archivo:
        for ruta in rutas:
            manifiesto = _leer_json(ruta)
            filas = _proyectar_partido(
                ruta.parent, manifiesto, artefactos, parametros, umbrales, corpus,
            )
            if len(filas) != manifiesto["total_jugadas_elegibles"]:
                raise ValueError(f"La clasificación no concilia con {ruta}")
            for fila in filas:
                archivo.write(json.dumps(
                    fila, ensure_ascii=False, sort_keys=True, allow_nan=False,
                ) + "\n")
            total += len(filas)
    if total != esperado:
        raise ValueError("La clasificación no cubre todas las jugadas vigentes")


def publicar_clasificacion(
    ruta_corpus: str | Path, ruta_evaluacion: str | Path, ruta_salida: str | Path,
) -> dict[str, Any]:
    """Publica el diagnóstico; solo una evaluación aceptada habilita filas por jugada."""
    corpus = Path(ruta_corpus)
    evaluacion_path = Path(ruta_evaluacion)
    salida = Path(ruta_salida)
    if salida.exists():
        raise FileExistsError(f"La salida ya existe: {salida}")
    evaluacion = _leer_json(evaluacion_path)
    decision = evaluacion["decision"]
    estados = {"aceptado": "clasificacion_validada", "rechazado": "diagnostico_exploratorio"}
    if decision not in estados or evaluacion.get("estado") != estados[decision]:
        raise ValueError("La decisión y el estado de la evaluación no concilian")
    if any((fila["clase_principal"] is None) == (decision == "aceptado")
           for fila in evaluacion["jugadas"]):
        raise ValueError("La tabla exploratoria contradice la decisión de evaluación")
    candidato = evaluacion["candidato"]
    resultado_path = Path(evaluacion["corridas"][candidato]["ruta_resultados"])
    resultado = _leer_json(resultado_path)
    if Path(resultado["ruta_corpus"]).resolve() != corpus.resolve():
        raise ValueError("El modelo fue ajustado sobre otro corpus")
    rutas, resumen = _resumir_manifiestos(corpus)
    parametros = resultado["parametros"]
    umbrales = evaluacion["umbrales_error"]
    if set(resultado["modelos"]) != set(CATEGORIAS) or set(umbrales) != set(CATEGORIAS):
        raise ValueError("Falta un modelo o umbral de clasificación")

    salida.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".publicacion-", dir=salida.parent) as temporal:
        provisional = Path(temporal) / "resultado"
        provisional.mkdir()
        artefactos, modelos = _copiar_modelos(
            resultado_path.parent, provisional, resultado["modelos"],
            verificar_huellas=False,
        )

        estado = "clasificacion_publicada" if decision == "aceptado" else "diagnostico_publicado"
        informe = {
            "estado": estado, "decision": decision,
            "justificacion": evaluacion["justificacion"],
            "candidato": candidato, "cuantil_error": evaluacion["cuantil_error"],
            "convencion_pesos": evaluacion["convencion_pesos"],
            "umbrales_error": umbrales,
            "partidos_entrenamiento": evaluacion["partidos_entrenamiento"],
            "partidos_evaluacion": evaluacion["partidos_evaluacion"],
            "parametros": parametros, "modelos": modelos,
            "diagnosticos": evaluacion["corridas"],
            "comparaciones": evaluacion["comparaciones"],
            "evaluacion_sha256": _huella(evaluacion_path),
            "resultados_candidato_sha256": _huella(resultado_path),
            "corpus": str(corpus), **resumen,
        }
        if decision == "aceptado":
            _escribir_tabla(
                corpus, rutas, provisional / "clasificacion.jsonl", artefactos,
                parametros, umbrales, resumen["total_jugadas"],
            )
            informe["tabla_clasificacion"] = "clasificacion.jsonl"
        with (provisional / "informe.json").open("w", encoding="utf-8") as archivo:
            json.dump(informe, archivo, ensure_ascii=False, indent=2,
                      sort_keys=True, allow_nan=False)
        provisional.replace(salida)
    return {"estado": estado, "decision": decision,
            "total_partidos": resumen["total_partidos"],
            "total_jugadas": resumen["total_jugadas"]}


def aplicar_modelos_publicados(
    ruta_corpus: str | Path, ruta_modelos: str | Path, ruta_salida: str | Path,
) -> dict[str, Any]:
    """Clasifica otro corpus con un paquete aceptado, sin el piloto original."""
    corpus, origen, salida = Path(ruta_corpus), Path(ruta_modelos), Path(ruta_salida)
    if salida.exists():
        raise FileExistsError(f"La salida ya existe: {salida}")
    informe_origen = origen / "informe.json"
    informe = _leer_json(informe_origen)
    if informe.get("decision") != "aceptado" or informe.get("estado") != "clasificacion_publicada":
        raise ValueError("No se pueden aplicar modelos de un diagnóstico rechazado")
    rutas, resumen = _resumir_manifiestos(corpus)
    modelos = informe["modelos"]
    umbrales = informe["umbrales_error"]
    if set(modelos) != set(CATEGORIAS) or set(umbrales) != set(CATEGORIAS):
        raise ValueError("Falta un modelo o umbral de clasificación")
    salida.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".publicacion-", dir=salida.parent) as temporal:
        provisional = Path(temporal) / "resultado"
        provisional.mkdir()
        artefactos, _ = _copiar_modelos(
            origen, provisional, modelos, verificar_huellas=True,
        )
        _escribir_tabla(
            corpus, rutas, provisional / "clasificacion.jsonl", artefactos,
            informe["parametros"], umbrales, resumen["total_jugadas"],
        )
        nuevo = {
            "estado": "clasificacion_publicada", "decision": "aceptado",
            "origen_modelos_sha256": _huella(informe_origen),
            "candidato": informe["candidato"], "parametros": informe["parametros"],
            "convencion_pesos": informe["convencion_pesos"],
            "umbrales_error": umbrales, "modelos": modelos,
            "corpus": str(corpus), "tabla_clasificacion": "clasificacion.jsonl",
            **resumen,
        }
        with (provisional / "informe.json").open("w", encoding="utf-8") as archivo:
            json.dump(nuevo, archivo, ensure_ascii=False, indent=2,
                      sort_keys=True, allow_nan=False)
        provisional.replace(salida)
    return {"estado": "clasificacion_publicada", "decision": "aceptado",
            "total_partidos": resumen["total_partidos"],
            "total_jugadas": resumen["total_jugadas"]}


def main() -> None:
    parser = argparse.ArgumentParser(description="Publicar clasificación o diagnóstico NMF")
    parser.add_argument("ruta_corpus", type=Path)
    parser.add_argument("ruta_evaluacion", type=Path)
    parser.add_argument("ruta_salida", type=Path)
    parser.add_argument("--usar-modelos-publicados", action="store_true")
    args = parser.parse_args()
    ejecutar = aplicar_modelos_publicados if args.usar_modelos_publicados else publicar_clasificacion
    print(json.dumps(ejecutar(args.ruta_corpus, args.ruta_evaluacion, args.ruta_salida),
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
