"""Evaluación reproducible de componentes NMF exploratorios (#18)."""

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Literal, Mapping

import joblib
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr

from src.classification.exploratory_nmf import CATEGORIAS


Decision = Literal["aceptado", "rechazado"]


def _nombre(valor: Any) -> str:
    return valor.get("name", "sin_dato") if isinstance(valor, dict) else "sin_dato"


def _metadatos(resultado: dict[str, Any]) -> dict[str, dict[str, Any]]:
    corpus = Path(resultado["ruta_corpus"])
    manifiestos: dict[str, dict[str, Any]] = {}
    metadatos = {}
    for fila in resultado["jugadas"]:
        ruta = fila["manifiesto"]
        if ruta not in manifiestos:
            manifiesto = json.loads((corpus / ruta).read_text(encoding="utf-8"))
            manifiestos[ruta] = {jugada["play_id"]: jugada for jugada in manifiesto["jugadas"]}
        detalle = manifiestos[ruta][fila["play_id"]]
        archivo = json.loads((corpus / ruta).parent.joinpath(fila["archivo"]).read_text(
            encoding="utf-8",
        ))
        columnas = archivo["columns"]
        primer_evento = dict(zip(columnas, archivo["data"][0]))
        metadatos[fila["play_id"]] = {
            "total_eventos": detalle["total_eventos"],
            "pases_completados": detalle["pases_completados"],
            "intervenciones_rivales": len(detalle["intervenciones_rivales_absorbidas"]),
            "reanudacion": _nombre(primer_evento.get("play_pattern")),
            "temporada_id": int((corpus / ruta).parent.parent.name.removeprefix("temporada_")),
        }
    return metadatos


def _ejemplo(fila: dict[str, Any], metadatos: dict[str, dict[str, Any]]) -> dict[str, Any]:
    return {
        "play_id": fila["play_id"], "particion": fila["particion"],
        "error": fila["error_reconstruccion"],
        "total_eventos": metadatos[fila["play_id"]]["total_eventos"],
        "manifiesto": fila["manifiesto"], "archivo": fila["archivo"],
    }


def _resumen_filas(filas: list[dict[str, Any]]) -> dict[str, Any]:
    errores = [fila["error_reconstruccion"] for fila in filas
               if fila["error_reconstruccion"] is not None]
    return {
        "jugadas": len(filas),
        "sin_rasgos_conocidos": len(filas) - len(errores),
        "error_medio": float(np.mean(errores)) if errores else None,
        "error_mediano": float(np.median(errores)) if errores else None,
        "cobertura_media": float(np.mean([fila["cobertura_vocabulario"] for fila in filas]))
        if filas else None,
    }


def _contribuciones(
    resultado: dict[str, Any], ruta: Path,
) -> dict[str, dict[str, Any]]:
    normas = {}
    for categoria in CATEGORIAS:
        modelo = resultado["modelos"][categoria]
        componentes = joblib.load(ruta / modelo["archivo_modelo"])["nmf"].components_
        normas[categoria] = dict(zip(modelo["componentes"], np.linalg.norm(componentes, axis=1)))
    contribuciones = {}
    for fila in resultado["jugadas"]:
        magnitudes = {
            nombre: float(peso * normas[fila["categoria"]][nombre])
            for nombre, peso in fila["pesos"].items()
        }
        total = sum(magnitudes.values())
        contribuciones[fila["play_id"]] = {
            "normalizadas": {nombre: valor / total for nombre, valor in magnitudes.items()}
            if total > 0 else None,
            "dominante": max(magnitudes, key=magnitudes.__getitem__) if total > 0 else None,
        }
    return contribuciones


def _errores_entrenamiento(resultado: dict[str, Any], categoria: str) -> list[float]:
    return [
        fila["error_reconstruccion"] for fila in resultado["jugadas"]
        if fila["categoria"] == categoria and fila["particion"] == "entrenamiento"
        and fila["error_reconstruccion"] is not None
    ]


def _diagnosticar_categoria(
    resultado: dict[str, Any], ruta: Path, categoria: str,
    metadatos: dict[str, dict[str, Any]], cuantil: float,
    contribuciones: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    filas = [fila for fila in resultado["jugadas"] if fila["categoria"] == categoria]
    entrenamiento = [fila for fila in filas if fila["particion"] == "entrenamiento"]
    evaluacion = [fila for fila in filas if fila["particion"] == "evaluacion"]
    errores_train = _errores_entrenamiento(resultado, categoria)
    umbrales = []
    for q in sorted({0.9, 0.95, 0.99, cuantil}):
        umbral = float(np.quantile(errores_train, q))
        umbrales.append({
            "cuantil": q, "umbral": umbral,
            "rechazadas_entrenamiento": sum(
                fila["error_reconstruccion"] is None
                or fila["error_reconstruccion"] > umbral for fila in entrenamiento
            ),
            "rechazadas_evaluacion": sum(
                fila["error_reconstruccion"] is None
                or fila["error_reconstruccion"] > umbral for fila in evaluacion
            ),
        })

    modelo = resultado["modelos"][categoria]
    artefacto = joblib.load(ruta / modelo["archivo_modelo"])
    h = artefacto["nmf"].components_
    vocabulario = modelo["vocabulario"]
    ids = modelo["componentes"]
    fichas = []
    for indice, componente in enumerate(ids):
        pesos = h[indice]
        total = float(pesos.sum())
        asignadas = [fila for fila in filas
                     if contribuciones[fila["play_id"]]["dominante"] == componente]
        representantes = sorted(
            (fila for fila in entrenamiento
             if contribuciones[fila["play_id"]]["dominante"] == componente
             and fila["error_reconstruccion"] is not None),
            key=lambda fila: (fila["error_reconstruccion"], fila["play_id"]),
        )
        criterio_representantes = "dominante_y_mejor_encaje"
        if not representantes:
            representantes = sorted(
                (fila for fila in entrenamiento
                 if fila["error_reconstruccion"] is not None),
                key=lambda fila: (
                    -(contribuciones[fila["play_id"]]["normalizadas"] or {}).get(
                        componente, 0,
                    ),
                    fila["error_reconstruccion"], fila["play_id"],
                ),
            )
            criterio_representantes = "sin_dominantes_mayor_contribucion"
        asignadas.sort(key=lambda fila: (
            fila["error_reconstruccion"] is None,
            -(fila["error_reconstruccion"] or 0), fila["play_id"],
        ))
        destacados = np.argsort(-pesos, kind="stable")[:10]
        fichas.append({
            "componente": componente,
            "ngramas_destacados": [
                {"ngramas": vocabulario[i], "peso_h": float(pesos[i])} for i in destacados
            ],
            "ejemplos_representativos": [_ejemplo(fila, metadatos)
                                        for fila in representantes[:3]],
            "criterio_ejemplos": criterio_representantes,
            "peor_encaje": [_ejemplo(fila, metadatos) for fila in asignadas[:3]],
            "fraccion_masa_rival": float(sum(
                peso for peso, rasgo in zip(pesos, vocabulario) if "rival_" in rasgo
            ) / total) if total else 0.0,
            "fraccion_masa_sin_dato": float(sum(
                peso for peso, rasgo in zip(pesos, vocabulario) if "sin_dato" in rasgo
            ) / total) if total else 0.0,
            "fraccion_masa_espacial": float(sum(
                peso for peso, rasgo in zip(pesos, vocabulario) if rasgo.startswith("espacio:")
            ) / total) if total else 0.0,
            "jugadas_dominantes_evaluacion": sum(
                contribuciones[fila["play_id"]]["dominante"] == componente
                for fila in evaluacion
            ),
            "eventos_medios_dominantes": float(np.mean([
                metadatos[fila["play_id"]]["total_eventos"] for fila in asignadas
            ])) if asignadas else None,
        })

    errores_eval = [fila for fila in evaluacion if fila["error_reconstruccion"] is not None]
    candidatos_peor_encaje = errores_eval or [
        fila for fila in entrenamiento if fila["error_reconstruccion"] is not None
    ]
    largos = [metadatos[fila["play_id"]]["total_eventos"] for fila in errores_eval]
    errores = [fila["error_reconstruccion"] for fila in errores_eval]
    correlacion = None
    if len(largos) > 2 and len(set(largos)) > 1 and len(set(errores)) > 1:
        correlacion = float(spearmanr(largos, errores).statistic)
        if not math.isfinite(correlacion):
            correlacion = None
    patrones: dict[str, Counter[str]] = defaultdict(Counter)
    temporadas: dict[int, list[dict[str, Any]]] = defaultdict(list)
    rivales: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for fila in evaluacion:
        patrones[metadatos[fila["play_id"]]["reanudacion"]][
            contribuciones[fila["play_id"]]["dominante"] or "sin_rasgos_conocidos"
        ] += 1
        temporadas[metadatos[fila["play_id"]]["temporada_id"]].append(fila)
        rivales["con" if metadatos[fila["play_id"]]["intervenciones_rivales"] else "sin"].append(fila)
    concentracion = Counter(contribuciones[fila["play_id"]]["dominante"]
                            or "sin_rasgos_conocidos"
                            for fila in evaluacion)
    return {
        "entrenamiento": _resumen_filas(entrenamiento),
        "evaluacion": _resumen_filas(evaluacion),
        "umbrales_experimentales": umbrales,
        "matriz_entrenamiento": modelo["matriz_entrenamiento"],
        "concentracion_evaluacion": dict(concentracion),
        "mayor_fraccion_componente": max(concentracion.values()) / len(evaluacion)
        if evaluacion else None,
        "por_reanudacion": {nombre: dict(conteos) for nombre, conteos in sorted(patrones.items())},
        "por_temporada": {str(id_): _resumen_filas(grupo)
                          for id_, grupo in sorted(temporadas.items())},
        "por_intervencion_rival": {nombre: _resumen_filas(rivales[nombre])
                                   for nombre in ("con", "sin")},
        "por_longitud": {"correlacion_error_eventos": correlacion},
        "peor_encaje": [_ejemplo(fila, metadatos) for fila in sorted(
            candidatos_peor_encaje,
            key=lambda fila: (-fila["error_reconstruccion"], fila["play_id"]),
        )[:5]],
        "componentes": fichas,
    }


def _umbrales(resultado: dict[str, Any], cuantil: float) -> dict[str, float]:
    umbrales = {}
    for categoria in CATEGORIAS:
        errores = _errores_entrenamiento(resultado, categoria)
        if not errores:
            raise ValueError(f"No hay errores de entrenamiento para {categoria}")
        umbrales[categoria] = float(np.quantile(errores, cuantil))
    return umbrales


def _comparar_componentes(
    base: dict[str, Any], ruta_base: Path,
    variante: dict[str, Any], ruta_variante: Path,
    categoria: str,
    contribuciones_base: dict[str, dict[str, Any]],
    contribuciones_variante: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    modelo_base = base["modelos"][categoria]
    modelo_variante = variante["modelos"][categoria]
    vocab_base = {nombre: i for i, nombre in enumerate(modelo_base["vocabulario"])}
    vocab_variante = {nombre: i for i, nombre in enumerate(modelo_variante["vocabulario"])}
    comunes = sorted(vocab_base.keys() & vocab_variante.keys())
    h_base = joblib.load(ruta_base / modelo_base["archivo_modelo"])["nmf"].components_
    h_variante = joblib.load(ruta_variante / modelo_variante["archivo_modelo"])["nmf"].components_
    if comunes:
        a = h_base[:, [vocab_base[nombre] for nombre in comunes]]
        b = h_variante[:, [vocab_variante[nombre] for nombre in comunes]]
        a = a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)
        b = b / np.maximum(np.linalg.norm(b, axis=1, keepdims=True), 1e-12)
        similitud = a @ b.T
    else:
        similitud = np.zeros((len(h_base), len(h_variante)))
    indices_base, indices_variante = linear_sum_assignment(-similitud)
    pares = [
        {"base": modelo_base["componentes"][i],
         "variante": modelo_variante["componentes"][j],
         "similitud_coseno": float(similitud[i, j])}
        for i, j in zip(indices_base, indices_variante)
    ]
    equivalencias = {par["variante"]: par["base"] for par in pares}
    acuerdos = {}
    for particion in ("entrenamiento", "evaluacion"):
        filas_base = {fila["play_id"]: fila for fila in base["jugadas"]
                      if fila["categoria"] == categoria and fila["particion"] == particion}
        filas_variante = {fila["play_id"]: fila for fila in variante["jugadas"]
                          if fila["categoria"] == categoria and fila["particion"] == particion}
        comparables = [
            (fila, filas_variante[play_id]) for play_id, fila in filas_base.items()
            if play_id in filas_variante
            and contribuciones_base[play_id]["dominante"] is not None
            and contribuciones_variante[play_id]["dominante"] in equivalencias
        ]
        acuerdos[f"jugadas_comparables_{particion}"] = len(comparables)
        acuerdos[f"acuerdo_dominante_{particion}"] = sum(
            contribuciones_base[base_fila["play_id"]]["dominante"]
            == equivalencias[contribuciones_variante[variante_fila["play_id"]]["dominante"]]
            for base_fila, variante_fila in comparables
        ) / len(comparables) if comparables else None
    return {
        "ngramas_compartidos": len(comunes),
        "pares_componentes": pares,
        "similitud_coseno_media": float(np.mean([
            par["similitud_coseno"] for par in pares
        ])) if pares else None,
        **acuerdos,
    }


def evaluar_corridas(
    corridas: Mapping[str, str | Path],
    ruta_salida: str | Path,
    *,
    candidato: str,
    cuantil_error: float,
    decision: Decision,
    justificacion: str,
) -> dict[str, Any]:
    """Compara corridas y aplica un umbral aprendido solo en entrenamiento."""
    salida = Path(ruta_salida)
    if salida.exists():
        raise FileExistsError(f"La salida ya existe: {salida}")
    if not corridas or candidato not in corridas:
        raise ValueError("El candidato elegido debe pertenecer a las corridas")
    if not 0 < cuantil_error < 1 or not justificacion.strip():
        raise ValueError("Se requiere cuantil entre cero y uno y justificación")
    if decision not in ("aceptado", "rechazado"):
        raise ValueError("La decisión debe ser aceptado o rechazado")

    resultados = {
        nombre: json.loads((Path(ruta) / "resultados.json").read_text(encoding="utf-8"))
        for nombre, ruta in corridas.items()
    }
    referencia = {
        fila["play_id"]: fila["categoria"] for fila in resultados[candidato]["jugadas"]
    }
    if len(referencia) != len(resultados[candidato]["jugadas"]):
        raise ValueError("El candidato tiene play_id duplicados")
    for nombre, resultado in resultados.items():
        if {fila["play_id"]: fila["categoria"] for fila in resultado["jugadas"]} != referencia:
            raise ValueError(f"La corrida {nombre} no cubre las mismas jugadas")

    elegido = resultados[candidato]
    metadatos = _metadatos(elegido)
    contribuciones = {
        nombre: _contribuciones(resultado, Path(corridas[nombre]))
        for nombre, resultado in resultados.items()
    }
    umbrales = _umbrales(elegido, cuantil_error)
    jugadas = []
    for fila in elegido["jugadas"]:
        umbral = umbrales[fila["categoria"]]
        sin_patron = (fila["error_reconstruccion"] is None
                      or fila["error_reconstruccion"] > umbral
                      or contribuciones[candidato][fila["play_id"]]["dominante"] is None)
        dominante = contribuciones[candidato][fila["play_id"]]["dominante"]
        etiqueta = "Sin patrón claro" if sin_patron else dominante
        jugadas.append({
            "play_id": fila["play_id"], "categoria": fila["categoria"],
            "partido_id": fila["partido_id"], "particion": fila["particion"],
            "clase_principal": etiqueta if decision == "aceptado" else None,
            "etiqueta_exploratoria": etiqueta,
            "componente_dominante": dominante,
            "pesos": fila["pesos"],
            "contribuciones_normalizadas": contribuciones[candidato][fila["play_id"]]["normalizadas"],
            "error_reconstruccion": fila["error_reconstruccion"],
            "cobertura_vocabulario": fila["cobertura_vocabulario"],
            "umbral_error": umbral,
            "manifiesto": fila["manifiesto"], "archivo": fila["archivo"],
        })
    evaluacion: dict[str, Any] = {
        "estado": "clasificacion_validada" if decision == "aceptado" else "diagnostico_exploratorio",
        "decision": decision, "justificacion": justificacion,
        "candidato": candidato, "cuantil_error": cuantil_error,
        "umbrales_error": umbrales,
        "convencion_pesos": (
            "Contribución W_k × norma L2(H_k), normalizada por suma de jugada; "
            "no es probabilidad calibrada. Se conservan los W crudos."
        ),
        "partidos_entrenamiento": elegido["partidos_entrenamiento"],
        "partidos_evaluacion": elegido["partidos_evaluacion"],
        "corridas": {
            nombre: {"ruta_resultados": str(Path(ruta) / "resultados.json"),
                     "parametros": resultados[nombre]["parametros"],
                     "por_categoria": {
                         categoria: _diagnosticar_categoria(
                             resultados[nombre], Path(ruta), categoria, metadatos, cuantil_error,
                             contribuciones[nombre],
                         ) for categoria in CATEGORIAS
                     }}
            for nombre, ruta in corridas.items()
        },
        "comparaciones": {
            nombre: {
                categoria: _comparar_componentes(
                    elegido, Path(corridas[candidato]), resultado, Path(corridas[nombre]),
                    categoria, contribuciones[candidato], contribuciones[nombre],
                ) for categoria in CATEGORIAS
            }
            for nombre, resultado in resultados.items() if nombre != candidato
        },
        "jugadas": jugadas,
    }
    salida.mkdir(parents=True)
    with (salida / "evaluacion.json").open("w", encoding="utf-8") as archivo:
        json.dump(evaluacion, archivo, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
    return evaluacion


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluar corridas NMF y documentar la decisión")
    parser.add_argument("ruta_salida", type=Path)
    parser.add_argument("--corrida", action="append", required=True, metavar="NOMBRE=RUTA")
    parser.add_argument("--candidato", required=True)
    parser.add_argument("--cuantil-error", type=float, required=True)
    parser.add_argument("--decision", choices=("aceptado", "rechazado"), required=True)
    parser.add_argument("--justificacion", required=True)
    args = parser.parse_args()
    corridas = {}
    for entrada in args.corrida:
        if "=" not in entrada:
            parser.error("Cada --corrida debe ser NOMBRE=RUTA")
        nombre, ruta = entrada.split("=", 1)
        if not nombre or not ruta or nombre in corridas:
            parser.error("Cada corrida debe tener nombre y ruta únicos")
        corridas[nombre] = Path(ruta)
    resultado = evaluar_corridas(
        corridas, args.ruta_salida, candidato=args.candidato,
        cuantil_error=args.cuantil_error, decision=args.decision,
        justificacion=args.justificacion,
    )
    print(json.dumps({
        "estado": resultado["estado"], "decision": resultado["decision"],
        "candidato": resultado["candidato"], "jugadas": len(resultado["jugadas"]),
        "ruta_salida": str(args.ruta_salida),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
