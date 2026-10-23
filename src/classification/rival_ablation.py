"""Ablación exploratoria de las acciones rivales en NMF (#16)."""

import json
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
from scipy.optimize import linear_sum_assignment

from src.classification.exploratory_nmf import CATEGORIAS, clasificar_muestra
from src.classification.features import ConfiguracionEspacial


ARTEFACTOS_POTENCIALES = ("Ball_Receipt*", "Dribbled_Past", "Foul_Won")


def _fraccion_masa(pesos: np.ndarray, vocabulario: list[str], marcadores: tuple[str, ...]) -> float:
    total = float(pesos.sum())
    if total == 0:
        return 0.0
    return float(sum(peso for peso, termino in zip(pesos, vocabulario)
                     if any(marcador in termino for marcador in marcadores)) / total)


def _fichas(resultado: dict[str, Any], categoria: str, artefacto: dict[str, Any]) -> list[dict[str, Any]]:
    vocabulario = resultado["modelos"][categoria]["vocabulario"]
    componentes = artefacto["nmf"].components_
    filas = [fila for fila in resultado["jugadas"] if fila["categoria"] == categoria]
    fichas = []
    for identificador, pesos in zip(artefacto["componentes_ids"], componentes):
        indices = np.argsort(-pesos, kind="stable")[:10]
        fraccion_rival = _fraccion_masa(pesos, vocabulario, ("rival_",))
        fraccion_interrupcion = _fraccion_masa(pesos, vocabulario, ("rival_interrupcion:",))
        fraccion_recuperacion = _fraccion_masa(pesos, vocabulario, ("rival_recuperacion:",))
        fraccion_contextual = _fraccion_masa(pesos, vocabulario, ARTEFACTOS_POTENCIALES)
        fraccion_sin_dato = _fraccion_masa(pesos, vocabulario, ("sin_dato",))
        ejemplos = sorted(filas, key=lambda fila: (-fila["pesos"][identificador], fila["play_id"]))[:3]
        fichas.append({
            "componente": identificador,
            "ngramas_destacados": [
                {"ngramas": vocabulario[i], "peso": float(pesos[i])} for i in indices
            ],
            "ejemplos": [
                {"play_id": fila["play_id"], "particion": fila["particion"],
                 "peso": fila["pesos"][identificador], "manifiesto": fila["manifiesto"],
                 "archivo": fila["archivo"]} for fila in ejemplos
            ],
            "fraccion_masa_rival": fraccion_rival,
            "fraccion_masa_interrupcion": fraccion_interrupcion,
            "fraccion_masa_recuperacion": fraccion_recuperacion,
            "fraccion_masa_contextual": fraccion_contextual,
            "fraccion_masa_sin_dato": fraccion_sin_dato,
            "posible_dominio_rival": fraccion_rival >= 0.5,
            "posible_dominio_interrupcion": fraccion_interrupcion >= 0.5,
            "posible_artefacto_registro": fraccion_contextual >= 0.5 or fraccion_sin_dato >= 0.5,
        })
    return fichas


def _emparejar(
    con: dict[str, Any], sin: dict[str, Any],
    modelo_con: dict[str, Any], modelo_sin: dict[str, Any], categoria: str,
) -> tuple[list[dict[str, Any]], int]:
    vocab_con = con["modelos"][categoria]["vocabulario"]
    vocab_sin = sin["modelos"][categoria]["vocabulario"]
    indices_con = {ng: i for i, ng in enumerate(vocab_con)}
    indices_sin = {ng: i for i, ng in enumerate(vocab_sin)}
    comunes = sorted(set(indices_con) & set(indices_sin))
    comunes = [ng for ng in comunes if "rival_" not in ng]
    h_con = modelo_con["nmf"].components_
    h_sin = modelo_sin["nmf"].components_
    if comunes:
        a = h_con[:, [indices_con[ng] for ng in comunes]]
        b = h_sin[:, [indices_sin[ng] for ng in comunes]]
        a = a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)
        b = b / np.maximum(np.linalg.norm(b, axis=1, keepdims=True), 1e-12)
        similitud = a @ b.T
    else:
        similitud = np.zeros((len(h_con), len(h_sin)))
    indices_a, indices_b = linear_sum_assignment(-similitud)
    pares = [
        {"con_rivales": modelo_con["componentes_ids"][i],
         "sin_rivales": modelo_sin["componentes_ids"][j],
         "similitud_coseno_rasgos_compartidos": float(similitud[i, j]),
         "comparable": bool(comunes and similitud[i, j] > 0)}
        for i, j in zip(indices_a, indices_b)
    ]
    return pares, len(comunes)


def _contribuciones(fila: dict[str, Any], normas: dict[str, float]) -> dict[str, float] | None:
    """Reparte la magnitud reconstruida por componente, invariante a la escala W/H."""
    magnitudes = {nombre: peso * normas[nombre] for nombre, peso in fila["pesos"].items()}
    total = sum(magnitudes.values())
    return {nombre: valor / total for nombre, valor in magnitudes.items()} if total > 0 else None


def comparar_rivales(
    ruta_corpus: str | Path,
    ruta_salida: str | Path,
    *,
    max_partidos: int | None = 12,
    partidos_reservados: Iterable[int] | None = None,
    n_componentes: int = 4,
    semilla: int = 42,
    configuracion_espacial: ConfiguracionEspacial | None = None,
) -> dict[str, Any]:
    """Ajusta dos variantes sobre los mismos partidos y compara contenido y pesos."""
    salida = Path(ruta_salida)
    if salida.exists():
        raise FileExistsError(f"La salida ya existe: {salida}")
    con = clasificar_muestra(
        ruta_corpus, salida / "con_rivales", max_partidos=max_partidos,
        partidos_reservados=partidos_reservados, n_componentes=n_componentes,
        semilla=semilla, configuracion_espacial=configuracion_espacial,
        modo_rivales="con_rivales",
    )
    sin = clasificar_muestra(
        ruta_corpus, salida / "sin_rivales", max_partidos=max_partidos,
        partidos_reservados=con["partidos_evaluacion"], n_componentes=n_componentes,
        semilla=semilla, configuracion_espacial=configuracion_espacial,
        modo_rivales="sin_rivales",
    )
    if [(j["play_id"], j["particion"]) for j in con["jugadas"]] != [
        (j["play_id"], j["particion"]) for j in sin["jugadas"]
    ]:
        raise ValueError("Las variantes no concilian en cobertura o partición")
    comparacion: dict[str, Any] = {
        "estado": "exploratorio", "jugadas": len(con["jugadas"]),
        "partidos_entrenamiento": con["partidos_entrenamiento"],
        "partidos_evaluacion": con["partidos_evaluacion"],
        "escala_contribuciones": "W_k × norma L2(H_k), normalizada por suma de fila; no es probabilidad calibrada",
        "emparejamiento": "coseno de H en n-gramas no rivales compartidos; asignación óptima",
        "umbral_alerta_masa": 0.5,
        "variantes": {
            "con_rivales": "con_rivales/resultados.json",
            "sin_rivales": "sin_rivales/resultados.json",
        },
        "por_categoria": {}, "cambios_contribuciones": [],
    }
    pares_por_categoria = {}
    normas_por_categoria = {}
    for categoria in CATEGORIAS:
        modelo_con = joblib.load(salida / "con_rivales" / con["modelos"][categoria]["archivo_modelo"])
        modelo_sin = joblib.load(salida / "sin_rivales" / sin["modelos"][categoria]["archivo_modelo"])
        normas_por_categoria[categoria] = {
            "con_rivales": dict(zip(modelo_con["componentes_ids"],
                                    np.linalg.norm(modelo_con["nmf"].components_, axis=1))),
            "sin_rivales": dict(zip(modelo_sin["componentes_ids"],
                                    np.linalg.norm(modelo_sin["nmf"].components_, axis=1))),
        }
        pares, n_comunes = _emparejar(con, sin, modelo_con, modelo_sin, categoria)
        fichas_con = _fichas(con, categoria, modelo_con)
        fichas_sin = _fichas(sin, categoria, modelo_sin)
        por_id_con = {f["componente"]: f for f in fichas_con}
        por_id_sin = {f["componente"]: f for f in fichas_sin}
        for par in pares:
            actual = por_id_con[par["con_rivales"]]
            base = por_id_sin[par["sin_rivales"]]
            destacados_actual = {x["ngramas"] for x in actual["ngramas_destacados"]}
            destacados_base = {x["ngramas"] for x in base["ngramas_destacados"]}
            par["ngramas_nuevos_destacados"] = sorted(destacados_actual - destacados_base)
            par["ngramas_que_salen"] = sorted(destacados_base - destacados_actual)
            par["ejemplos_nuevos"] = sorted(
                {x["play_id"] for x in actual["ejemplos"]}
                - {x["play_id"] for x in base["ejemplos"]}
            )
        pares_por_categoria[categoria] = pares
        comparacion["por_categoria"][categoria] = {
            "ngramas_no_rivales_compartidos": n_comunes,
            "con_rivales": fichas_con, "sin_rivales": fichas_sin,
            "componentes_emparejados": pares,
        }
    for fila_con, fila_sin in zip(con["jugadas"], sin["jugadas"]):
        normas = normas_por_categoria[fila_con["categoria"]]
        pesos_con = _contribuciones(fila_con, normas["con_rivales"])
        pesos_sin = _contribuciones(fila_sin, normas["sin_rivales"])
        cambios = []
        for par in pares_por_categoria[fila_con["categoria"]]:
            actual = pesos_con[par["con_rivales"]] if pesos_con is not None else None
            base = pesos_sin[par["sin_rivales"]] if pesos_sin is not None else None
            cambios.append({
                "con_rivales": par["con_rivales"], "sin_rivales": par["sin_rivales"],
                "contribucion_con_rivales": actual, "contribucion_sin_rivales": base,
                "cambio": actual - base if actual is not None and base is not None
                and par["comparable"] else None,
            })
        comparacion["cambios_contribuciones"].append({
            "play_id": fila_con["play_id"], "categoria": fila_con["categoria"],
            "particion": fila_con["particion"], "componentes": cambios,
        })
    with (salida / "comparacion.json").open("w", encoding="utf-8") as archivo:
        json.dump(comparacion, archivo, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
    return comparacion
