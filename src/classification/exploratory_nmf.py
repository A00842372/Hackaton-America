"""Piloto NMF por categoría sobre jugadas preparadas (#14)."""

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
from sklearn.decomposition import NMF
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfTransformer


CATEGORIAS = ("balon_parado", "juego_abierto")
UMBRAL_AVANCE_X = 5.0
DISTANCIA_CORTA = 10.0
DISTANCIA_MEDIA = 30.0
AUXILIARES = {
    "Ball Receipt*", "Camera On", "Camera Off", "Tactical Shift",
    "Starting XI", "Half Start", "Half End",
}
AUXILIARES_VINCULADOS = {"Dribbled Past", "Foul Won"}


def _nombre(valor: Any) -> str | None:
    if isinstance(valor, dict):
        return valor.get("name")
    return valor if isinstance(valor, str) else None


def _punto(valor: Any) -> tuple[float, float] | None:
    if not isinstance(valor, list) or len(valor) < 2:
        return None
    x, y = valor[:2]
    if not all(isinstance(n, (int, float)) and math.isfinite(n) for n in (x, y)):
        return None
    return float(x), float(y)


def _accion(evento: dict[str, Any], protagonista: str) -> str | None:
    tipo = _nombre(evento.get("type"))
    if not tipo or tipo in AUXILIARES or (
        tipo in AUXILIARES_VINCULADOS and evento.get("related_events")
    ):
        return None
    equipo = _nombre(evento.get("team"))
    actor = "propio" if equipo == protagonista else "rival" if equipo else "desconocido"
    token = f"{actor}:{tipo.replace(' ', '_')}"
    origen = _punto(evento.get("location"))
    detalle = evento.get(tipo.lower().replace(" ", "_"))
    destino = _punto(detalle.get("end_location")) if isinstance(detalle, dict) else None
    if origen is not None and destino is not None:
        avance = destino[0] - origen[0]
        distancia = math.dist(origen, destino)
        direccion = (
            "avance" if avance > UMBRAL_AVANCE_X else
            "retroceso" if avance < -UMBRAL_AVANCE_X else "lateral"
        )
        tramo = (
            "corto" if distancia < DISTANCIA_CORTA else
            "medio" if distancia < DISTANCIA_MEDIA else "largo"
        )
        token += f":{direccion}:{tramo}"
    return token


def _ngramas(eventos: list[dict[str, Any]], protagonista: str) -> Counter[str]:
    acciones = [accion for evento in eventos if (accion := _accion(evento, protagonista))]
    return Counter(
        f"n{largo}:" + " → ".join(acciones[inicio:inicio + largo])
        for largo in (2, 3)
        for inicio in range(len(acciones) - largo + 1)
    )


def _leer_jugadas(corpus: Path, manifiestos: list[Path]) -> list[dict[str, Any]]:
    jugadas = []
    ids: set[str] = set()
    for ruta in manifiestos:
        manifiesto = json.loads(ruta.read_text(encoding="utf-8"))
        if manifiesto.get("parametros", {}).get("dividir_fases_clasificables") is not True:
            raise ValueError(f"El partido no tiene categorías de clasificación: {ruta}")
        if len(manifiesto["jugadas"]) != manifiesto["total_jugadas_elegibles"]:
            raise ValueError(f"El manifiesto no concilia sus jugadas: {ruta}")
        for detalle in manifiesto["jugadas"]:
            categoria = detalle.get("categoria_clasificacion")
            if categoria not in CATEGORIAS:
                raise ValueError(f"Categoría inválida en {ruta}: {categoria}")
            play_id = detalle["play_id"]
            if play_id in ids:
                raise ValueError(f"play_id duplicado: {play_id}")
            ids.add(play_id)
            archivo = detalle["archivo"]
            if Path(archivo).name != archivo or not archivo.endswith(".json"):
                raise ValueError(f"Archivo de jugada inválido: {archivo}")
            bruto = json.loads((ruta.parent / archivo).read_text(encoding="utf-8"))
            columnas = bruto["columns"]
            if columnas != manifiesto["columnas"] or len(bruto["data"]) != detalle["total_eventos"]:
                raise ValueError(f"Archivo y manifiesto no concilian: {ruta.parent / archivo}")
            eventos = [dict(zip(columnas, fila)) for fila in bruto["data"]]
            if not eventos or eventos[0].get("id") != detalle["primer_evento_id"]:
                raise ValueError(f"Eventos y manifiesto no concilian: {ruta.parent / archivo}")
            jugadas.append({
                "play_id": play_id,
                "partido_id": manifiesto["partido_id"],
                "categoria": categoria,
                "manifiesto": str(ruta.relative_to(corpus)),
                "archivo": archivo,
                "ngramas": _ngramas(eventos, detalle["equipo_protagonista"]),
            })
    return jugadas


def _errores(matriz: Any, pesos: np.ndarray, componentes: np.ndarray) -> np.ndarray:
    """Norma L2 de cada residuo sin densificar la matriz de características."""
    norma_datos = np.asarray(matriz.multiply(matriz).sum(axis=1)).ravel()
    norma_reconstruccion = np.sum((pesos @ (componentes @ componentes.T)) * pesos, axis=1)
    producto = np.sum(np.asarray(matriz @ componentes.T) * pesos, axis=1)
    return np.sqrt(np.maximum(0, norma_datos + norma_reconstruccion - 2 * producto))


def clasificar_muestra(
    ruta_corpus: str | Path,
    ruta_salida: str | Path,
    *,
    max_partidos: int | None = 12,
    partidos_reservados: Iterable[int] | None = None,
    n_componentes: int = 4,
    semilla: int = 42,
) -> dict[str, Any]:
    """Ajusta dos NMF independientes y puntúa entrenamiento y partidos reservados.

    Las etiquetas son exploratorias; ningún umbral de validez táctica se aplica aquí.
    El directorio de salida debe ser nuevo para conservar cada corrida íntegra.
    """
    corpus, salida = Path(ruta_corpus), Path(ruta_salida)
    if n_componentes < 1 or max_partidos is not None and max_partidos < 2:
        raise ValueError("Se requieren al menos dos partidos y un componente")
    if salida.exists():
        raise FileExistsError(f"La salida ya existe: {salida}")
    manifiestos = sorted(corpus.glob("competicion_*/temporada_*/partido_*/manifiesto.json"))
    if max_partidos is not None:
        manifiestos = manifiestos[:max_partidos]
    if len(manifiestos) < 2:
        raise ValueError("Se requieren al menos dos partidos preparados")
    jugadas = _leer_jugadas(corpus, manifiestos)
    partidos = [int(ruta.parent.name.removeprefix("partido_")) for ruta in manifiestos]
    if len(set(partidos)) != len(partidos):
        raise ValueError("La muestra contiene IDs de partido duplicados")
    if partidos_reservados is None:
        rng = np.random.default_rng(semilla)
        cantidad = max(1, math.ceil(len(partidos) * 0.2))
        reservados = set(rng.choice(partidos, size=cantidad, replace=False).tolist())
    else:
        reservados = set(partidos_reservados)
        if not reservados or not reservados.issubset(partidos):
            raise ValueError("Los partidos reservados deben pertenecer a la muestra")
    entrenamiento = sorted(set(partidos) - reservados)
    if not entrenamiento:
        raise ValueError("La muestra debe conservar partidos de entrenamiento")

    resultado: dict[str, Any] = {
        "estado": "exploratorio",
        "ruta_corpus": str(corpus),
        "parametros": {
            "max_partidos": max_partidos, "n_componentes_solicitados": n_componentes,
            "semilla": semilla, "particion": "por_partido",
            "vectorizacion": "conteos_ngramas_2_3_tfidf_l2",
            "tokenizacion": "acciones_desplazamientos_v1",
            "umbrales_movimiento": {
                "avance_x": UMBRAL_AVANCE_X,
                "distancia_corta": DISTANCIA_CORTA,
                "distancia_media": DISTANCIA_MEDIA,
            },
            "nmf_init": "nndsvda", "nmf_max_iter": 500,
            "escala_pesos": "W cruda sobre TF-IDF L2; no son probabilidades",
        },
        "partidos_entrenamiento": entrenamiento,
        "partidos_evaluacion": sorted(reservados),
        "modelos": {},
        "jugadas": [],
    }
    artefactos = {}
    for categoria in CATEGORIAS:
        filas = [fila for fila in jugadas if fila["categoria"] == categoria]
        filas_train = [fila for fila in filas if fila["partido_id"] not in reservados]
        if not filas_train or not filas:
            raise ValueError(f"La categoría {categoria} no tiene jugadas de entrenamiento")
        if any(not fila["ngramas"] for fila in filas):
            raise ValueError(f"Una jugada de {categoria} no tiene n-gramas de 2 o 3 acciones")
        vectorizador = DictVectorizer(sparse=True)
        conteos_train = vectorizador.fit_transform([fila["ngramas"] for fila in filas_train])
        tfidf = TfidfTransformer(norm="l2")
        matriz_train = tfidf.fit_transform(conteos_train)
        componentes = min(n_componentes, *matriz_train.shape)
        modelo = NMF(n_components=componentes, init="nndsvda", random_state=semilla, max_iter=500)
        modelo.fit(matriz_train)
        conteos = vectorizador.transform([fila["ngramas"] for fila in filas])
        matriz = tfidf.transform(conteos)
        pesos = modelo.transform(matriz)
        errores = _errores(matriz, pesos, modelo.components_)
        cobertura = [
            sum(cantidad for ngrama, cantidad in fila["ngramas"].items()
                if ngrama in vectorizador.vocabulary_) / sum(fila["ngramas"].values())
            for fila in filas
        ]
        componentes_ids = [f"{categoria}_c{i:02d}" for i in range(1, componentes + 1)]
        archivo_modelo = f"modelo_{categoria}.joblib"
        resultado["modelos"][categoria] = {
            "componentes": componentes_ids,
            "n_componentes": componentes,
            "vocabulario": vectorizador.get_feature_names_out().tolist(),
            "archivo_modelo": archivo_modelo,
            "jugadas_entrenamiento": len(filas_train),
            "jugadas_evaluacion": len(filas) - len(filas_train),
        }
        artefactos[archivo_modelo] = {
            "vectorizador": vectorizador, "tfidf": tfidf, "nmf": modelo,
            "componentes_ids": componentes_ids,
        }
        for fila, vector, error, fraccion in zip(filas, pesos, errores, cobertura):
            sin_rasgos = fraccion == 0
            resultado["jugadas"].append({
                "play_id": fila["play_id"], "partido_id": fila["partido_id"],
                "categoria": categoria,
                "particion": "evaluacion" if fila["partido_id"] in reservados else "entrenamiento",
                "componente_dominante": None if sin_rasgos else componentes_ids[int(np.argmax(vector))],
                "pesos": {id_: float(peso) for id_, peso in zip(componentes_ids, vector)},
                "error_reconstruccion": None if sin_rasgos else float(error),
                "cobertura_vocabulario": float(fraccion),
                "estado_proyeccion": "sin_rasgos_conocidos" if sin_rasgos else "proyectada",
                "manifiesto": fila["manifiesto"], "archivo": fila["archivo"],
            })
    resultado["jugadas"].sort(key=lambda fila: fila["play_id"])
    salida.mkdir(parents=True)
    for nombre, artefacto in artefactos.items():
        joblib.dump(artefacto, salida / nombre)
    with (salida / "resultados.json").open("w", encoding="utf-8") as archivo:
        json.dump(resultado, archivo, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
    return resultado


def main() -> None:
    parser = argparse.ArgumentParser(description="Piloto NMF exploratorio por categoría")
    parser.add_argument("ruta_corpus", type=Path)
    parser.add_argument("ruta_salida", type=Path)
    parser.add_argument("--max-partidos", type=int, default=12)
    parser.add_argument("--componentes", type=int, default=4)
    parser.add_argument("--semilla", type=int, default=42)
    parser.add_argument("--reservar-partido", type=int, action="append", dest="reservados")
    args = parser.parse_args()
    resultado = clasificar_muestra(
        args.ruta_corpus, args.ruta_salida,
        max_partidos=args.max_partidos, partidos_reservados=args.reservados,
        n_componentes=args.componentes, semilla=args.semilla,
    )
    print(json.dumps({
        "estado": resultado["estado"], "jugadas": len(resultado["jugadas"]),
        "partidos_entrenamiento": len(resultado["partidos_entrenamiento"]),
        "partidos_evaluacion": len(resultado["partidos_evaluacion"]),
        "ruta_salida": str(args.ruta_salida),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
