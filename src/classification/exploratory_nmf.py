"""Piloto NMF por categoría sobre jugadas preparadas (#14)."""

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping, Sequence

import joblib
import numpy as np
from sklearn.decomposition import NMF
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

from src.classification.features import (
    ConfiguracionEspacial, DISTANCIA_CORTA, DISTANCIA_MEDIA, ModoRivales, UMBRAL_AVANCE_X,
    representar_jugada, verificar_orientacion,
)


CATEGORIAS = ("balon_parado", "juego_abierto")


def _muestra_por_temporada(manifiestos: list[Path], limite: int) -> list[Path]:
    temporadas: dict[tuple[str, str], list[Path]] = {}
    for ruta in manifiestos:
        temporadas.setdefault((ruta.parent.parent.parent.name, ruta.parent.parent.name), []).append(ruta)
    muestra: list[Path] = []
    while len(muestra) < limite and any(temporadas.values()):
        for rutas in temporadas.values():
            if rutas and len(muestra) < limite:
                muestra.append(rutas.pop(0))
    return sorted(muestra)


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
                "eventos": eventos,
                "protagonista": detalle["equipo_protagonista"],
            })
    return jugadas


def _errores(matriz: Any, pesos: np.ndarray, componentes: np.ndarray) -> np.ndarray:
    """Norma L2 de cada residuo sin densificar la matriz de características."""
    norma_datos = np.asarray(matriz.multiply(matriz).sum(axis=1)).ravel()
    norma_reconstruccion = np.sum((pesos @ (componentes @ componentes.T)) * pesos, axis=1)
    producto = np.sum(np.asarray(matriz @ componentes.T) * pesos, axis=1)
    return np.sqrt(np.maximum(0, norma_datos + norma_reconstruccion - 2 * producto))


def proyectar_rasgos(
    rasgos: Sequence[Mapping[str, float]], vectorizador: DictVectorizer,
    transformador: TfidfTransformer, modelo: NMF,
) -> tuple[np.ndarray, np.ndarray, list[float]]:
    """Aplica el mismo vectorizador y NMF a jugadas del piloto o del lote."""
    conteos = vectorizador.transform(rasgos)
    matriz = transformador.transform(conteos)
    pesos = modelo.transform(matriz)
    errores = _errores(matriz, pesos, modelo.components_)
    cobertura = [
        sum(cantidad for ngrama, cantidad in fila.items()
            if ngrama in vectorizador.vocabulary_) / sum(fila.values())
        if fila else 0.0
        for fila in rasgos
    ]
    return pesos, errores, cobertura


def clasificar_muestra(
    ruta_corpus: str | Path,
    ruta_salida: str | Path,
    *,
    max_partidos: int | None = 12,
    partidos_reservados: Iterable[int] | None = None,
    n_componentes: int = 4,
    semilla: int = 42,
    configuracion_espacial: ConfiguracionEspacial | None = None,
    modo_rivales: ModoRivales = "con_rivales",
    posicion: bool = False,
    ritmo: bool = False,
    muestra_diversa: bool = False,
    ponderacion: Literal["tfidf_l2", "frecuencia_l2"] = "tfidf_l2",
) -> dict[str, Any]:
    """Ajusta dos NMF independientes y puntúa entrenamiento y partidos reservados.

    Las etiquetas son exploratorias; ningún umbral de validez táctica se aplica aquí.
    El directorio de salida debe ser nuevo para conservar cada corrida íntegra.
    """
    corpus, salida = Path(ruta_corpus), Path(ruta_salida)
    if n_componentes < 1 or max_partidos is not None and max_partidos < 2:
        raise ValueError("Se requieren al menos dos partidos y un componente")
    if ponderacion not in ("tfidf_l2", "frecuencia_l2"):
        raise ValueError(f"Ponderación desconocida: {ponderacion}")
    if salida.exists():
        raise FileExistsError(f"La salida ya existe: {salida}")
    manifiestos = sorted(corpus.glob("competicion_*/temporada_*/partido_*/manifiesto.json"))
    if max_partidos is not None:
        manifiestos = (_muestra_por_temporada(manifiestos, max_partidos) if muestra_diversa
                      else manifiestos[:max_partidos])
    if len(manifiestos) < 2:
        raise ValueError("Se requieren al menos dos partidos preparados")
    jugadas = _leer_jugadas(corpus, manifiestos)
    verificacion = verificar_orientacion(jugadas) if configuracion_espacial else None
    for jugada in jugadas:
        jugada["ngramas"], metadatos = representar_jugada(
            jugada.pop("eventos"), jugada.pop("protagonista"), configuracion_espacial,
            modo_rivales=modo_rivales, posicion=posicion, ritmo=ritmo,
        )
        jugada.update(metadatos)
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

    representacion = "_".join(
        nombre for activo, nombre in (
            (configuracion_espacial is not None, "espacial"),
            (posicion, "posicional"), (ritmo, "temporal"),
        ) if activo
    ) or "base"

    resultado: dict[str, Any] = {
        "estado": "exploratorio",
        "ruta_corpus": str(corpus),
        "parametros": {
            "max_partidos": max_partidos, "n_componentes_solicitados": n_componentes,
            "muestra_diversa": muestra_diversa,
            "semilla": semilla, "particion": "por_partido",
            "vectorizacion": f"conteos_ngramas_2_3_{ponderacion}",
            "ponderacion": ponderacion,
            "tokenizacion": "acciones_desplazamientos_lances_v2",
            "modo_rivales": modo_rivales,
            "posicion": posicion,
            "ritmo": ritmo,
            "umbrales_ritmo_segundos": {"rapido_hasta": 1, "medio_hasta": 3},
            "intervalo_ritmo": "diferencia_positiva_entre_inicios_observados",
            "umbrales_movimiento": {
                "avance_x": UMBRAL_AVANCE_X,
                "distancia_corta": DISTANCIA_CORTA,
                "distancia_media": DISTANCIA_MEDIA,
            },
            "nmf_init": "nndsvda", "nmf_max_iter": 500,
            "escala_pesos": f"W cruda sobre {ponderacion}; no son probabilidades",
            "representacion": representacion,
            "cuadricula": None if configuracion_espacial is None else {
                "zonas_x": configuracion_espacial.zonas_x,
                "zonas_y": configuracion_espacial.zonas_y,
            },
            "peso_espacial": None if configuracion_espacial is None else configuracion_espacial.peso,
        },
        "verificacion_orientacion": verificacion,
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
        tfidf = TfidfTransformer(norm="l2", use_idf=ponderacion == "tfidf_l2")
        matriz_train = tfidf.fit_transform(conteos_train)
        componentes = min(n_componentes, *matriz_train.shape)
        modelo = NMF(n_components=componentes, init="nndsvda", random_state=semilla, max_iter=500)
        modelo.fit(matriz_train)
        pesos, errores, cobertura = proyectar_rasgos(
            [fila["ngramas"] for fila in filas], vectorizador, tfidf, modelo,
        )
        componentes_ids = [f"{categoria}_c{i:02d}" for i in range(1, componentes + 1)]
        archivo_modelo = f"modelo_{categoria}.joblib"
        resultado["modelos"][categoria] = {
            "componentes": componentes_ids,
            "n_componentes": componentes,
            "vocabulario": vectorizador.get_feature_names_out().tolist(),
            "archivo_modelo": archivo_modelo,
            "jugadas_entrenamiento": len(filas_train),
            "jugadas_evaluacion": len(filas) - len(filas_train),
            "matriz_entrenamiento": {
                "filas": int(matriz_train.shape[0]),
                "columnas": int(matriz_train.shape[1]),
                "elementos_no_nulos": int(matriz_train.nnz),
            },
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
                "banda_original": fila["banda_original"], "reflejada": fila["reflejada"],
            })
    resultado["jugadas"].sort(key=lambda fila: fila["play_id"])
    salida.mkdir(parents=True)
    for nombre, artefacto in artefactos.items():
        joblib.dump(artefacto, salida / nombre)
    with (salida / "resultados.json").open("w", encoding="utf-8") as archivo:
        json.dump(resultado, archivo, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
    return resultado


def comparar_representaciones(
    ruta_corpus: str | Path,
    ruta_salida: str | Path,
    *,
    max_partidos: int | None = 12,
    partidos_reservados: Iterable[int] | None = None,
    n_componentes: int = 4,
    semilla: int = 42,
    zonas_x: int = 6,
    zonas_y: int = 4,
    peso_espacial: float = 1.0,
) -> dict[str, Any]:
    """Ejecuta dos variantes con la misma partición y resume su ajuste exploratorio."""
    salida = Path(ruta_salida)
    if salida.exists():
        raise FileExistsError(f"La salida ya existe: {salida}")
    config = ConfiguracionEspacial(zonas_x, zonas_y, peso_espacial)
    # La corrida espacial valida la orientación antes de producir rasgos.
    espacial = clasificar_muestra(
        ruta_corpus, salida / "espacial", max_partidos=max_partidos,
        partidos_reservados=partidos_reservados, n_componentes=n_componentes,
        semilla=semilla, configuracion_espacial=config,
    )
    base = clasificar_muestra(
        ruta_corpus, salida / "base", max_partidos=max_partidos,
        partidos_reservados=espacial["partidos_evaluacion"],
        n_componentes=n_componentes, semilla=semilla,
    )
    return _guardar_comparacion(salida, (("base", base), ("espacial", espacial)))


def comparar_posiciones(
    ruta_corpus: str | Path,
    ruta_salida: str | Path,
    *,
    max_partidos: int | None = 12,
    partidos_reservados: Iterable[int] | None = None,
    n_componentes: int = 4,
    semilla: int = 42,
    configuracion_espacial: ConfiguracionEspacial | None = None,
) -> dict[str, Any]:
    """Compara n-gramas locales con la variante que registra su tercio."""
    salida = Path(ruta_salida)
    if salida.exists():
        raise FileExistsError(f"La salida ya existe: {salida}")
    base = clasificar_muestra(
        ruta_corpus, salida / "base", max_partidos=max_partidos,
        partidos_reservados=partidos_reservados, n_componentes=n_componentes,
        semilla=semilla, configuracion_espacial=configuracion_espacial,
    )
    posicional = clasificar_muestra(
        ruta_corpus, salida / "posicional", max_partidos=max_partidos,
        partidos_reservados=base["partidos_evaluacion"], n_componentes=n_componentes,
        semilla=semilla, configuracion_espacial=configuracion_espacial,
        posicion=True,
    )
    return _guardar_comparacion(salida, (("base", base), ("posicional", posicional)))


def comparar_cronologia_y_ritmo(
    ruta_corpus: str | Path,
    ruta_salida: str | Path,
    *,
    max_partidos: int | None = 12,
    partidos_reservados: Iterable[int] | None = None,
    n_componentes: int = 4,
    semilla: int = 42,
    muestra_diversa: bool = True,
) -> dict[str, Any]:
    """Compara base, posición y ritmo en las mismas jugadas y particiones."""
    salida = Path(ruta_salida)
    if salida.exists():
        raise FileExistsError(f"La salida ya existe: {salida}")
    base = clasificar_muestra(
        ruta_corpus, salida / "base", max_partidos=max_partidos,
        partidos_reservados=partidos_reservados, n_componentes=n_componentes,
        semilla=semilla, muestra_diversa=muestra_diversa,
    )
    posicional = clasificar_muestra(
        ruta_corpus, salida / "posicional", max_partidos=max_partidos,
        partidos_reservados=base["partidos_evaluacion"], n_componentes=n_componentes,
        semilla=semilla, posicion=True, muestra_diversa=muestra_diversa,
    )
    temporal = clasificar_muestra(
        ruta_corpus, salida / "temporal", max_partidos=max_partidos,
        partidos_reservados=base["partidos_evaluacion"], n_componentes=n_componentes,
        semilla=semilla, ritmo=True, muestra_diversa=muestra_diversa,
    )
    return _guardar_comparacion(salida, (
        ("base", base), ("posicional", posicional), ("temporal", temporal),
    ))


def _guardar_comparacion(
    salida: Path, variantes: tuple[tuple[str, dict[str, Any]], ...],
) -> dict[str, Any]:
    base = variantes[0][1]
    referencia = [(j["play_id"], j["particion"]) for j in base["jugadas"]]
    for _, variante in variantes[1:]:
        if referencia != [(j["play_id"], j["particion"]) for j in variante["jugadas"]]:
            raise ValueError("Las variantes no concilian en cobertura o partición")
    comparacion: dict[str, Any] = {
        "estado": "exploratorio",
        "jugadas": len(base["jugadas"]),
        "partidos_entrenamiento": base["partidos_entrenamiento"],
        "partidos_evaluacion": base["partidos_evaluacion"],
        "variantes": {
            nombre: {
                "ruta_resultados": f"{nombre}/resultados.json",
                "por_categoria": {
                    categoria: _resumir_categoria(resultado, categoria)
                    for categoria in CATEGORIAS
                },
            }
            for nombre, resultado in variantes
        },
    }
    with (salida / "comparacion.json").open("w", encoding="utf-8") as archivo:
        json.dump(comparacion, archivo, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
    return comparacion


def _resumir_categoria(resultado: dict[str, Any], categoria: str) -> dict[str, Any]:
    filas = [fila for fila in resultado["jugadas"] if fila["categoria"] == categoria]
    evaluacion = [fila for fila in filas if fila["particion"] == "evaluacion"]
    errores = [fila["error_reconstruccion"] for fila in evaluacion
               if fila["error_reconstruccion"] is not None]
    return {
        "jugadas": len(filas),
        "jugadas_evaluacion": len(evaluacion),
        "error_medio_evaluacion": float(np.mean(errores)) if errores else None,
        "cobertura_media_evaluacion": float(np.mean([
            fila["cobertura_vocabulario"] for fila in evaluacion
        ])) if evaluacion else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Piloto NMF exploratorio por categoría")
    parser.add_argument("ruta_corpus", type=Path)
    parser.add_argument("ruta_salida", type=Path)
    parser.add_argument("--max-partidos", type=int, default=12)
    parser.add_argument("--componentes", type=int, default=4)
    parser.add_argument("--semilla", type=int, default=42)
    parser.add_argument("--reservar-partido", type=int, action="append", dest="reservados")
    parser.add_argument("--comparar-espacio", action="store_true")
    parser.add_argument("--comparar-rivales", action="store_true")
    parser.add_argument("--comparar-posicion", action="store_true")
    parser.add_argument("--comparar-ritmo", action="store_true")
    parser.add_argument("--zonas-x", type=int, default=6)
    parser.add_argument("--zonas-y", type=int, default=4)
    parser.add_argument("--peso-espacial", type=float, default=1.0)
    args = parser.parse_args()
    if sum((args.comparar_espacio, args.comparar_rivales,
            args.comparar_posicion, args.comparar_ritmo)) > 1:
        parser.error("Elige una sola comparación")
    if args.comparar_espacio:
        resumen = comparar_representaciones(
            args.ruta_corpus, args.ruta_salida,
            max_partidos=args.max_partidos, partidos_reservados=args.reservados,
            n_componentes=args.componentes, semilla=args.semilla,
            zonas_x=args.zonas_x, zonas_y=args.zonas_y, peso_espacial=args.peso_espacial,
        )
        total_jugadas = resumen["jugadas"]
    elif args.comparar_rivales:
        from src.classification.rival_ablation import comparar_rivales

        resumen = comparar_rivales(
            args.ruta_corpus, args.ruta_salida,
            max_partidos=args.max_partidos, partidos_reservados=args.reservados,
            n_componentes=args.componentes, semilla=args.semilla,
            configuracion_espacial=ConfiguracionEspacial(
                args.zonas_x, args.zonas_y, args.peso_espacial,
            ),
        )
        total_jugadas = resumen["jugadas"]
    elif args.comparar_posicion:
        resumen = comparar_posiciones(
            args.ruta_corpus, args.ruta_salida,
            max_partidos=args.max_partidos, partidos_reservados=args.reservados,
            n_componentes=args.componentes, semilla=args.semilla,
        )
        total_jugadas = resumen["jugadas"]
    elif args.comparar_ritmo:
        resumen = comparar_cronologia_y_ritmo(
            args.ruta_corpus, args.ruta_salida,
            max_partidos=args.max_partidos, partidos_reservados=args.reservados,
            n_componentes=args.componentes, semilla=args.semilla,
        )
        total_jugadas = resumen["jugadas"]
    else:
        resumen = clasificar_muestra(
            args.ruta_corpus, args.ruta_salida,
            max_partidos=args.max_partidos, partidos_reservados=args.reservados,
            n_componentes=args.componentes, semilla=args.semilla,
        )
        total_jugadas = len(resumen["jugadas"])
    print(json.dumps({
        "estado": resumen["estado"], "jugadas": total_jugadas,
        "partidos_entrenamiento": len(resumen["partidos_entrenamiento"]),
        "partidos_evaluacion": len(resumen["partidos_evaluacion"]),
        "ruta_salida": str(args.ruta_salida),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
