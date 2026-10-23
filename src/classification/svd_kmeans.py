"""Piloto de clustering SVD + MiniBatchKMeans sobre los rasgos NMF vigentes.

El comando produce evidencia exploratoria. Una decisión táctica explícita es
necesaria antes de publicar etiquetas sobre todo el corpus.
"""

import argparse
import json
import math
import shutil
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from scipy.stats import spearmanr
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import NMF, TruncatedSVD
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import normalize

from src.classification.exploratory_nmf import CATEGORIAS
from src.classification.features import representar_jugada


DIMENSIONES = (32, 64)
NUM_CLUSTERS = (6, 8, 12, 16)
SEMILLAS = (42, 43, 44)
UMBRAL_SILUETA = 0.03
UMBRAL_ARI = 0.75
MIN_FRACCION = 0.01
MAX_FRACCION = 0.40


def _manifiestos(corpus: Path) -> list[dict[str, Any]]:
    rutas = sorted(corpus.glob("competicion_*/temporada_*/partido_*/manifiesto.json"))
    if not rutas:
        raise ValueError(f"No hay manifiestos en {corpus}")
    resultado = []
    for ruta in rutas:
        dato = json.loads(ruta.read_text(encoding="utf-8"))
        if dato.get("parametros", {}).get("dividir_fases_clasificables") is not True:
            raise ValueError(f"Faltan fases clasificables en {ruta}")
        if len(dato["jugadas"]) != dato["total_jugadas_elegibles"]:
            raise ValueError(f"El manifiesto no concilia: {ruta}")
        resultado.append({"ruta": ruta, "dato": dato,
                          "temporada": ruta.parent.parent.name,
                          "cantidad": len(dato["jugadas"])})
    return resultado


def _cuotas(total: int, pesos: dict[str, int]) -> dict[str, int]:
    suma = sum(pesos.values())
    brutas = {clave: total * peso / suma for clave, peso in pesos.items()}
    cuotas = {clave: math.floor(valor) for clave, valor in brutas.items()}
    for clave in sorted(pesos, key=lambda k: (-(brutas[k] - cuotas[k]), k))[:total - sum(cuotas.values())]:
        cuotas[clave] += 1
    return cuotas


def seleccionar_partidos(
    manifiestos: list[dict[str, Any]], *, entrenamiento: int = 30_000,
    evaluacion: int = 6_000, semilla: int = 42,
) -> tuple[list[tuple[dict[str, Any], int]], list[tuple[dict[str, Any], int]]]:
    """Muestrea partidos sin solapamiento, con cuotas por temporada y recorte final."""
    if entrenamiento < 1 or evaluacion < 1:
        raise ValueError("Las muestras deben ser positivas")
    grupos: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in manifiestos:
        grupos[item["temporada"]].append(item)
    pesos = {k: sum(item["cantidad"] for item in v) for k, v in grupos.items()}
    if entrenamiento + evaluacion > sum(pesos.values()):
        raise ValueError("La muestra excede el corpus")
    cuotas_train = _cuotas(entrenamiento, pesos)
    cuotas_eval = _cuotas(evaluacion, pesos)
    rng = np.random.default_rng(semilla)
    train, test = [], []
    for temporada, grupo in sorted(grupos.items()):
        orden = list(rng.permutation(len(grupo)))
        libres = [grupo[i] for i in orden if grupo[i]["cantidad"]]
        for destino, cuota in ((train, cuotas_train[temporada]),
                               (test, cuotas_eval[temporada])):
            restante = cuota
            while restante:
                if not libres:
                    raise ValueError(f"No hay partidos suficientes en {temporada}")
                item = libres.pop()
                tomar = min(item["cantidad"], restante)
                destino.append((item, tomar))
                restante -= tomar
    return train, test


def _leer_rasgos(corpus: Path, seleccion: list[tuple[dict[str, Any], int]]) -> list[dict[str, Any]]:
    filas = []
    vistos = set()
    inicio = time.monotonic()
    with ThreadPoolExecutor(max_workers=8) as lector:
        for indice, (item, cantidad) in enumerate(seleccion, 1):
            ruta, manifiesto = item["ruta"], item["dato"]
            detalles = manifiesto["jugadas"][:cantidad]
            rutas_archivos = []
            for detalle in detalles:
                nombre = detalle["archivo"]
                if Path(nombre).name != nombre or not nombre.endswith(".json"):
                    raise ValueError(f"Archivo inválido en {ruta}: {nombre}")
                rutas_archivos.append(ruta.parent / nombre)
            brutos = lector.map(lambda archivo: json.loads(archivo.read_text(encoding="utf-8")),
                                rutas_archivos)
            for detalle, bruto in zip(detalles, brutos):
                play_id = detalle["play_id"]
                if play_id in vistos:
                    raise ValueError(f"Jugada duplicada: {play_id}")
                vistos.add(play_id)
                categoria = detalle["categoria_clasificacion"]
                if categoria not in CATEGORIAS:
                    raise ValueError(f"Categoría desconocida: {categoria}")
                nombre = detalle["archivo"]
                if (bruto["columns"] != manifiesto["columnas"]
                        or len(bruto["data"]) != detalle["total_eventos"]):
                    raise ValueError(f"La jugada no concilia: {ruta.parent / nombre}")
                eventos = [dict(zip(bruto["columns"], valores)) for valores in bruto["data"]]
                if not eventos or eventos[0].get("id") != detalle["primer_evento_id"]:
                    raise ValueError(f"El primer evento no concilia: {ruta.parent / nombre}")
                rasgos, _ = representar_jugada(eventos, detalle["equipo_protagonista"])
                if not rasgos:
                    raise ValueError(f"La jugada no tiene n-gramas: {play_id}")
                primer_patron = eventos[0].get("play_pattern")
                filas.append({
                    "play_id": play_id, "categoria": categoria,
                    "partido_id": manifiesto["partido_id"],
                    "temporada": item["temporada"],
                    "manifiesto": str(ruta.relative_to(corpus)), "archivo": nombre,
                    "total_eventos": detalle["total_eventos"],
                    "pases_completados": detalle["pases_completados"],
                    "reanudacion": primer_patron.get("name") if isinstance(primer_patron, dict) else None,
                    "intervenciones_rivales": len(detalle["intervenciones_rivales_absorbidas"]),
                    "ngramas": rasgos,
                })
            if indice % 25 == 0:
                print(f"Leídos {indice}/{len(seleccion)} partidos; {len(filas)} jugadas; "
                      f"{time.monotonic() - inicio:.1f} s", flush=True)
    return filas


def _matrices(train: list[dict[str, Any]], test: list[dict[str, Any]]):
    vectorizador = DictVectorizer(sparse=True)
    conteos_train = vectorizador.fit_transform([fila["ngramas"] for fila in train])
    tfidf = TfidfTransformer(norm="l2", use_idf=True)
    x_train = tfidf.fit_transform(conteos_train)
    x_test = tfidf.transform(vectorizador.transform([fila["ngramas"] for fila in test]))
    return vectorizador, tfidf, x_train, x_test


def _silueta(matriz: np.ndarray, etiquetas: np.ndarray, semilla: int) -> float | None:
    if len(np.unique(etiquetas)) < 2:
        return None
    # sklearn solo materializa distancias entre estas <= 2000 filas.
    rng = np.random.default_rng(semilla)
    indices = (rng.choice(len(etiquetas), min(2000, len(etiquetas)), replace=False)
               if len(etiquetas) > 2000 else np.arange(len(etiquetas)))
    if len(np.unique(etiquetas[indices])) < 2:
        return None
    return float(silhouette_score(matriz[indices], etiquetas[indices], metric="cosine"))


def _perfil(
    x_train: Any, train: list[dict[str, Any]], x_test: Any,
    test: list[dict[str, Any]], labels_train: np.ndarray,
    labels_test: np.ndarray, distancias_train: np.ndarray,
    vocabulario: np.ndarray, nombres: list[str],
) -> list[dict[str, Any]]:
    media_global = np.asarray(x_train.mean(axis=0)).ravel()
    fichas = []
    for indice, nombre in enumerate(nombres):
        indices_train = np.flatnonzero(labels_train == indice)
        indices_test = np.flatnonzero(labels_test == indice)
        if not len(indices_train):
            fichas.append({"cluster": nombre, "ngramas": [], "ejemplos": [],
                           "jugadas_entrenamiento": 0, "jugadas_evaluacion": len(indices_test)})
            continue
        media = np.asarray(x_train[indices_train].mean(axis=0)).ravel()
        puntaje = (media - media_global) * np.sqrt(np.maximum(media, 0))
        destacados = np.argsort(-puntaje, kind="stable")[:12]
        orden = indices_train[np.argsort(distancias_train[indices_train], kind="stable")]
        ejemplos = []
        partidos = set()
        for j in orden:
            fila = train[j]
            if fila["partido_id"] in partidos:
                continue
            partidos.add(fila["partido_id"])
            ejemplos.append({k: fila[k] for k in (
                "play_id", "partido_id", "manifiesto", "archivo", "total_eventos",
                "pases_completados", "reanudacion", "intervenciones_rivales")})
            ejemplos[-1]["distancia_centroide"] = float(distancias_train[j])
            if len(ejemplos) == 3:
                break
        largos = [test[j]["total_eventos"] for j in indices_test]
        reanudaciones = Counter(test[j]["reanudacion"] or "sin_dato" for j in indices_test)
        fichas.append({
            "cluster": nombre,
            "jugadas_entrenamiento": len(indices_train),
            "jugadas_evaluacion": len(indices_test),
            "ngramas": [{"ngramas": str(vocabulario[i]), "contraste_tfidf": float(puntaje[i]),
                         "media_cluster": float(media[i])} for i in destacados if puntaje[i] > 0],
            "ejemplos": ejemplos,
            "longitud_media_eventos_evaluacion": float(np.mean(largos)) if largos else None,
            "reanudaciones_evaluacion": dict(reanudaciones),
            "fraccion_intervencion_rival_evaluacion": (
                float(np.mean([test[j]["intervenciones_rivales"] > 0 for j in indices_test]))
                if len(indices_test) else None),
        })
    return fichas


def _resumen(
    espacio_train: np.ndarray, espacio_test: np.ndarray,
    train: list[dict[str, Any]], test: list[dict[str, Any]],
    labels_train: np.ndarray, labels_test: np.ndarray,
    distancias_train: np.ndarray, distancias_test: np.ndarray,
    estabilidad: float, x_train: Any, x_test: Any,
    vocabulario: np.ndarray, nombres: list[str], semilla: int,
) -> dict[str, Any]:
    silueta = _silueta(espacio_test, labels_test, semilla)
    tamanos = np.bincount(labels_test, minlength=len(nombres)) / len(test)
    umbrales = {nombre: float(np.quantile(distancias_train[labels_train == i], .95))
                if np.any(labels_train == i) else None for i, nombre in enumerate(nombres)}
    claros = np.array([distancias_test[j] <= umbrales[nombres[cluster]]
                       if umbrales[nombres[cluster]] is not None else False
                       for j, cluster in enumerate(labels_test)])
    valores = [fila["total_eventos"] for fila in test]
    correlacion = (float(spearmanr(valores, distancias_test).statistic)
                   if len(set(valores)) > 1 and len(set(distancias_test)) > 1 else None)
    if correlacion is not None and not math.isfinite(correlacion):
        correlacion = None
    pasa = (silueta is not None and silueta >= UMBRAL_SILUETA
            and estabilidad >= UMBRAL_ARI
            and bool(np.all(tamanos >= MIN_FRACCION))
            and bool(np.all(tamanos <= MAX_FRACCION)))
    return {
        "silueta_coseno_evaluacion": silueta,
        "estabilidad_ari_evaluacion": estabilidad,
        "fracciones_clusters_evaluacion": dict(zip(nombres, map(float, tamanos))),
        "fraccion_sin_patron_claro_evaluacion": float(np.mean(~claros)),
        "umbrales_distancia_centroide_q95": umbrales,
        "correlacion_spearman_longitud_distancia": correlacion,
        "filtros_cuantitativos": bool(pasa),
        "fichas": _perfil(x_train, train, x_test, test, labels_train, labels_test,
                           distancias_train, vocabulario, nombres),
    }


def _distancias_centroides(espacio: np.ndarray, modelo: MiniBatchKMeans):
    centros = modelo.cluster_centers_
    # Solo N x K, nunca N x N.
    d2 = (np.sum(espacio * espacio, axis=1)[:, None]
          + np.sum(centros * centros, axis=1)[None, :]
          - 2 * espacio @ centros.T)
    distancias = np.sqrt(np.maximum(d2, 0))
    orden = np.argsort(distancias, axis=1, kind="stable")[:, :2]
    filas = np.arange(len(espacio))
    return orden[:, 0], distancias[filas, orden[:, 0]], (
        distancias[filas, orden[:, 1]] - distancias[filas, orden[:, 0]])


def _ajustar_svd(
    x_train: Any, x_test: Any, train: list[dict[str, Any]],
    test: list[dict[str, Any]], vocabulario: np.ndarray,
    categoria: str, dimensiones: int, k: int, svd: TruncatedSVD,
    z_train: np.ndarray, z_test: np.ndarray,
) -> tuple[dict[str, Any], dict[str, Any]]:
    inicio = time.monotonic()
    modelos = [MiniBatchKMeans(n_clusters=k, random_state=semilla,
                               batch_size=2048, n_init=3, max_iter=100)
               for semilla in SEMILLAS]
    asignaciones = []
    for modelo in modelos:
        modelo.fit(z_train)
        asignaciones.append(modelo.predict(z_test))
    estabilidad = float(np.mean([adjusted_rand_score(asignaciones[0], otra)
                                 for otra in asignaciones[1:]]))
    labels_train, d_train, _ = _distancias_centroides(z_train, modelos[0])
    labels_test, d_test, margen_test = _distancias_centroides(z_test, modelos[0])
    nombres = [f"{categoria}_k{k:02d}_c{i + 1:02d}" for i in range(k)]
    resumen = _resumen(z_train, z_test, train, test, labels_train, labels_test,
                       d_train, d_test, estabilidad, x_train, x_test,
                       vocabulario, nombres, SEMILLAS[0])
    resumen.update({"dimensiones": dimensiones, "k": k,
                    "varianza_explicada_svd": float(svd.explained_variance_ratio_.sum()),
                    "segundos": time.monotonic() - inicio,
                    "margen_mediano_evaluacion": float(np.median(margen_test))})
    return resumen, {"svd": svd, "kmeans": modelos[0]}


def _ajustar_nmf(
    x_train: Any, x_test: Any, train: list[dict[str, Any]],
    test: list[dict[str, Any]], vocabulario: np.ndarray, categoria: str,
) -> dict[str, Any]:
    inicio = time.monotonic()
    modelos = [NMF(n_components=4, init="nndsvda", random_state=semilla,
                   max_iter=500) for semilla in SEMILLAS]
    proyecciones = []
    for modelo in modelos:
        modelo.fit(x_train)
        normas = np.linalg.norm(modelo.components_, axis=1)
        proyecciones.append(normalize(modelo.transform(x_test) * normas, norm="l2"))
    normas_base = np.linalg.norm(modelos[0].components_, axis=1)
    z_train = normalize(modelos[0].transform(x_train) * normas_base, norm="l2")
    z_test = proyecciones[0]
    labels_train = np.argmax(z_train, axis=1)
    labels_test = np.argmax(z_test, axis=1)
    estabilidad = float(np.mean([adjusted_rand_score(labels_test, np.argmax(z, axis=1))
                                 for z in proyecciones[1:]]))
    nombres = [f"{categoria}_nmf_c{i + 1:02d}" for i in range(4)]
    # Distancias en el espacio W normalizado, con las mismas métricas que SVD.
    centroides = np.vstack([
        z_train[labels_train == i].mean(axis=0) if np.any(labels_train == i)
        else np.zeros(z_train.shape[1]) for i in range(4)
    ])
    d_train = np.linalg.norm(z_train - centroides[labels_train], axis=1)
    d_test = np.linalg.norm(z_test - centroides[labels_test], axis=1)
    resumen = _resumen(z_train, z_test, train, test, labels_train, labels_test,
                       d_train, d_test, estabilidad, x_train, x_test,
                       vocabulario, nombres, SEMILLAS[0])
    for i, ficha in enumerate(resumen["fichas"]):
        h = modelos[0].components_[i]
        ficha["ngramas"] = [{"ngramas": str(vocabulario[j]), "peso_h": float(h[j])}
                             for j in np.argsort(-h, kind="stable")[:12]]
    resumen.update({"componentes": 4, "segundos": time.monotonic() - inicio,
                    "error_reconstruccion_entrenamiento": float(modelos[0].reconstruction_err_)})
    return resumen


def ejecutar_piloto(corpus: str | Path, salida: str | Path,
                    *, entrenamiento: int = 30_000, evaluacion: int = 6_000) -> dict[str, Any]:
    corpus, salida = Path(corpus), Path(salida)
    if salida.exists():
        raise FileExistsError(salida)
    manifiestos = _manifiestos(corpus)
    train_sel, test_sel = seleccionar_partidos(
        manifiestos, entrenamiento=entrenamiento, evaluacion=evaluacion)
    train = _leer_rasgos(corpus, train_sel)
    test = _leer_rasgos(corpus, test_sel)
    if {fila["partido_id"] for fila in train} & {fila["partido_id"] for fila in test}:
        raise ValueError("La evaluación comparte partidos con entrenamiento")
    resultado: dict[str, Any] = {
        "estado": "exploratorio", "decision": "pendiente_revision_tactica",
        "corpus": str(corpus), "total_corpus": sum(i["cantidad"] for i in manifiestos),
        "muestra_entrenamiento": len(train), "muestra_evaluacion": len(test),
        "partidos_entrenamiento": sorted({f["partido_id"] for f in train}),
        "partidos_evaluacion": sorted({f["partido_id"] for f in test}),
        "distribucion_temporada_entrenamiento": dict(Counter(f["temporada"] for f in train)),
        "distribucion_temporada_evaluacion": dict(Counter(f["temporada"] for f in test)),
        "representacion": "base_n2_n3_acciones_desplazamientos_lances_v2",
        "vectorizacion": "DictVectorizer + TfidfTransformer(norm=l2,use_idf=True)",
        "umbrales": {"silueta_coseno": UMBRAL_SILUETA,
                     "estabilidad_ari": UMBRAL_ARI,
                     "fraccion_minima": MIN_FRACCION,
                     "fraccion_maxima": MAX_FRACCION,
                     "distancia_sin_patron_cuantil": .95},
        "categorias": {},
    }
    artefactos = {}
    for categoria in CATEGORIAS:
        t = [fila for fila in train if fila["categoria"] == categoria]
        e = [fila for fila in test if fila["categoria"] == categoria]
        if len(t) < 64 or len(e) < 32:
            raise ValueError(f"Muestra insuficiente de {categoria}")
        vectorizador, tfidf, x_train, x_test = _matrices(t, e)
        nombres = vectorizador.get_feature_names_out()
        corridas = {}
        tiempos_svd = {}
        for dimensiones in DIMENSIONES:
            if dimensiones >= min(x_train.shape):
                raise ValueError(f"No caben {dimensiones} componentes en {categoria}")
            inicio_svd = time.monotonic()
            svd = TruncatedSVD(n_components=dimensiones, random_state=SEMILLAS[0])
            z_train = normalize(svd.fit_transform(x_train), norm="l2")
            z_test = normalize(svd.transform(x_test), norm="l2")
            tiempos_svd[str(dimensiones)] = time.monotonic() - inicio_svd
            for k in NUM_CLUSTERS:
                clave = f"svd{dimensiones}_k{k}"
                resumen, modelo = _ajustar_svd(x_train, x_test, t, e, nombres,
                                                categoria, dimensiones, k,
                                                svd, z_train, z_test)
                corridas[clave] = resumen
                artefactos[f"{categoria}_{clave}"] = {
                    "vectorizador": vectorizador, "tfidf": tfidf, **modelo}
                print(f"{categoria} {clave}: silueta={resumen['silueta_coseno_evaluacion']} "
                      f"ARI={resumen['estabilidad_ari_evaluacion']:.3f} "
                      f"pasa={resumen['filtros_cuantitativos']}", flush=True)
        mejores = [clave for clave, r in corridas.items() if r["filtros_cuantitativos"]]
        candidato = (max(mejores, key=lambda clave: (
            corridas[clave]["silueta_coseno_evaluacion"],
            corridas[clave]["estabilidad_ari_evaluacion"],
            -corridas[clave]["k"], -corridas[clave]["dimensiones"])) if mejores else None)
        resultado["categorias"][categoria] = {
            "entrenamiento": len(t), "evaluacion": len(e),
            "matriz_entrenamiento": {"filas": x_train.shape[0], "columnas": x_train.shape[1],
                                      "elementos_no_nulos": x_train.nnz},
            "segundos_svd_por_dimension": tiempos_svd,
            "candidato_cuantitativo": candidato,
            "corridas": corridas,
            "nmf_reentrenado_misma_particion": _ajustar_nmf(x_train, x_test, t, e, nombres,
                                                              categoria),
        }
    salida.mkdir(parents=True)
    with (salida / "evaluacion.json").open("w", encoding="utf-8") as archivo:
        json.dump(resultado, archivo, ensure_ascii=False, indent=2, sort_keys=True,
                  allow_nan=False)
    for categoria in CATEGORIAS:
        clave = resultado["categorias"][categoria]["candidato_cuantitativo"]
        if clave:
            joblib.dump(artefactos[f"{categoria}_{clave}"], salida / f"modelo_{categoria}.joblib")
    return resultado


def _revision_aprobada(revision: dict[str, Any]) -> bool:
    return (revision.get("aprobado") is True
            and isinstance(revision.get("justificacion"), str)
            and bool(revision["justificacion"].strip())
            and all(revision.get("categorias", {}).get(categoria, {}).get("aprobado") is True
                    and bool(revision["categorias"][categoria].get("justificacion", "").strip())
                    for categoria in CATEGORIAS))


def entrenar_corpus_completo(
    corpus: str | Path, ruta_piloto: str | Path, ruta_revision: str | Path,
    salida: str | Path, *, exploratorio_rechazado: bool = False,
) -> dict[str, Any]:
    """Reajusta el corpus y conserva el estado no validado cuando el piloto falló."""
    corpus, salida = Path(corpus), Path(salida)
    if salida.exists():
        raise FileExistsError(salida)
    piloto = json.loads(Path(ruta_piloto).read_text(encoding="utf-8"))
    revision = json.loads(Path(ruta_revision).read_text(encoding="utf-8"))
    if exploratorio_rechazado:
        if revision.get("aprobado") is not False:
            raise ValueError("La ruta exploratoria requiere una revisión rechazada del piloto")
    elif not _revision_aprobada(revision):
        raise ValueError("Falta aprobación táctica motivada del piloto para ambas categorías")
    if Path(piloto["corpus"]).resolve() != corpus.resolve():
        raise ValueError("El piloto pertenece a otro corpus")
    candidatos = {}
    claves_candidatos = {}
    for categoria in CATEGORIAS:
        corridas = piloto["categorias"][categoria]["corridas"]
        if exploratorio_rechazado:
            elegibles = [clave for clave, corrida in corridas.items()
                         if corrida["silueta_coseno_evaluacion"] is not None
                         and corrida["silueta_coseno_evaluacion"] >= UMBRAL_SILUETA
                         and min(corrida["fracciones_clusters_evaluacion"].values()) >= MIN_FRACCION
                         and max(corrida["fracciones_clusters_evaluacion"].values()) <= MAX_FRACCION]
            if not elegibles:
                raise ValueError(f"Ninguna configuración exploratoria tiene separación y tamaños suficientes en {categoria}")
            clave = max(elegibles, key=lambda nombre: (
                corridas[nombre]["estabilidad_ari_evaluacion"],
                corridas[nombre]["silueta_coseno_evaluacion"],
                -corridas[nombre]["k"], -corridas[nombre]["dimensiones"]))
        else:
            clave = piloto["categorias"][categoria]["candidato_cuantitativo"]
            if not clave or not corridas[clave]["filtros_cuantitativos"]:
                raise ValueError(f"El piloto no pasó los filtros de {categoria}")
        claves_candidatos[categoria] = clave
        candidatos[categoria] = corridas[clave]
    manifiestos = _manifiestos(corpus)
    filas = _leer_rasgos(corpus, [(item, item["cantidad"]) for item in manifiestos])
    if len(filas) != piloto["total_corpus"]:
        raise ValueError("Cambió el tamaño del corpus desde el piloto")
    salida.mkdir(parents=True)
    diagnostico: dict[str, Any] = {
        "estado": ("reentrenado_exploratorio_rechazado" if exploratorio_rechazado
                   else "reentrenado_pendiente_revision"),
        "decision": "rechazado" if exploratorio_rechazado else "pendiente_revision",
        "total_jugadas": len(filas),
        "corpus": str(corpus), "piloto": str(ruta_piloto),
        "revision_piloto": revision, "categorias": {},
        "configuraciones_elegidas": claves_candidatos,
        "criterio_seleccion": ("mayor_ari_con_silueta_y_tamanos_minimos; "
                               "etiquetas_no_validadas" if exploratorio_rechazado
                               else "candidato_cuantitativo_aprobado"),
    }
    with (salida / "clasificacion_exploratoria.jsonl").open("w", encoding="utf-8") as archivo:
        for categoria in CATEGORIAS:
            grupo = [fila for fila in filas if fila["categoria"] == categoria]
            vectorizador, tfidf, x, _ = _matrices(grupo, grupo[:1])
            dimensiones = candidatos[categoria]["dimensiones"]
            k = candidatos[categoria]["k"]
            svd = TruncatedSVD(n_components=dimensiones, random_state=SEMILLAS[0])
            z = normalize(svd.fit_transform(x), norm="l2")
            modelo = MiniBatchKMeans(n_clusters=k, random_state=SEMILLAS[0],
                                     batch_size=2048, n_init=3, max_iter=100)
            modelo.fit(z)
            etiquetas, distancias, margen = _distancias_centroides(z, modelo)
            nombres = [f"{categoria}_k{k:02d}_c{i + 1:02d}" for i in range(k)]
            umbrales = {nombre: float(np.quantile(distancias[etiquetas == i], .95))
                        if np.any(etiquetas == i) else None
                        for i, nombre in enumerate(nombres)}
            fracciones = np.bincount(etiquetas, minlength=k) / len(grupo)
            rng = np.random.default_rng(SEMILLAS[0])
            muestra = rng.choice(len(grupo), min(2000, len(grupo)), replace=False)
            silueta = _silueta(z[muestra], etiquetas[muestra], SEMILLAS[0])
            fichas = _perfil(x, grupo, x[muestra], [grupo[i] for i in muestra],
                            etiquetas, etiquetas[muestra], distancias,
                            vectorizador.get_feature_names_out(), nombres)
            chequeo = (silueta is not None and silueta >= UMBRAL_SILUETA
                       and bool(np.all(fracciones >= MIN_FRACCION))
                       and bool(np.all(fracciones <= MAX_FRACCION))
                       and all(len(f["ngramas"]) and len(f["ejemplos"]) == 3 for f in fichas))
            diagnostico["categorias"][categoria] = {
                "dimensiones": dimensiones, "k": k,
                "silueta_coseno_muestra_2000": silueta,
                "fracciones_clusters": dict(zip(nombres, map(float, fracciones))),
                "umbrales_distancia_centroide_q95": umbrales,
                "filtros_cuantitativos_y_ejemplos": bool(chequeo),
                "fichas": fichas,
            }
            joblib.dump({"vectorizador": vectorizador, "tfidf": tfidf,
                         "svd": svd, "kmeans": modelo, "cluster_ids": nombres,
                         "umbrales": umbrales}, salida / f"modelo_{categoria}.joblib")
            for j, fila in enumerate(grupo):
                nombre = nombres[etiquetas[j]]
                umbral = umbrales[nombre]
                etiqueta = nombre if umbral is not None and distancias[j] <= umbral else "Sin patrón claro"
                archivo.write(json.dumps({
                    "play_id": fila["play_id"], "partido_id": fila["partido_id"],
                    "categoria": categoria,
                    "etiqueta_exploratoria": etiqueta,
                    "cluster_cercano": nombre,
                    "distancia_centroide": float(distancias[j]),
                    "margen_segundo_centroide": float(margen[j]),
                    "umbral_distancia_cluster": umbral,
                    "manifiesto": fila["manifiesto"], "archivo": fila["archivo"],
                    "total_eventos": fila["total_eventos"],
                    "pases_completados": fila["pases_completados"],
                }, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")
            print(f"Corpus completo {categoria}: {len(grupo)} jugadas, "
                  f"silueta={silueta}, pasa={chequeo}", flush=True)
    with (salida / "evaluacion_completa.json").open("w", encoding="utf-8") as archivo:
        json.dump(diagnostico, archivo, ensure_ascii=False, indent=2,
                  sort_keys=True, allow_nan=False)
    return diagnostico


def publicar_corpus_completo(
    ruta_entrenamiento: str | Path, ruta_revision: str | Path, salida: str | Path,
) -> dict[str, Any]:
    """Publica las etiquetas solo tras comprobar el reajuste y revisar sus fichas."""
    origen, salida = Path(ruta_entrenamiento), Path(salida)
    if salida.exists():
        raise FileExistsError(salida)
    diagnostico = json.loads((origen / "evaluacion_completa.json").read_text(encoding="utf-8"))
    if diagnostico.get("estado") != "reentrenado_pendiente_revision":
        raise ValueError("No se pueden publicar como validadas etiquetas de un piloto rechazado")
    revision = json.loads(Path(ruta_revision).read_text(encoding="utf-8"))
    if not _revision_aprobada(revision):
        raise ValueError("Falta aprobación táctica motivada del modelo completo")
    if not all(diagnostico["categorias"][categoria]["filtros_cuantitativos_y_ejemplos"]
               for categoria in CATEGORIAS):
        raise ValueError("El modelo completo perdió los filtros de aceptación")
    salida.mkdir(parents=True)
    for categoria in CATEGORIAS:
        shutil.copy2(origen / f"modelo_{categoria}.joblib", salida)
    cantidad = 0
    vistos = set()
    with (origen / "clasificacion_exploratoria.jsonl").open(encoding="utf-8") as entrada, (
            salida / "clasificacion.jsonl").open("w", encoding="utf-8") as archivo:
        for linea in entrada:
            fila = json.loads(linea)
            if fila["play_id"] in vistos:
                raise ValueError(f"play_id duplicado en tabla completa: {fila['play_id']}")
            vistos.add(fila["play_id"])
            fila["clase_principal"] = fila.pop("etiqueta_exploratoria")
            archivo.write(json.dumps(fila, ensure_ascii=False, sort_keys=True,
                                     allow_nan=False) + "\n")
            cantidad += 1
    if cantidad != diagnostico["total_jugadas"]:
        raise ValueError("La tabla completa no concilia")
    informe = {"estado": "clasificacion_publicada", "decision": "aceptado",
               "total_jugadas": cantidad, "revision_completa": revision,
               "diagnostico": diagnostico,
               "tabla_clasificacion": "clasificacion.jsonl"}
    with (salida / "informe.json").open("w", encoding="utf-8") as archivo:
        json.dump(informe, archivo, ensure_ascii=False, indent=2,
                  sort_keys=True, allow_nan=False)
    return {"estado": informe["estado"], "total_jugadas": cantidad}


def main() -> None:
    parser = argparse.ArgumentParser(description="Piloto SVD + MiniBatchKMeans vs NMF")
    parser.add_argument("corpus", type=Path)
    parser.add_argument("salida", type=Path)
    parser.add_argument("--entrenamiento", type=int, default=30_000)
    parser.add_argument("--evaluacion", type=int, default=6_000)
    parser.add_argument("--piloto", type=Path,
                        help="Evaluación del piloto para reajustar el corpus completo")
    parser.add_argument("--revision-piloto", type=Path)
    parser.add_argument("--exploratorio-rechazado", action="store_true",
                        help="Genera etiquetas exploratorias aun si el piloto fue rechazado")
    parser.add_argument("--publicar", type=Path,
                        help="Carpeta de reajuste completo ya evaluada")
    parser.add_argument("--revision-completa", type=Path)
    args = parser.parse_args()
    if args.publicar:
        if not args.revision_completa:
            parser.error("--publicar requiere --revision-completa")
        resultado = publicar_corpus_completo(args.publicar, args.revision_completa,
                                              args.salida)
    elif args.piloto:
        if not args.revision_piloto:
            parser.error("--piloto requiere --revision-piloto")
        resultado = entrenar_corpus_completo(args.corpus, args.piloto,
                                              args.revision_piloto, args.salida,
                                              exploratorio_rechazado=args.exploratorio_rechazado)
    else:
        resultado = ejecutar_piloto(args.corpus, args.salida,
                                    entrenamiento=args.entrenamiento,
                                    evaluacion=args.evaluacion)
    print(json.dumps({"estado": resultado["estado"],
                      "total_jugadas": resultado.get("total_jugadas"),
                      "candidatos": {k: v["candidato_cuantitativo"]
                                     for k, v in resultado.get("categorias", {}).items()
                                     if "candidato_cuantitativo" in v}}, ensure_ascii=False))


if __name__ == "__main__":
    main()
