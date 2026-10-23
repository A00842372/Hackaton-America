"""Rasgos secuenciales y espaciales de jugadas preparadas."""

import math
from collections import Counter
from dataclasses import dataclass
from statistics import median
from typing import Any, Iterable


LARGO_CAMPO = 120.0
ANCHO_CAMPO = 80.0
UMBRAL_AVANCE_X = 5.0
DISTANCIA_CORTA = 10.0
DISTANCIA_MEDIA = 30.0
AUXILIARES = {
    "Ball Receipt*", "Camera On", "Camera Off", "Tactical Shift",
    "Starting XI", "Half Start", "Half End",
}
AUXILIARES_VINCULADOS = {"Dribbled Past", "Foul Won"}


@dataclass(frozen=True)
class ConfiguracionEspacial:
    zonas_x: int = 6
    zonas_y: int = 4
    peso: float = 1.0

    def __post_init__(self) -> None:
        if (
            not isinstance(self.zonas_x, int) or isinstance(self.zonas_x, bool)
            or not isinstance(self.zonas_y, int) or isinstance(self.zonas_y, bool)
            or self.zonas_x < 1 or self.zonas_y < 1
            or not math.isfinite(self.peso) or self.peso <= 0
        ):
            raise ValueError("La cuadrícula y el peso espacial deben ser positivos")


def _nombre(valor: Any) -> str | None:
    if isinstance(valor, dict):
        return valor.get("name")
    return valor if isinstance(valor, str) else None


def _punto(valor: Any) -> tuple[float, float] | None:
    if not isinstance(valor, list) or len(valor) < 2:
        return None
    x, y = valor[:2]
    if not all(isinstance(n, (int, float)) and not isinstance(n, bool) and math.isfinite(n)
               for n in (x, y)):
        return None
    return float(x), float(y)


def _es_auxiliar(evento: dict[str, Any], tipo: str) -> bool:
    return tipo in AUXILIARES or (
        tipo in AUXILIARES_VINCULADOS and bool(evento.get("related_events"))
    )


def _actor(evento: dict[str, Any], protagonista: str) -> str:
    equipo = _nombre(evento.get("team"))
    return "propio" if equipo == protagonista else "rival" if equipo else "desconocido"


def _extremos(evento: dict[str, Any], tipo: str) -> tuple[tuple[float, float] | None, tuple[float, float] | None]:
    origen = _punto(evento.get("location"))
    detalle = evento.get(tipo.lower().replace(" ", "_"))
    destino = _punto(detalle.get("end_location")) if isinstance(detalle, dict) else None
    return origen, destino


def _direccion_x(delta: float) -> str:
    return "avance" if delta > UMBRAL_AVANCE_X else "retroceso" if delta < -UMBRAL_AVANCE_X else "lateral"


def _distancia(origen: tuple[float, float], destino: tuple[float, float]) -> str:
    distancia = math.dist(origen, destino)
    return "corto" if distancia < DISTANCIA_CORTA else "medio" if distancia < DISTANCIA_MEDIA else "largo"


def _token_basico(evento: dict[str, Any], tipo: str, actor: str) -> str:
    token = f"{actor}:{tipo.replace(' ', '_')}"
    origen, destino = _extremos(evento, tipo)
    if origen is not None and destino is not None:
        token += f":{_direccion_x(destino[0] - origen[0])}:{_distancia(origen, destino)}"
    return token


def _banda_original(acciones: list[tuple[dict[str, Any], str, str]]) -> str:
    tiene_punto = False
    for evento, tipo, actor in acciones:
        if actor != "propio":
            continue
        for punto in _extremos(evento, tipo):
            if punto is None:
                continue
            tiene_punto = True
            if punto[1] < ANCHO_CAMPO / 2:
                return "y_baja"
            if punto[1] > ANCHO_CAMPO / 2:
                return "y_alta"
    return "centro" if tiene_punto else "sin_dato"


def _marco_protagonista(
    punto: tuple[float, float] | None, actor: str, reflejar: bool,
) -> tuple[float, float] | None:
    if punto is None:
        return None
    x, y = punto
    if actor == "rival":
        x, y = LARGO_CAMPO - x, ANCHO_CAMPO - y
    if reflejar:
        y = ANCHO_CAMPO - y
    return x, y


def _zona(punto: tuple[float, float] | None, config: ConfiguracionEspacial) -> str:
    if punto is None:
        return "sin_dato"
    x, y = punto
    if not 0 <= x <= LARGO_CAMPO or not 0 <= y <= ANCHO_CAMPO:
        return "fuera_campo"
    zona_x = min(config.zonas_x - 1, int(x / LARGO_CAMPO * config.zonas_x))
    zona_y = min(config.zonas_y - 1, int(y / ANCHO_CAMPO * config.zonas_y))
    return f"x{zona_x}_y{zona_y}"


def _token_espacial(
    evento: dict[str, Any], tipo: str, actor: str, reflejar: bool,
    config: ConfiguracionEspacial,
) -> str:
    origen, destino = _extremos(evento, tipo)
    origen = _marco_protagonista(origen, actor, reflejar)
    destino = _marco_protagonista(destino, actor, reflejar)
    zona_origen, zona_destino = _zona(origen, config), _zona(destino, config)
    token = (
        f"{actor}:{tipo.replace(' ', '_')}:origen={zona_origen}"
        f":destino={zona_destino}"
    )
    if origen is None or destino is None or "fuera_campo" in (zona_origen, zona_destino):
        return token + ":movimiento=sin_dato"
    delta_y = destino[1] - origen[1]
    direccion_y = "y_alta" if delta_y > UMBRAL_AVANCE_X else "y_baja" if delta_y < -UMBRAL_AVANCE_X else "y_estable"
    return token + f":movimiento={_direccion_x(destino[0] - origen[0])}_{direccion_y}_{_distancia(origen, destino)}"


def _ngramas(tokens: list[str], prefijo: str = "") -> Counter[str]:
    return Counter(
        prefijo + f"n{largo}:" + " → ".join(tokens[inicio:inicio + largo])
        for largo in (2, 3)
        for inicio in range(len(tokens) - largo + 1)
    )


def representar_jugada(
    eventos: list[dict[str, Any]], protagonista: str,
    config: ConfiguracionEspacial | None = None,
) -> tuple[Counter[str], dict[str, Any]]:
    """Conserva los n-gramas base y añade los espaciales en un marco común."""
    acciones = []
    for evento in eventos:
        tipo = _nombre(evento.get("type"))
        if tipo and not _es_auxiliar(evento, tipo):
            acciones.append((evento, tipo, _actor(evento, protagonista)))
    banda = _banda_original(acciones)
    reflejar = banda == "y_alta"
    basicos = [_token_basico(evento, tipo, actor) for evento, tipo, actor in acciones]
    rasgos = _ngramas(basicos)
    if config is not None:
        espaciales = [
            _token_espacial(evento, tipo, actor, reflejar, config)
            for evento, tipo, actor in acciones
        ]
        for ngrama, cantidad in _ngramas(espaciales, "espacio:").items():
            rasgos[ngrama] = cantidad * config.peso
    return rasgos, {"banda_original": banda, "reflejada": reflejar}


def verificar_orientacion(jugadas: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Comprueba origen de tiros y avance de tiros y pases en ambos sentidos."""
    direcciones = ("left_to_right", "right_to_left")
    tiros_origen: dict[str, list[float]] = {direccion: [] for direccion in direcciones}
    avances: dict[str, dict[str, list[float]]] = {
        tipo: {direccion: [] for direccion in direcciones} for tipo in ("Shot", "Pass")
    }
    for jugada in jugadas:
        for evento in jugada["eventos"]:
            tipo = _nombre(evento.get("type"))
            if tipo not in avances:
                continue
            direccion = evento.get("attacking_direction")
            origen, destino = _extremos(evento, tipo)
            if direccion not in direcciones or origen is None:
                continue
            if tipo == "Shot":
                tiros_origen[direccion].append(origen[0])
            if destino is not None:
                avances[tipo][direccion].append(destino[0] - origen[0])
    resumen_tiros = {
        direccion: {"total": len(valores), "mediana_x": median(valores) if valores else None}
        for direccion, valores in tiros_origen.items()
    }
    resumen_avances = {
        tipo: {
            direccion: {"total": len(valores), "mediana_dx": median(valores) if valores else None}
            for direccion, valores in grupos.items()
        }
        for tipo, grupos in avances.items()
    }
    if any(
        resumen_tiros[d]["total"] < 2 or resumen_tiros[d]["mediana_x"] <= LARGO_CAMPO / 2
        or resumen_avances["Shot"][d]["total"] < 2 or resumen_avances["Shot"][d]["mediana_dx"] <= 0
        or resumen_avances["Pass"][d]["total"] < 2 or resumen_avances["Pass"][d]["mediana_dx"] <= 0
        for d in direcciones
    ):
        raise ValueError("No se confirmó que tiros y pases apunten a x=120 en ambos sentidos físicos")
    return {
        "convencion": "x_hacia_120_por_equipo",
        "tiros_por_direccion": resumen_tiros,
        "avance_por_direccion": resumen_avances,
    }
