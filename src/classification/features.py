"""Rasgos secuenciales y espaciales de jugadas preparadas."""

import math
from collections import Counter
from dataclasses import dataclass
from statistics import median
from typing import Any, Iterable, Literal


LARGO_CAMPO = 120.0
ANCHO_CAMPO = 80.0
UMBRAL_AVANCE_X = 5.0
DISTANCIA_CORTA = 10.0
DISTANCIA_MEDIA = 30.0
AUXILIARES = {
    "Camera On", "Camera Off", "Tactical Shift",
    "Starting XI", "Half Start", "Half End",
}
CONTEXTUALES = {"Ball Receipt*", "Dribbled Past", "Pressure", "Foul Won"}
CONTEXTUALES_ENTRE_EQUIPOS = {"Ball Receipt*", "Dribbled Past", "Foul Won"}
CONTACTO = {"Duel", "Ball Recovery", "Interception", "Dispossessed"}
PRIORIDAD_CONTACTO = {"Ball Recovery": 0, "Interception": 1, "Duel": 2, "Dispossessed": 3}
ModoRivales = Literal["con_rivales", "sin_rivales"]
FilaAccion = tuple[dict[str, Any], str, str]


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


def _relacionados(a: dict[str, Any], b: dict[str, Any]) -> bool:
    a_rel = set(a.get("related_events") or [])
    b_rel = set(b.get("related_events") or [])
    return (
        b.get("id") in a_rel or a.get("id") in b_rel or bool(a_rel & b_rel)
    )


def _contactos_redundantes(filas: list[FilaAccion]) -> tuple[set[int], dict[int, int]]:
    """Elige un registro para cada lance simultáneo de contacto."""
    omitidos: set[int] = set()
    sustitutos: dict[int, int] = {}
    pendientes = {i for i, (_, tipo, _) in enumerate(filas) if tipo in CONTACTO}
    while pendientes:
        inicio = min(pendientes)
        grupo = {inicio}
        frontera = [inicio]
        while frontera:
            actual = frontera.pop()
            marca = filas[actual][0].get("timestamp")
            if marca is None:
                continue
            vinculados = {
                j for j in pendientes - grupo
                if filas[j][0].get("timestamp") == marca
                and _relacionados(filas[actual][0], filas[j][0])
            }
            grupo.update(vinculados)
            frontera.extend(vinculados)
        pendientes.difference_update(grupo)
        if len(grupo) > 1:
            elegido = min(grupo, key=lambda k: (
                PRIORIDAD_CONTACTO[filas[k][1]],
                _nombre(filas[k][0].get("team")) != _nombre(filas[k][0].get("possession_team")),
                k,
            ))
            sustitutos[inicio] = elegido
            omitidos.update(grupo - {inicio})
    return omitidos, sustitutos


def _contextuales_redundantes(filas: list[FilaAccion]) -> set[int]:
    """Descarta contexto vinculado sin descartar acciones aisladas."""
    principales = [
        (evento, actor) for evento, tipo, actor in filas if tipo not in CONTEXTUALES
    ]
    contextuales_vistos: list[tuple[dict[str, Any], str]] = []
    omitidos: set[int] = set()
    for i, (evento, tipo, actor) in enumerate(filas):
        if tipo not in CONTEXTUALES:
            continue
        if any((actor == otro_actor or tipo in CONTEXTUALES_ENTRE_EQUIPOS)
               and _relacionados(evento, otra)
               for otra, otro_actor in principales):
            omitidos.add(i)
            continue
        if any(actor == otro_actor and evento.get("timestamp") is not None
               and evento.get("timestamp") == otra.get("timestamp")
               and _relacionados(evento, otra) for otra, otro_actor in contextuales_vistos):
            omitidos.add(i)
            continue
        contextuales_vistos.append((evento, actor))
    return omitidos


def _acciones(eventos: list[dict[str, Any]], protagonista: str) -> list[FilaAccion]:
    """Una acción por lance vinculado, conservando decisiones de balón distintas."""
    filas = [
        (evento, tipo, _actor(evento, protagonista))
        for evento in eventos
        if (tipo := _nombre(evento.get("type"))) and tipo not in AUXILIARES
    ]
    omitidos, sustitutos = _contactos_redundantes(filas)
    omitidos.update(_contextuales_redundantes(filas))
    return [filas[sustitutos.get(i, i)] for i in range(len(filas)) if i not in omitidos]


def _banda_original(acciones: list[FilaAccion]) -> str:
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
    evento: dict[str, Any], tipo: str, actor: str, papel: str, reflejar: bool,
    config: ConfiguracionEspacial,
) -> str:
    origen, destino = _extremos(evento, tipo)
    origen = _marco_protagonista(origen, actor, reflejar)
    destino = _marco_protagonista(destino, actor, reflejar)
    zona_origen, zona_destino = _zona(origen, config), _zona(destino, config)
    token = (
        f"{papel}:{tipo.replace(' ', '_')}:origen={zona_origen}"
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
    *, modo_rivales: ModoRivales = "con_rivales",
) -> tuple[Counter[str], dict[str, Any]]:
    """Conserva los n-gramas base y añade los espaciales en un marco común."""
    if modo_rivales not in ("con_rivales", "sin_rivales"):
        raise ValueError(f"Modo de rivales desconocido: {modo_rivales}")
    acciones = _acciones(eventos, protagonista)
    banda = _banda_original(acciones)
    reflejar = banda == "y_alta"
    seleccionadas = []
    rival_con_control: str | None = None
    for evento, tipo, actor in acciones:
        posesion = _nombre(evento.get("possession_team"))
        if posesion == protagonista:
            rival_con_control = None
        if actor == "rival":
            equipo = _nombre(evento.get("team"))
            if posesion == equipo and rival_con_control != equipo:
                papel = "rival_recuperacion"
            elif posesion == equipo:
                papel = "rival_control"
            else:
                papel = "rival_interrupcion"
            if posesion == equipo:
                rival_con_control = equipo
        else:
            papel = actor
        if actor != "rival" or modo_rivales == "con_rivales":
            seleccionadas.append((evento, tipo, actor, papel))
    basicos = [
        _token_basico(evento, tipo, papel)
        for evento, tipo, _, papel in seleccionadas
    ]
    rasgos = _ngramas(basicos)
    if config is not None:
        espaciales = [
            _token_espacial(evento, tipo, actor, papel, reflejar, config)
            for evento, tipo, actor, papel in seleccionadas
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
