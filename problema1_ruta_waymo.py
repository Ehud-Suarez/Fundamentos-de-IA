"""
ACTIVIDAD 2 (Secciones 2.1 y 2.2) - Fundamentos de Inteligencia Artificial
PROBLEMA 1: Planificación de ruta de un vehículo autónomo (inspirado en Waymo Driver)

Modelado como PROBLEMA DE BÚSQUEDA
----------------------------------
  Estado    : (x, y, rumbo)  -> intersección + dirección en la que mira el vehículo.
              rumbo en {N, E, S, W}.
  Estado inicial : (0, 0, 'E')
  Acciones  : 'recto', 'izquierda', 'derecha'   (no se permiten giros en U)
  Modelo de transición : mover el vehículo una cuadra en la nueva dirección.
  Coste     : tiempo de viaje en segundos = tiempo_base_de_la_cuadra
              + penalización_de_giro (izq. no protegida > der. > recto)
              + congestión de la calle.  Cuadras bloqueadas (obras) no existen.
  Meta      : llegar a la intersección destino (con cualquier rumbo).

Espacio de búsqueda -> GRAFO DIRIGIDO Y PONDERADO (networkx.DiGraph):
  nodos = estados, aristas = acciones, peso = coste en segundos.

NOTA: la red vial y los costes son un MODELO DIDÁCTICO (datos sintéticos),
no datos reales de Waymo. Sólo se toma la idea del problema (planificación).

Librerías: networkx, matplotlib, numpy (pip) + dataclasses, collections, time,
           random, pathlib, typing (biblioteca estándar).
"""

from __future__ import annotations

import random
import time
from collections import deque
from pathlib import Path
from typing import Dict, Iterator, List, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

# ----------------------------------------------------------------------------
# 1. PARÁMETROS DEL PROBLEMA (datos de entrada)
# ----------------------------------------------------------------------------
N = 4                                   # ciudad de N x N intersecciones
INICIO = (0, 0, "E")                    # estado inicial
META = (3, 3)                           # intersección destino
SEMILLA = 7                             # reproducibilidad
OBRAS = {((1, 1), (2, 1)), ((2, 2), (2, 3))}   # cuadras bloqueadas (en ambos sentidos)

VECTOR = {"N": (0, 1), "E": (1, 0), "S": (0, -1), "W": (-1, 0)}
ORDEN = ["N", "E", "S", "W"]            # sentido horario
# Penalización en segundos por tipo de maniobra (el giro a la izquierda sin
# semáforo protegido obliga a esperar hueco entre el tráfico contrario).
PENALIZACION = {"recto": 0.0, "derecha": 4.0, "izquierda": 12.0}


# ----------------------------------------------------------------------------
# 2. RED VIAL: tiempo base de cada cuadra (con congestión aleatoria fija)
# ----------------------------------------------------------------------------
def construir_tiempos_base(n: int, semilla: int) -> Dict[frozenset, float]:
    """Devuelve {frozenset({A,B}): segundos} para cada cuadra entre intersecciones."""
    rng = random.Random(semilla)
    tiempos = {}
    for x in range(n):
        for y in range(n):
            for dx, dy in ((1, 0), (0, 1)):
                a, b = (x, y), (x + dx, y + dy)
                if b[0] < n and b[1] < n:
                    tiempos[frozenset({a, b})] = 20 + rng.choice([0, 5, 10, 20])
    return tiempos


def cuadra_bloqueada(a: Tuple[int, int], b: Tuple[int, int]) -> bool:
    return (a, b) in OBRAS or (b, a) in OBRAS


# ----------------------------------------------------------------------------
# 3. GENERADOR DE ESTADOS (función sucesor)
# ----------------------------------------------------------------------------
def girar(rumbo: str, accion: str) -> str:
    """Nuevo rumbo tras aplicar la acción."""
    i = ORDEN.index(rumbo)
    if accion == "recto":
        return rumbo
    if accion == "derecha":
        return ORDEN[(i + 1) % 4]
    return ORDEN[(i - 1) % 4]           # izquierda


def sucesores(estado, tiempos, n=None) -> Iterator[Tuple[str, tuple, float]]:
    """
    GENERADOR DE ESTADOS: produce (acción, estado_siguiente, coste) para cada
    acción legal. Una acción es legal si la cuadra existe y no está bloqueada.
    Complejidad por llamada: O(b) con b = 3 (factor de ramificación).
    """
    n = N if n is None else n          # tamaño actual del mapa
    x, y, rumbo = estado
    for accion in ("recto", "izquierda", "derecha"):
        nuevo_rumbo = girar(rumbo, accion)
        dx, dy = VECTOR[nuevo_rumbo]
        nx_, ny_ = x + dx, y + dy
        if not (0 <= nx_ < n and 0 <= ny_ < n):          # fuera del mapa
            continue
        if cuadra_bloqueada((x, y), (nx_, ny_)):          # obra
            continue
        coste = tiempos[frozenset({(x, y), (nx_, ny_)})] + PENALIZACION[accion]
        yield accion, (nx_, ny_, nuevo_rumbo), coste


def es_meta(estado) -> bool:
    return (estado[0], estado[1]) == META


# ----------------------------------------------------------------------------
# 4. CONSTRUCCIÓN DEL GRAFO DEL ESPACIO DE ESTADOS (BFS con el generador)
# ----------------------------------------------------------------------------
def construir_grafo(inicio, tiempos) -> nx.DiGraph:
    """Explora todos los estados alcanzables y los guarda en un DiGraph. O(V + E)."""
    G = nx.DiGraph()
    G.add_node(inicio)
    frontera = deque([inicio])
    while frontera:
        s = frontera.popleft()
        for accion, s2, coste in sucesores(s, tiempos):
            if s2 not in G:
                G.add_node(s2)
                frontera.append(s2)
            G.add_edge(s, s2, weight=coste, accion=accion)
    return G


# ----------------------------------------------------------------------------
# 5. SOLUCIÓN (para comprobar que el modelo funciona)
# ----------------------------------------------------------------------------
def resolver(G: nx.DiGraph, inicio):
    """Dijkstra hacia el nodo meta más barato + BFS (menos maniobras)."""
    metas = [s for s in G.nodes if es_meta(s)]
    dist, caminos = nx.single_source_dijkstra(G, inicio, weight="weight")
    mejor = min((m for m in metas if m in dist), key=lambda m: dist[m])
    ruta_optima = caminos[mejor]

    bfs = nx.single_source_shortest_path(G, inicio)          # sin pesos
    meta_bfs = min((m for m in metas if m in bfs), key=lambda m: len(bfs[m]))
    ruta_bfs = bfs[meta_bfs]
    return ruta_optima, dist[mejor], ruta_bfs


def coste_ruta(G, ruta) -> float:
    return sum(G[a][b]["weight"] for a, b in zip(ruta, ruta[1:]))


def acciones_ruta(G, ruta) -> List[str]:
    return [G[a][b]["accion"] for a, b in zip(ruta, ruta[1:])]


# ----------------------------------------------------------------------------
# 6. DIAGRAMAS
# ----------------------------------------------------------------------------
def posicion(estado, desplazamiento=0.18):
    """Posición en el dibujo: la intersección, desplazada según el rumbo."""
    x, y, r = estado
    dx, dy = VECTOR[r]
    return (x + dx * desplazamiento, y + dy * desplazamiento)


def dibujar_grafo_estados(G, ruta, ruta_bfs, ruta_salida: Path):
    fig, ax = plt.subplots(figsize=(11, 9))
    pos = {s: posicion(s) for s in G.nodes}
    en_ruta = set(zip(ruta, ruta[1:]))

    nx.draw_networkx_edges(G, pos, ax=ax, edge_color="#bbbbbb", width=0.8,
                           arrowsize=8, connectionstyle="arc3,rad=0.12")
    nx.draw_networkx_edges(G, pos, ax=ax, edgelist=list(en_ruta), edge_color="#d62728",
                           width=2.8, arrowsize=14, connectionstyle="arc3,rad=0.12")
    colores = ["#2ca02c" if s == INICIO else "#ff7f0e" if es_meta(s) else
               "#d62728" if s in ruta else "#4c78a8" for s in G.nodes]
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=colores, node_size=140)
    # etiqueta de la intersección
    for x in range(N):
        for y in range(N):
            ax.text(x, y - 0.42, f"({x},{y})", ha="center", fontsize=8, color="#555")
    ax.set_title(f"Espacio de estados (x, y, rumbo): {G.number_of_nodes()} nodos, "
                 f"{G.number_of_edges()} aristas\n"
                 f"Verde = inicio · Naranja = meta · Rojo = ruta de menor tiempo",
                 fontsize=11)
    ax.set_aspect("equal"); ax.axis("off")
    fig.tight_layout(); fig.savefig(ruta_salida, dpi=160)


def dibujar_red_vial(tiempos, ruta, ruta_ints_bfs, ruta_salida: Path):
    """Mapa de la ciudad: intersecciones, tiempos por cuadra, obras y ruta óptima."""
    fig, ax = plt.subplots(figsize=(8, 7))
    for k, t in tiempos.items():
        a, b = tuple(k)
        bloq = cuadra_bloqueada(a, b)
        ax.plot([a[0], b[0]], [a[1], b[1]], color="#d62728" if bloq else "#999",
                lw=2, ls="--" if bloq else "-")
        mx, my = (a[0] + b[0]) / 2, (a[1] + b[1]) / 2
        ax.text(mx, my, "OBRA" if bloq else f"{t:.0f}s", fontsize=8, ha="center",
                bbox=dict(fc="white", ec="none", pad=1))
    inter = [(s[0], s[1]) for s in ruta]
    ax.plot(*zip(*inter), color="#1f77b4", lw=4, alpha=.7, label="Ruta de menor tiempo (Dijkstra)")
    ax.plot(*zip(*ruta_ints_bfs), color="#2ca02c", lw=2, ls=":", label="Ruta con menos pasos (BFS)")
    xs, ys = zip(*[(x, y) for x in range(N) for y in range(N)])
    ax.scatter(xs, ys, s=250, c="white", ec="k", zorder=3)
    ax.scatter(*INICIO[:2], s=300, c="#2ca02c", ec="k", zorder=4)
    ax.scatter(*META, s=300, c="#ff7f0e", ec="k", zorder=4)
    ax.legend(loc="upper left", fontsize=8); ax.set_aspect("equal"); ax.axis("off")
    ax.set_title("Red vial modelada (tiempo base por cuadra en segundos)")
    fig.tight_layout(); fig.savefig(ruta_salida, dpi=160)


# ----------------------------------------------------------------------------
# 7. ANÁLISIS DE COMPLEJIDAD (teórico + medición empírica)
# ----------------------------------------------------------------------------
def experimento_escalabilidad(tamanos=(3, 4, 6, 8, 10, 14, 18)):
    """Mide |V|, |E| y tiempo al crecer la ciudad. Debe crecer ~ lineal en |V|."""
    global N, META, OBRAS
    N0, M0, O0 = N, META, OBRAS
    filas = []
    for n in tamanos:
        N, META, OBRAS = n, (n - 1, n - 1), set()
        tiempos = construir_tiempos_base(n, SEMILLA)
        t0 = time.perf_counter()
        G = construir_grafo((0, 0, "E"), tiempos)
        t_gen = time.perf_counter() - t0
        t0 = time.perf_counter()
        resolver(G, (0, 0, "E"))
        t_dij = time.perf_counter() - t0
        filas.append((n, G.number_of_nodes(), G.number_of_edges(), t_gen, t_dij))
    N, META, OBRAS = N0, M0, O0
    return filas


def graficar_escalabilidad(filas, ruta_salida: Path):
    v = [f[1] for f in filas]
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(v, [f[2] for f in filas], "o-"); ax[0].set_xlabel("|V| estados")
    ax[0].set_ylabel("|E| aristas"); ax[0].set_title("|E| ≈ 3·|V| (b = 3)")
    ax[1].plot(v, [f[3] * 1000 for f in filas], "o-", label="Generar grafo (BFS)")
    ax[1].plot(v, [f[4] * 1000 for f in filas], "s-", label="Dijkstra")
    ax[1].set_xlabel("|V| estados"); ax[1].set_ylabel("tiempo (ms)")
    ax[1].legend(); ax[1].set_title("Tiempo de ejecución vs |V|")
    for a in ax: a.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(ruta_salida, dpi=160)


# ----------------------------------------------------------------------------
# 8. PROGRAMA PRINCIPAL
# ----------------------------------------------------------------------------
def main():
    salida = Path(__file__).resolve().parent
    tiempos = construir_tiempos_base(N, SEMILLA)

    print("=" * 70)
    print("PROBLEMA 1 - Ruta de vehículo autónomo (modelo de búsqueda)")
    print("=" * 70)
    print(f"Estado inicial: {INICIO} | Meta: intersección {META} | Ciudad {N}x{N}")
    print(f"Cuadras bloqueadas (obras): {sorted(OBRAS)}\n")

    # Muestra del generador de estados
    print("Sucesores del estado inicial:")
    for a, s2, c in sucesores(INICIO, tiempos):
        print(f"  acción={a:<10} -> {s2}  coste={c:.0f} s")

    t0 = time.perf_counter()
    G = construir_grafo(INICIO, tiempos)
    t_gen = time.perf_counter() - t0
    print(f"\nGrafo generado: |V|={G.number_of_nodes()}  |E|={G.number_of_edges()}  "
          f"({t_gen*1000:.2f} ms)  | dirigido={G.is_directed()}")

    ruta, coste, ruta_bfs = resolver(G, INICIO)
    print("\n--- Ruta de MENOR TIEMPO (Dijkstra) ---")
    print("Estados :", " -> ".join(f"({x},{y},{r})" for x, y, r in ruta))
    print("Acciones:", acciones_ruta(G, ruta))
    print(f"Coste total: {coste:.0f} s | maniobras: {len(ruta)-1}")

    print("\n--- Ruta con MENOS PASOS (BFS, ignora costes) ---")
    print("Acciones:", acciones_ruta(G, ruta_bfs))
    print(f"Coste real: {coste_ruta(G, ruta_bfs):.0f} s | maniobras: {len(ruta_bfs)-1}")

    # Diagramas
    dibujar_grafo_estados(G, ruta, ruta_bfs, salida / "p1_grafo_estados.png")
    dibujar_red_vial(tiempos, ruta, [(s[0], s[1]) for s in ruta_bfs], salida / "p1_red_vial.png")

    # Complejidad empírica
    filas = experimento_escalabilidad()
    print("\nESCALABILIDAD (ciudad n x n, sin obras):")
    print(f"{'n':>3} {'|V|':>6} {'|E|':>6} {'generar ms':>11} {'dijkstra ms':>12}")
    for n, v, e, tg, td in filas:
        print(f"{n:>3} {v:>6} {e:>6} {tg*1000:>11.2f} {td*1000:>12.2f}")
    graficar_escalabilidad(filas, salida / "p1_escalabilidad.png")

    print("""
ANÁLISIS DE COMPLEJIDAD
  Tamaño del espacio : |V| <= 4·n² (n² intersecciones x 4 rumbos; los estados que miran
                       hacia fuera del mapa en el borde son inalcanzables)  -> O(n²)
  Factor ramificación: b = 3 (recto, izquierda, derecha)  =>  |E| <= 3|V|
  Generación (BFS)   : O(|V| + |E|) = O(n²) en tiempo y espacio
  Dijkstra (heap)    : O((|V| + |E|) log |V|) = O(n² log n)
  BFS sin pesos      : O(|V| + |E|); óptimo en nº de pasos, NO en tiempo.
  Sin detección de repetidos el árbol de búsqueda sería O(b^d) = O(3^d).
""")
    print("Archivos generados: p1_grafo_estados.png, p1_red_vial.png, p1_escalabilidad.png")
    # Muestra TODAS las gráficas en pantalla (ventana, Jupyter o Colab)
    plt.show()


if __name__ == "__main__":
    main()
