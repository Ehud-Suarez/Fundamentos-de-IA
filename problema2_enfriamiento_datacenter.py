"""
ACTIVIDAD 2 (Secciones 2.1 y 2.2) - Fundamentos de Inteligencia Artificial
PROBLEMA 2: Ajuste del enfriamiento de un centro de datos (inspirado en el caso
            Google/DeepMind de reducción de energía de enfriamiento)

Modelado como PROBLEMA DE BÚSQUEDA
----------------------------------
  Estado    : (consigna, ventilador)
                consigna   = temperatura del aire frío de suministro en °C (18..27)
                ventilador = velocidad de los ventiladores en % (30..100)
  Estado inicial : (18, 100)  -> operación muy conservadora (mucho enfriamiento).
  Acciones  : consigna +1 / consigna -1 / ventilador +10 / ventilador -10.
  Restricción de SEGURIDAD: temperatura de entrada a los servidores <= 27 °C
              (límite superior del rango recomendado por ASHRAE). Los estados que
              la violan son inválidos y el generador NO los produce.
  Coste     : energía consumida en un paso de control de 5 min (kWh)
              + penalización fija por cambiar un parámetro (desgaste/estabilidad).
  Meta      : estado seguro con PUE <= PUE_META.

Espacio de búsqueda -> GRAFO DIRIGIDO Y PONDERADO (networkx.DiGraph).

IMPORTANTE: el modelo físico es una SIMPLIFICACIÓN DIDÁCTICA con parámetros
supuestos (no son datos reales de Google). Sirve para practicar el modelado como
búsqueda; las magnitudes no deben presentarse como resultados de Google.

Librerías: networkx, matplotlib, numpy (pip) + dataclasses, time, pathlib, typing
           (biblioteca estándar).
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

# ----------------------------------------------------------------------------
# 1. PARÁMETROS DEL MODELO FÍSICO SIMPLIFICADO (supuestos didácticos)
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class Parametros:
    carga_ti_kw: float = 1000.0      # potencia de los servidores (carga TI)
    otros_kw: float = 80.0           # pérdidas eléctricas/iluminación (fijas)
    ventilador_max_kw: float = 60.0  # potencia de ventiladores al 100 %
    salto_termico_100: float = 4.0   # °C que se calienta el aire al 100 % de flujo
    t_entrada_max: float = 27.0      # límite de seguridad en servidores (°C)
    cop_base: float = 4.0            # eficiencia del chiller a 18 °C
    cop_por_grado: float = 0.2       # mejora del COP por cada °C de consigna
    paso_horas: float = 5 / 60       # duración de un paso de control (5 min)
    penal_cambio_kwh: float = 0.5    # coste de cada cambio de parámetro
    pue_meta: float = 1.33           # objetivo de eficiencia


P = Parametros()
CONSIGNAS = range(18, 28)            # 18..27 °C
VENTILADORES = range(30, 101, 10)    # 30..100 %
INICIO = (18, 100)


def temperatura_entrada(consigna: int, vent: int) -> float:
    """El aire se calienta más cuando el flujo (ventilador) es menor."""
    return consigna + P.salto_termico_100 * (100 / vent)


def es_seguro(estado) -> bool:
    return temperatura_entrada(*estado) <= P.t_entrada_max


def potencia_enfriamiento_kw(estado) -> float:
    """Ventiladores (ley de afinidad: potencia ∝ velocidad³) + chiller (calor / COP)."""
    consigna, vent = estado
    p_vent = P.ventilador_max_kw * (vent / 100) ** 3
    cop = P.cop_base + P.cop_por_grado * (consigna - 18)
    p_chiller = (P.carga_ti_kw + p_vent) / cop
    return p_vent + p_chiller


def pue(estado) -> float:
    """PUE = energía total / energía de TI."""
    return (P.carga_ti_kw + P.otros_kw + potencia_enfriamiento_kw(estado)) / P.carga_ti_kw


def es_meta(estado) -> bool:
    return es_seguro(estado) and pue(estado) <= P.pue_meta


# ----------------------------------------------------------------------------
# 2. GENERADOR DE ESTADOS (función sucesor)
# ----------------------------------------------------------------------------
ACCIONES = {
    "consigna +1 °C": (+1, 0),
    "consigna -1 °C": (-1, 0),
    "ventilador +10 %": (0, +10),
    "ventilador -10 %": (0, -10),
}


def sucesores(estado) -> Iterator[Tuple[str, tuple, float]]:
    """
    GENERADOR DE ESTADOS: produce (acción, estado_siguiente, coste).
    Se descartan estados fuera de rango o que violan la restricción de seguridad.
    Complejidad por llamada: O(b), con b = 4.
    """
    c, v = estado
    for nombre, (dc, dv) in ACCIONES.items():
        s2 = (c + dc, v + dv)
        if s2[0] not in CONSIGNAS or s2[1] not in VENTILADORES:
            continue
        if not es_seguro(s2):                       # restricción dura
            continue
        energia = potencia_enfriamiento_kw(s2) * P.paso_horas      # kWh en el paso
        yield nombre, s2, energia + P.penal_cambio_kwh


# ----------------------------------------------------------------------------
# 3. GRAFO DEL ESPACIO DE ESTADOS
# ----------------------------------------------------------------------------
def construir_grafo(inicio=INICIO) -> nx.DiGraph:
    """BFS con el generador: guarda cada estado alcanzable y cada acción. O(V + E)."""
    G = nx.DiGraph()
    G.add_node(inicio)
    cola = deque([inicio])
    while cola:
        s = cola.popleft()
        for accion, s2, coste in sucesores(s):
            if s2 not in G:
                G.add_node(s2)
                cola.append(s2)
            G.add_edge(s, s2, weight=coste, accion=accion)
    for s in G.nodes:                               # atributos útiles para graficar
        G.nodes[s]["pue"] = pue(s)
        G.nodes[s]["kw"] = potencia_enfriamiento_kw(s)
    return G


# ----------------------------------------------------------------------------
# 4. SOLUCIÓN
# ----------------------------------------------------------------------------
def resolver(G: nx.DiGraph, inicio=INICIO):
    """Dijkstra: secuencia de ajustes de menor energía hasta un estado meta."""
    metas = [s for s in G.nodes if es_meta(s)]
    if not metas:
        return None, None, None
    dist, caminos = nx.single_source_dijkstra(G, inicio, weight="weight")
    mejor = min((m for m in metas if m in dist), key=lambda m: dist[m])
    bfs = nx.single_source_shortest_path(G, inicio)           # menos cambios
    meta_bfs = min((m for m in metas if m in bfs), key=lambda m: len(bfs[m]))
    return caminos[mejor], dist[mejor], bfs[meta_bfs]


def acciones_ruta(G, ruta) -> List[str]:
    return [G[a][b]["accion"] for a, b in zip(ruta, ruta[1:])]


def coste_ruta(G, ruta) -> float:
    return sum(G[a][b]["weight"] for a, b in zip(ruta, ruta[1:]))


# ----------------------------------------------------------------------------
# 5. DIAGRAMAS
# ----------------------------------------------------------------------------
def dibujar_grafo(G, ruta, ruta_salida: Path):
    """Nodos en una malla (consigna, ventilador), coloreados por potencia de enfriamiento."""
    fig, ax = plt.subplots(figsize=(11, 7.5))
    pos = {s: (s[0], s[1] / 10) for s in G.nodes}
    kw = np.array([G.nodes[s]["kw"] for s in G.nodes])

    nx.draw_networkx_edges(G, pos, ax=ax, edge_color="#c8c8c8", width=0.8,
                           arrowsize=7, connectionstyle="arc3,rad=0.10", node_size=700)
    en_ruta = list(zip(ruta, ruta[1:]))
    nx.draw_networkx_edges(G, pos, ax=ax, edgelist=en_ruta, edge_color="#d62728",
                           width=3, arrowsize=14, connectionstyle="arc3,rad=0.10", node_size=700)
    nodos = nx.draw_networkx_nodes(G, pos, ax=ax, node_color=kw, cmap="viridis_r",
                                   node_size=700, edgecolors="k", linewidths=0.6)
    for s in G.nodes:
        borde = "#2ca02c" if s == INICIO else "#ff7f0e" if es_meta(s) else None
        if borde:
            ax.scatter(*pos[s], s=1000, facecolors="none", edgecolors=borde, linewidths=3)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=6,
                            labels={s: f"{G.nodes[s]['kw']:.0f}" for s in G.nodes})
    cb = fig.colorbar(nodos, ax=ax, shrink=.8); cb.set_label("Potencia de enfriamiento (kW)")
    ax.set_xticks(list(CONSIGNAS)); ax.set_yticks([v / 10 for v in VENTILADORES])
    ax.set_yticklabels([str(v) for v in VENTILADORES])
    ax.set_xlabel("Consigna de aire frío (°C)"); ax.set_ylabel("Velocidad de ventiladores (%)")
    ax.set_title(f"Espacio de estados seguros: {G.number_of_nodes()} nodos, {G.number_of_edges()} aristas\n"
                 "Verde = inicio · Naranja = metas (PUE ≤ %.2f) · Rojo = ruta de menor energía" % P.pue_meta)
    ax.grid(alpha=.2); fig.tight_layout(); fig.savefig(ruta_salida, dpi=160)


def dibujar_superficie(ruta_salida: Path):
    """Mapa de calor del PUE y zona prohibida (T de entrada > 27 °C)."""
    cs, vs = list(CONSIGNAS), list(VENTILADORES)
    Z = np.array([[pue((c, v)) if es_seguro((c, v)) else np.nan for c in cs] for v in vs])
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(Z, origin="lower", aspect="auto", cmap="viridis_r",
                   extent=[cs[0] - .5, cs[-1] + .5, vs[0] - 5, vs[-1] + 5])
    ax.contour(cs, vs, np.nan_to_num(Z, nan=9), levels=[P.pue_meta], colors="orange")
    fig.colorbar(im, ax=ax, label="PUE (blanco = estado inseguro)")
    ax.set_xlabel("Consigna (°C)"); ax.set_ylabel("Ventiladores (%)")
    ax.set_title("PUE por estado (naranja: frontera PUE = %.2f)" % P.pue_meta)
    fig.tight_layout(); fig.savefig(ruta_salida, dpi=160)


# ----------------------------------------------------------------------------
# 6. COMPLEJIDAD EMPÍRICA: refinar la discretización (más estados)
# ----------------------------------------------------------------------------
def escalabilidad(factores=(1, 2, 4, 8)):
    """Divide los pasos de consigna y ventilador entre k para tener más estados."""
    filas = []
    for k in factores:
        pasos_c, pasos_v = 1 / k, 10 / k
        cs = [18 + i * pasos_c for i in range(int(9 * k) + 1)]
        vs = [30 + i * pasos_v for i in range(int(70 / pasos_v) + 1)]
        G = nx.DiGraph()
        t0 = time.perf_counter()
        idx = lambda c, v: (round(c, 3), round(v, 3))
        for c in cs:
            for v in vs:
                if (c + P.salto_termico_100 * 100 / v) <= P.t_entrada_max:
                    G.add_node(idx(c, v))
        for (c, v) in list(G.nodes):
            for dc, dv in ((pasos_c, 0), (-pasos_c, 0), (0, pasos_v), (0, -pasos_v)):
                s2 = idx(c + dc, v + dv)
                if s2 in G:
                    G.add_edge((c, v), s2, weight=1.0)
        t_gen = time.perf_counter() - t0
        t0 = time.perf_counter()
        nx.single_source_dijkstra(G, idx(18, 100), weight="weight")
        t_dij = time.perf_counter() - t0
        filas.append((k, G.number_of_nodes(), G.number_of_edges(), t_gen, t_dij))
    return filas


# ----------------------------------------------------------------------------
# 7. PROGRAMA PRINCIPAL
# ----------------------------------------------------------------------------
def main():
    salida = Path(__file__).resolve().parent
    print("=" * 70)
    print("PROBLEMA 2 - Enfriamiento de centro de datos (modelo de búsqueda)")
    print("=" * 70)
    print(f"Estado inicial: consigna={INICIO[0]} °C, ventilador={INICIO[1]} %")
    print(f"  Enfriamiento inicial: {potencia_enfriamiento_kw(INICIO):.1f} kW | PUE={pue(INICIO):.3f}")
    print(f"Meta: estado seguro (T entrada <= {P.t_entrada_max} °C) con PUE <= {P.pue_meta}\n")

    print("Sucesores del estado inicial:")
    for a, s2, c in sucesores(INICIO):
        print(f"  acción={a:<17} -> {s2}  coste={c:.2f} kWh")

    t0 = time.perf_counter()
    G = construir_grafo()
    print(f"\nGrafo generado: |V|={G.number_of_nodes()}  |E|={G.number_of_edges()}  "
          f"({(time.perf_counter()-t0)*1000:.2f} ms) | dirigido={G.is_directed()}")
    n_metas = sum(es_meta(s) for s in G.nodes)
    print(f"Estados meta en el grafo: {n_metas}")

    ruta, coste, ruta_bfs = resolver(G)
    if ruta is None:
        print("No existe estado meta con estos parámetros."); return
    print("\n--- Secuencia de MENOR ENERGÍA (Dijkstra) ---")
    for (a, b) in zip(ruta, ruta[1:]):
        print(f"  {a} --{G[a][b]['accion']:<17}--> {b}  PUE={G.nodes[b]['pue']:.3f}  "
              f"enfriamiento={G.nodes[b]['kw']:.1f} kW  T_entrada={temperatura_entrada(*b):.1f} °C")
    print(f"Coste total: {coste:.2f} kWh | cambios: {len(ruta)-1}")
    print("\n--- Secuencia con MENOS CAMBIOS (BFS) ---")
    print(f"  {len(ruta_bfs)-1} cambios, coste {coste_ruta(G, ruta_bfs):.2f} kWh")

    final = ruta[-1]
    ahorro = 100 * (1 - potencia_enfriamiento_kw(final) / potencia_enfriamiento_kw(INICIO))
    print(f"\nEstado final {final}: PUE {pue(INICIO):.3f} -> {pue(final):.3f} | "
          f"enfriamiento {potencia_enfriamiento_kw(INICIO):.0f} -> {potencia_enfriamiento_kw(final):.0f} kW "
          f"({ahorro:.1f} % menos, SOLO en este modelo didáctico)")

    # Mejor estado global (para contrastar con el estado meta hallado)
    seguros = [s for s in G.nodes]
    optimo = min(seguros, key=potencia_enfriamiento_kw)
    print(f"Mínimo global de enfriamiento en el espacio: {optimo} -> "
          f"{potencia_enfriamiento_kw(optimo):.1f} kW (PUE {pue(optimo):.3f})")

    dibujar_grafo(G, ruta, salida / "p2_grafo_estados.png")
    dibujar_superficie(salida / "p2_mapa_pue.png")

    filas = escalabilidad()
    print("\nESCALABILIDAD (refinando la discretización por k):")
    print(f"{'k':>3} {'|V|':>7} {'|E|':>7} {'generar ms':>11} {'dijkstra ms':>12}")
    for k, v, e, tg, td in filas:
        print(f"{k:>3} {v:>7} {e:>7} {tg*1000:>11.2f} {td*1000:>12.2f}")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot([f[1] for f in filas], [f[3] * 1000 for f in filas], "o-", label="Generar grafo")
    ax.plot([f[1] for f in filas], [f[4] * 1000 for f in filas], "s-", label="Dijkstra")
    ax.set_xlabel("|V| estados"); ax.set_ylabel("tiempo (ms)"); ax.legend(); ax.grid(alpha=.3)
    ax.set_title("Escalabilidad del modelo 2")
    fig.tight_layout(); fig.savefig(salida / "p2_escalabilidad.png", dpi=160)

    print("""
ANÁLISIS DE COMPLEJIDAD
  Tamaño del espacio : |V| <= |consignas| x |ventiladores| = 10 x 8 = 80 posibles, de los cuales sólo 25 son
                       seguros (T entrada <= 27 °C). Con d parámetros discretos de m niveles: O(m^d).
  Factor ramificación: b = 4 (dos parámetros, subir/bajar)  =>  |E| <= 4|V|
  Generación (BFS)   : O(|V| + |E|)
  Dijkstra (heap)    : O((|V| + |E|) log |V|)
  BFS sin pesos      : O(|V| + |E|); mínimo nº de cambios, no mínima energía.
  Sin control de repetidos el árbol sería O(b^d_sol) = O(4^d_sol) (explosión).
  Con más variables de control (maldición de la dimensionalidad) el espacio crece
  exponencialmente -> por eso en la práctica se usan aprendizaje automático y
  heurísticas (A*, algoritmos genéticos) en vez de búsqueda exhaustiva.
""")
    print("Archivos: p2_grafo_estados.png, p2_mapa_pue.png, p2_escalabilidad.png")
    # Muestra TODAS las gráficas en pantalla (ventana, Jupyter o Colab)
    plt.show()


if __name__ == "__main__":
    main()
