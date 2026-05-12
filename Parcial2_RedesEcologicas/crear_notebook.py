"""
Genera redes_ecologicas_parcial2.ipynb con teoría detallada y código documentado.
Ejecutar: python crear_notebook.py
"""
import json, textwrap

def md(src): return {"cell_type":"markdown","metadata":{},"source":textwrap.dedent(src).lstrip()}
def code(src): return {"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":textwrap.dedent(src).lstrip()}

cells = []

# ─── PORTADA ────────────────────────────────────────────────────────────────
cells.append(md(r"""
# Redes Ecológicas: Estructura, Dinámica y Estabilidad
## Parcial 2 — Modelamiento Computacional

**Contenidos:**
1. Fundamentos: grafos, matrices y representación de interacciones
2. Métricas estructurales: conectancia, centralidad, modularidad, asortatividad
3. Niveles tróficos y flujo de energía
4. Estabilidad dinámica y el problema complejidad-estabilidad (May 1972)
5. Análisis espectral y tiempos de relajación
6. Cascadas de extinción y robustez
7. Estructura Bow-Tie (componentes fuertemente conectados)
8. Dinámica de poblaciones: modelo Lotka-Volterra generalizado

---
> **Convención de lectura:** cada sección inicia con la teoría completa y concluye con su implementación numérica. Los bloques de código ilustran exactamente lo derivado en la teoría previa.
"""))

# ─── INSTALACIÓN ─────────────────────────────────────────────────────────────
cells.append(code(r"""
# En Google Colab estas librerías ya están disponibles.
# Localmente: pip install networkx numpy scipy matplotlib pandas seaborn
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd
from scipy.integrate import solve_ivp
import warnings, os
warnings.filterwarnings('ignore')

RNG = np.random.default_rng(42)
plt.rcParams.update({'figure.dpi': 110, 'font.size': 11,
                     'axes.spines.top': False, 'axes.spines.right': False})
print("Entorno listo.")
"""))

# ─── SECCIÓN 1: TEORÍA COMPLETA ───────────────────────────────────────────────
cells.append(md(r"""
---
## 1. Fundamentos de Ciencia de Redes Aplicada a Ecología

### 1.1 El lenguaje de los grafos

Un **grafo** $G = (V, E)$ consiste en un conjunto de *vértices* $V$ y un conjunto de
*aristas* $E \subseteq V \times V$. En ecología los vértices son **especies** (o grupos
tróficos) y las aristas son **interacciones biológicas**.

La distinción más importante para redes ecológicas es si las interacciones tienen
dirección:

- **Grafo no dirigido:** la arista $\{i, j\}$ es simétrica — si $i$ interactúa con $j$
  entonces $j$ interactúa con $i$ con la misma naturaleza. Apropiado para
  **mutualismos** (polinización, dispersión de semillas) donde ambas partes se
  benefician.

- **Grafo dirigido (dígrafo):** la arista $(j \to i)$ tiene orientación. Apropiado para
  **redes tróficas** donde el flujo de energía va de la presa al depredador.
  La relación no es simétrica: el hecho de que $i$ deprede a $j$ no implica que $j$
  deprede a $i$.

En este notebook trabajamos con **redes tróficas dirigidas**. La convención adoptada
(estándar en la literatura) es:

$$j \longrightarrow i \quad \Leftrightarrow \quad \text{la especie } j \text{ es presa de la especie } i$$

El flujo de energía (biomasa) va de $j$ a $i$.

### 1.2 La Matriz de Adyacencia $A$

La herramienta algebraica central es la **matriz de adyacencia** $A \in \{0,1\}^{n \times n}$:

$$A_{ij} = \begin{cases} 1 & \text{si existe la arista } j \to i \quad (j \text{ es presa de } i)\\ 0 & \text{en caso contrario} \end{cases}$$

Notar la convención: el **índice de fila** $i$ es el depredador y el **índice de columna**
$j$ es la presa. Esto hace que la columna $j$ enumere a todos los depredadores de $j$.

Para redes tróficas, $A$ es en general **asimétrica** ($A \neq A^\top$): el hecho de
que $j$ sea presa de $i$ no implica que $i$ sea presa de $j$.

**Redes ponderadas.** Si la interacción tiene intensidad medible (flujo de carbono en
$\text{mg C m}^{-2}\text{d}^{-1}$, frecuencia de visitas), se usa una
**matriz de pesos** $W$ donde $W_{ij} = w_{ij} > 0$ si existe la arista $j \to i$,
y $0$ en caso contrario.

### 1.3 Detección de ciclos mediante potencias de $A$

Una propiedad algebraica fundamental: el elemento $(A^r)_{ij}$ cuenta el número de
**caminos dirigidos de longitud** $r$ que van de $j$ a $i$.

La demostración es por inducción: $(A^1)_{ij} = A_{ij}$ cuenta caminos de longitud 1;
si $(A^{r-1})_{kj}$ cuenta caminos de longitud $r-1$ de $j$ a $k$, entonces

$$(A^r)_{ij} = \sum_k A_{ik}(A^{r-1})_{kj}$$

suma sobre todos los intermediarios $k$ (un paso desde $k$ a $i$, más $r-1$ pasos
de $j$ a $k$).

En consecuencia, la **traza**

$$\mathrm{Tr}(A^r) = \sum_i (A^r)_{ii} = \text{número de ciclos dirigidos de longitud } r$$

porque $(A^r)_{ii}$ cuenta caminos que parten de $i$ y regresan a $i$ en $r$ pasos.

Una red **acíclica** (DAG, *directed acyclic graph*) tiene $\mathrm{Tr}(A^r) = 0$
para todo $r \geq 1$, lo que implica que todos sus autovalores son cero.
Los ciclos en una red trófica representan **omnívoros** o **bucles de retroalimentación**
que, como veremos, tienen consecuencias directas sobre la estabilidad.
"""))

cells.append(code(r"""
# ── Red mínima: 5 especies tróficas ──────────────────────────────────────
# Etiquetas: 1=Fitoplancton, 2=Zooplancton, 3=Pez pequeño, 4=Pez grande, 5=Ave
# Arista j→i: j es presa de i

especies = {1:'Fitoplancton', 2:'Zooplancton', 3:'Pez pequeño', 4:'Pez grande', 5:'Ave'}
G_mini = nx.DiGraph()
G_mini.add_nodes_from(especies)
interacciones = [(1,2),(1,3),(2,3),(2,4),(3,4),(3,5),(4,5)]
G_mini.add_edges_from(interacciones)

nodos_ord = sorted(G_mini.nodes())
A = nx.to_numpy_array(G_mini, nodelist=nodos_ord)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Grafo
pos = {1:(0,0), 2:(1,1), 3:(2,0), 4:(3,1), 5:(4,0)}
nx.draw_networkx(G_mini, pos=pos, ax=axes[0], labels=especies,
                 node_color='steelblue', node_size=1400, font_color='white',
                 font_size=8, edge_color='#444', arrows=True, arrowsize=22)
axes[0].set_title('Red trófica (5 especies)\n$j \\to i$ significa $j$ es presa de $i$', fontweight='bold')
axes[0].axis('off')

# Matriz A con anotaciones
im = axes[1].imshow(A, cmap='Blues', vmin=0, vmax=1)
n = len(nodos_ord)
axes[1].set_xticks(range(n)); axes[1].set_yticks(range(n))
etiq = [especies[i] for i in nodos_ord]
axes[1].set_xticklabels(etiq, rotation=40, ha='right', fontsize=9)
axes[1].set_yticklabels(etiq, fontsize=9)
axes[1].set_xlabel('Presa $j$ (columna)'); axes[1].set_ylabel('Depredador $i$ (fila)')
axes[1].set_title('Matriz de adyacencia $A$\n$A_{ij}=1$ si $j$ es presa de $i$', fontweight='bold')
for i in range(n):
    for j in range(n):
        axes[1].text(j, i, int(A[i,j]), ha='center', va='center', fontsize=13,
                     color='white' if A[i,j] else 'black')
plt.colorbar(im, ax=axes[1])
plt.tight_layout(); plt.savefig('fig1_red_minima.png', bbox_inches='tight'); plt.show()
print(f"A (fila=depredador, col=presa):\n{A.astype(int)}")
"""))

cells.append(code(r"""
# ── Verificación algebraica: Tr(A^r) cuenta ciclos ───────────────────────
print("Análisis de ciclos vía potencias de A:")
print(f"  ¿Es DAG (sin ciclos)? {nx.is_directed_acyclic_graph(G_mini)}")
print()
for r in range(1, 7):
    tr = int(np.round(np.trace(np.linalg.matrix_power(A, r))))
    print(f"  Tr(A^{r}) = {tr:>3}  →  {tr} camino(s) cíclico(s) de longitud {r}")

print("\nAutovalores de A:")
evs = np.linalg.eigvals(A)
for ev in sorted(evs, key=lambda x: -abs(x)):
    print(f"  {np.real(ev):+.4f}{np.imag(ev):+.4f}i")
print("(Todos ≈ 0 ⟺ red acíclica pura)")
"""))

# ─── SECCIÓN 2: MÉTRICAS ─────────────────────────────────────────────────────
cells.append(md(r"""
---
## 2. Métricas Estructurales

Las métricas estructurales cuantifican la **topología** de la red sin referencia a
la dinámica. Se dividen en dos niveles: métricas globales (del grafo completo) y
métricas de nodo (de cada especie).

### 2.1 Métricas Globales

#### Conectancia $C$

La conectancia mide qué fracción de las interacciones posibles realmente existe:

$$C = \frac{m}{n^2}$$

donde $m = |E|$ es el número de aristas y $n = |V|$ el de nodos. El denominador $n^2$
corresponde al máximo teórico para un dígrafo con auto-loops. Sin ellos el máximo sería
$n(n-1)$, pero la convención $n^2$ es estándar en ecología porque simplifica la relación
con otros parámetros del modelo de May.

Empíricamente, la conectancia de redes tróficas reales permanece aproximadamente
**constante** (alrededor de $C \approx 0.1\text{–}0.2$) independientemente del tamaño
$n$. Esto implica que $m \propto n^2$: las especies más integradas en ecosistemas grandes
tienen, en promedio, más interacciones.

#### Diámetro $d(G)$

El **diámetro** es la longitud del camino geodésico más largo entre cualquier par de
nodos conectados:

$$d(G) = \max_{i,j \in V} \delta(i,j)$$

donde $\delta(i,j)$ es la distancia (número mínimo de aristas) de $i$ a $j$.
En redes tróficas, el diámetro refleja cuántos pasos separan a los productores primarios
de los depredadores tope.

#### Modularidad $Q$

La modularidad cuantifica la intensidad con la que la red se divide en
**módulos** o compartimentos:

$$Q = \frac{1}{2m} \sum_{i,j} \left[ A_{ij} - \frac{k_i k_j}{2m} \right] \delta(c_i, c_j)$$

donde $k_i$ es el grado del nodo $i$, $c_i$ es el módulo al que pertenece, y $\delta$
es la función delta de Kronecker. El término $\frac{k_i k_j}{2m}$ es la probabilidad
esperada de la arista $\{i,j\}$ en un grafo aleatorio con la misma distribución de
grados (*null model* de configuración).

Así, $Q > 0$ indica que hay más aristas dentro de los módulos de las que cabría esperar
por azar. Empíricamente:

- $Q \lesssim 0.2$: sin estructura modular significativa.
- $Q \in [0.3, 0.7]$: modularidad intermedia, típica de redes ecológicas.
- $Q \gtrsim 0.7$: modularidad muy fuerte (raro en redes naturales).

#### Asortatividad $r$

La asortatividad de grado mide la **correlación de Pearson** entre los grados de los
nodos en los extremos de cada arista:

$$r = \frac{\displaystyle\sum_{(i,j)\in E} k_i k_j - \left[\frac{1}{m}\sum_{(i,j)\in E}\frac{k_i+k_j}{2}\right]^2}{\displaystyle\frac{1}{m}\sum_{(i,j)\in E}\frac{k_i^2+k_j^2}{2} - \left[\frac{1}{m}\sum_{(i,j)\in E}\frac{k_i+k_j}{2}\right]^2}$$

- $r > 0$ (**asortativa**): nodos de alto grado tienden a conectarse entre sí.
  Típico en redes sociales (los "populares" se agrupan).
- $r < 0$ (**disasortativa**): nodos de alto grado se conectan a nodos de bajo grado.
  **Típico en redes biológicas.** Los depredadores generalistas (hubs) se conectan
  a presas especialistas.
- $r \approx 0$: neutral.

La disasortatividad ecológica tiene implicaciones evolutivas: protege el núcleo del
ecosistema de cascadas de extinción que comienzan en la periferia.

### 2.2 Métricas de Nodo: Centralidad

Las centralidades identifican las **especies clave** (*keystone species*), cuya remoción
tiene efectos desproporcionados sobre el ecosistema.

#### Grado y grado dirigido

En un dígrafo, el grado se descompone:

$$k_i = k_i^{in} + k_i^{out}$$

- $k_i^{in} = \sum_j A_{ij}$: número de **presas** de la especie $i$ (amplitud de dieta).
- $k_i^{out} = \sum_j A_{ji}$: número de **depredadores** de la especie $i$ (presión de depredación).

Productores primarios tienen $k^{in} = 0$; depredadores tope tienen $k^{out} = 0$.

#### Centralidad de Autovector

La centralidad de autovector $x_i$ reconoce que la importancia de un nodo depende de
la importancia de sus vecinos. Se define como el autovector correspondiente al mayor
autovalor $\lambda_1$ de $A$:

$$A \mathbf{x} = \lambda_1 \mathbf{x} \quad \Rightarrow \quad x_i = \frac{1}{\lambda_1}\sum_j A_{ij} x_j$$

La especie $i$ tiene centralidad alta si sus depredadores también la tienen. Identifica
las **presas más valiosas** del sistema.

#### Centralidad de Intermediación (Betweenness)

$$B_i = \sum_{s \neq t \neq i} \frac{\sigma_{st}(i)}{\sigma_{st}}$$

donde $\sigma_{st}$ es el número total de caminos geodésicos de $s$ a $t$, y $\sigma_{st}(i)$
los que pasan por $i$. Un nodo con alta betweenness actúa como **puente** en el flujo
de energía. Su eliminación puede desconectar subgrafos enteros — son las especies más
críticas para la conectividad trófica.

### 2.3 Nivel Trófico

El nivel trófico $x_i$ generaliza la noción de "posición en la cadena alimentaria":

$$x_i = 1 + \frac{1}{k_i^{in}} \sum_{j: A_{ij}=1} x_j \quad \text{(para } k_i^{in}>0\text{)}$$

Los **productores primarios** ($k_i^{in} = 0$) tienen $x_i = 1$ por definición.
Los herbívoros tienen $x \approx 2$; los depredadores secundarios $x \approx 3$; etc.

En forma matricial: sea $D = \mathrm{diag}(k^{in})$ y $\mathbf{1}$ el vector de unos.
La ecuación recursiva se reescribe como:

$$D\,\mathbf{x} = D\,\mathbf{1} + A\,\mathbf{x}$$
$$(D - A)\,\mathbf{x} = D\,\mathbf{1}$$
$$\mathbf{x} = (D - A)^{-1} D\,\mathbf{1}$$

La matriz $(D - A)$ es singular sólo si existen ciclos que hagan el sistema
indeterminado; en ese caso se usa mínimos cuadrados.
"""))

cells.append(code(r"""
# ── Funciones de métricas estructurales ──────────────────────────────────

def metricas_globales(G):
    n, m = G.number_of_nodes(), G.number_of_edges()
    Gu = G.to_undirected()
    Gu.remove_edges_from(nx.selfloop_edges(Gu))
    comp = max((Gu.subgraph(c) for c in nx.connected_components(Gu)), key=len)
    comunidades = nx.community.greedy_modularity_communities(Gu)
    return {
        'n_nodos': n, 'n_aristas': m,
        'conectancia C': m / n**2,
        'diametro': nx.diameter(comp),
        'modularidad Q': nx.community.modularity(Gu, comunidades),
        'asortatividad r': nx.degree_assortativity_coefficient(G),
        'n_modulos': len(comunidades),
    }

def nivel_trofico(G):
    '''Resuelve (D-A)x = D·1 para el vector de niveles tróficos.'''
    nodos = sorted(G.nodes())
    A = nx.to_numpy_array(G, nodelist=nodos)
    k_in = A.sum(axis=1)
    D = np.diag(k_in)
    niveles = np.ones(len(nodos))
    cons = k_in > 0
    if cons.any():
        M_sub = (D - A)[np.ix_(cons, cons)]
        rhs = k_in[cons] + A[np.ix_(cons, ~cons)].sum(axis=1)
        try:
            niveles[cons] = np.linalg.solve(M_sub, rhs)
        except np.linalg.LinAlgError:
            niveles[cons] = np.linalg.lstsq(M_sub, rhs, rcond=None)[0]
    return {n: niveles[i] for i, n in enumerate(nodos)}

def centralidades_df(G, etiquetas=None):
    nodos = sorted(G.nodes())
    try:
        eig = nx.eigenvector_centrality(G, max_iter=2000)
    except nx.PowerIterationFailedConvergence:
        eig = {n: 0.0 for n in G}
    bet = nx.betweenness_centrality(G)
    niv = nivel_trofico(G)
    rows = []
    for n in nodos:
        lbl = etiquetas[n] if etiquetas else str(n)
        rows.append({'Especie': lbl,
                     'k_in (presas)': G.in_degree(n),
                     'k_out (depred.)': G.out_degree(n),
                     'C. Autovector': round(eig[n], 4),
                     'Betweenness': round(bet[n], 4),
                     'Nivel trófico': round(niv[n], 3)})
    return pd.DataFrame(rows).set_index('Especie')

# Resultados para la red mínima
print("=== Métricas globales — red mínima ===")
mg = metricas_globales(G_mini)
for k, v in mg.items():
    print(f"  {k:<20} = {v:.4f}" if isinstance(v, float) else f"  {k:<20} = {v}")

print("\n=== Centralidades por especie ===")
display(centralidades_df(G_mini, especies))
"""))

cells.append(code(r"""
# ── Visualización: red con nivel trófico + scatter in/out-degree ──────────
niv_mini = nivel_trofico(G_mini)
nods = sorted(G_mini.nodes())
vals = np.array([niv_mini[n] for n in nods])
cmap = plt.cm.YlOrRd

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Izquierda: red con posición vertical = nivel trófico
ax = axes[0]
pos_trop = {n: (i * 1.5, niv_mini[n]) for i, n in enumerate(nods)}
nc = [cmap((niv_mini[n] - 1) / (vals.max() - 1 + 1e-9)) for n in nods]
nx.draw_networkx(G_mini, pos=pos_trop, ax=ax, labels=especies,
                 node_color=nc, node_size=1300, font_size=8,
                 edge_color='gray', arrows=True, arrowsize=20)
ax.set_ylabel('Nivel trófico')
ax.set_title('Red trófica\n(posición vertical = nivel trófico)', fontweight='bold')
ax.axis('on'); ax.set_xticks([])
sm = plt.cm.ScalarMappable(cmap=cmap, norm=mcolors.Normalize(1, vals.max()))
plt.colorbar(sm, ax=ax, label='Nivel trófico')

# Derecha: in-degree vs out-degree, coloreado por nivel trófico
ax2 = axes[1]
in_d  = [G_mini.in_degree(n)  for n in nods]
out_d = [G_mini.out_degree(n) for n in nods]
sc = ax2.scatter(in_d, out_d, c=vals, cmap=cmap, s=300, edgecolors='k', linewidths=1.5, zorder=5)
for i, n in enumerate(nods):
    ax2.annotate(especies[n], (in_d[i]+0.06, out_d[i]+0.06), fontsize=9)
ax2.set_xlabel('$k^{in}$ — número de presas')
ax2.set_ylabel('$k^{out}$ — número de depredadores')
ax2.set_title('Generalismo vs. Presión de depredación\n(color = nivel trófico)', fontweight='bold')
plt.colorbar(sc, ax=ax2, label='Nivel trófico')
ax2.grid(alpha=0.3)

plt.tight_layout(); plt.savefig('fig2_centralidades.png', bbox_inches='tight'); plt.show()
"""))

# ─── SECCIÓN 3: MODELO DE NICHO ──────────────────────────────────────────────
cells.append(md(r"""
---
## 3. El Modelo de Nicho y Redes Tróficas Realistas

### 3.1 Limitaciones de las redes aleatorias

El modelo de Erdős–Rényi coloca aristas entre pares de nodos con probabilidad $p$
independiente. Aunque útil como referencia nula, falla en reproducir propiedades
universales de redes tróficas reales:

- Distribución de grado aproximadamente exponencial (no de ley de potencias ni Normal).
- Alta fracción de especies basales (sin presas) y tope (sin depredadores).
- Modularidad significativa.
- Correlación entre conectividad y posición trófica.

### 3.2 El Modelo de Nicho (Williams & Martinez, 2000)

Williams y Martinez propusieron un modelo mecanístico mínimo que reproduce con
precisión estadística las redes tróficas empíricas como **Little Rock Lake** y
**Ythan Estuary**. El algoritmo es:

1. **Asignar un valor de nicho** $n_i \sim U(0, 1)$ a cada especie $i$, y ordenar:
   $n_1 \leq n_2 \leq \cdots \leq n_S$.

2. **Definir un rango de consumo.** Para la especie $i$ se sortea un radio
   $r_i \sim \mathrm{Beta}(1, \beta) \cdot n_i$ donde $\beta = \frac{1}{2C} - 1$ y
   $C$ es la conectancia objetivo. Nótese que $r_i \leq n_i$, lo que impide consumir
   "hacia arriba" indiscriminadamente.

3. **Centrar el rango.** El centro del intervalo de consumo es
   $c_i \sim U(r_i/2,\, n_i)$.

4. **Definir interacciones.** La especie $i$ consume a $j$ si $n_j \in [c_i - r_i/2,\, c_i + r_i/2]$.

La especie con el menor valor de nicho $n_1$ nunca consume a nadie (es el productor
primario). El parámetro $\beta$ controla la densidad de aristas: $\beta$ grande
$\Rightarrow$ rangos pequeños $\Rightarrow$ baja conectancia.

**¿Por qué funciona?** El modelo captura la observación empírica de que las interacciones
tróficas tienden a ser **contiguas** en el espacio de nicho (un depredador consume presas
de tamaño o hábitat similares, no cualquier especie aleatoria).
"""))

cells.append(code(r"""
# ── Implementación del modelo de nicho ───────────────────────────────────

def modelo_nicho(S, C, rng):
    '''
    Red trófica con el modelo de nicho de Williams & Martinez (2000).

    Parámetros
    ----------
    S   : número de especies
    C   : conectancia objetivo (m/S^2)
    rng : generador aleatorio

    Retorna
    -------
    G : DiGraph con aristas presa→depredador
    '''
    beta   = 1.0 / (2 * C) - 1.0
    nicho  = np.sort(rng.uniform(0, 1, S))
    radio  = nicho * rng.beta(1, beta, S)
    centro = rng.uniform(radio / 2, nicho, S)

    G = nx.DiGraph()
    G.add_nodes_from(range(S))

    for i in range(S):
        for j in range(S):
            # j es presa de i si n_j cae dentro del intervalo de consumo de i
            if i != j and abs(nicho[j] - centro[i]) <= radio[i] / 2:
                G.add_edge(j, i)  # j → i: j es presa de i

    for i in range(S):
        G.nodes[i].update({'nicho': nicho[i], 'radio': radio[i], 'centro': centro[i]})

    return G

# Red de referencia: 30 especies, C≈0.15
G_lrl = modelo_nicho(S=30, C=0.15, rng=RNG)
mg_lrl = metricas_globales(G_lrl)

print("=== Red de nicho (30 sp., C=0.15) ===")
for k, v in mg_lrl.items():
    print(f"  {k:<20} = {v:.4f}" if isinstance(v, float) else f"  {k:<20} = {v}")
print(f"\n  Conectancia real: {G_lrl.number_of_edges()/30**2:.4f}")
"""))

cells.append(code(r"""
# ── Visualización: espacio de nicho + distribución de grados ─────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Espacio de nicho
ax = axes[0]
nichos  = [G_lrl.nodes[n]['nicho']  for n in sorted(G_lrl.nodes())]
radios  = [G_lrl.nodes[n]['radio']  for n in sorted(G_lrl.nodes())]
centros = [G_lrl.nodes[n]['centro'] for n in sorted(G_lrl.nodes())]
y_pos   = range(len(nichos))
ax.barh(y_pos, radios, left=[c - r/2 for c, r in zip(centros, radios)],
        height=0.7, color='steelblue', alpha=0.5, label='Rango de consumo')
ax.scatter(nichos, y_pos, c='tomato', zorder=5, s=50, label='Valor de nicho')
ax.set_xlabel('Valor de nicho'); ax.set_ylabel('Especie (ordenada por nicho)')
ax.set_title('Espacio de nicho\nRangos de consumo (modelo de nicho)', fontweight='bold')
ax.legend(fontsize=9)

# Distribuciones de grado
for ax, data, titulo, col in zip(
    axes[1:],
    [[G_lrl.in_degree(n) for n in G_lrl.nodes()],
     [G_lrl.out_degree(n) for n in G_lrl.nodes()]],
    ['In-degree $k^{in}$ (N° presas)', 'Out-degree $k^{out}$ (N° depredadores)'],
    ['seagreen', 'tomato']
):
    ax.hist(data, bins=range(0, max(data)+2), color=col, edgecolor='white', rwidth=0.8)
    ax.axvline(np.mean(data), ls='--', color='black', label=f'$\\bar k$={np.mean(data):.1f}')
    ax.set_xlabel(titulo); ax.set_ylabel('Frecuencia')
    ax.set_title(titulo, fontweight='bold'); ax.legend()

plt.suptitle('Red de nicho — 30 especies, C≈0.15', fontweight='bold', y=1.02)
plt.tight_layout(); plt.savefig('fig3_modelo_nicho.png', bbox_inches='tight'); plt.show()
"""))

# ─── SECCIÓN 4: ESTABILIDAD ───────────────────────────────────────────────────
cells.append(md(r"""
---
## 4. Estabilidad Dinámica y el Problema Complejidad-Estabilidad

### 4.1 Linealización de la dinámica poblacional

Sea $\mathbf{N}(t) = (N_1(t), \ldots, N_n(t))^\top$ el vector de abundancias de las $n$
especies, con dinámica

$$\dot{\mathbf{N}} = \mathbf{f}(\mathbf{N}).$$

Sea $\mathbf{N}^*$ un **punto de equilibrio**: $\mathbf{f}(\mathbf{N}^*) = \mathbf{0}$.
Para estudiar la estabilidad, se define la perturbación $\boldsymbol{\delta}(t) = \mathbf{N}(t) - \mathbf{N}^*$
y se linealiza:

$$\dot{\boldsymbol{\delta}} = J\,\boldsymbol{\delta} + O(\|\boldsymbol{\delta}\|^2), \quad J_{ij} = \left.\frac{\partial f_i}{\partial N_j}\right|_{\mathbf{N}^*}$$

donde $J$ es la **Matriz Jacobiana** evaluada en el equilibrio (también llamada
**Matriz Comunitaria** en ecología). La solución lineal es:

$$\boldsymbol{\delta}(t) = \sum_{k=1}^n c_k\,\mathbf{v}_k\,e^{\lambda_k t}$$

donde $\{\lambda_k\}$ y $\{\mathbf{v}_k\}$ son los autovalores y autovectores de $J$.

**Teorema de Lyapunov (versión lineal):** el equilibrio $\mathbf{N}^*$ es asintóticamente
estable si y sólo si

$$\mathrm{Re}(\lambda_k) < 0 \quad \forall\, k = 1, \ldots, n.$$

El **autovalor dominante** $\lambda_1 = \max_k \mathrm{Re}(\lambda_k)$ determina el
comportamiento asintótico:

| $\lambda_1$ | Comportamiento |
|---|---|
| $< 0$ | Perturbaciones decaen exponencialmente. El ecosistema es resiliente. |
| $= 0$ | Bifurcación. El equilibrio es marginalmente estable. |
| $> 0$ | Perturbaciones crecen. El equilibrio colapsa. |

El **tiempo de recuperación** tras una perturbación pequeña es:

$$\tau = -\frac{1}{\lambda_1} \quad (\lambda_1 < 0)$$

Cuando $\lambda_1 \to 0^-$ (el sistema se acerca al umbral de estabilidad), $\tau \to \infty$:
el ecosistema tarda cada vez más en recuperarse. Este fenómeno se llama
**ralentización crítica** (*critical slowing down*) y es un **indicador de alerta
temprana** de colapso ecosistémico.

### 4.2 Construcción de la Matriz Comunitaria

Para una red trófica con interacciones del tipo presa-depredador:

$$J_{ij} = \begin{cases}
-d_i & i = j \quad \text{(auto-regulación, } d_i > 0\text{)}\\
+\alpha_{ij} & j \to i \text{ (depredador } i \text{ se beneficia de presa } j, \; \alpha_{ij}>0)\\
-\beta_{ij} & i \to j \text{ (presa } i \text{ es consumida por } j, \; \beta_{ij}>0)\\
0 & \text{sin interacción directa}
\end{cases}$$

Nótese que los pares de aristas $(j \to i, i \to j^{-1})$ producen entradas de signo
opuesto: $J_{ij} > 0$ y $J_{ji} < 0$, la asimetría característica de la interacción
presa-depredador.

### 4.3 El Criterio de May (1972)

Robert May estudió la estabilidad de **matrices comunitarias aleatorias** donde:

- La diagonal es $J_{ii} = -d$ (auto-regulación uniforme).
- Cada entrada fuera de la diagonal $J_{ij}$ ($i \neq j$) es cero con probabilidad
  $1-C$ e independiente $\sim \mathcal{N}(0, \sigma^2)$ con probabilidad $C$.

Usando el **Teorema del Círculo de Wigner** (teoría de matrices aleatorias), para $n \to \infty$
los autovalores de la parte aleatoria se distribuyen uniformemente dentro de un disco de
radio $\sigma\sqrt{nC}$ en el plano complejo. El sistema es estable si y sólo si este
disco no cruza el eje imaginario:

$$\boxed{\sigma\sqrt{nC} < d}$$

**Implicación ecológica:** un ecosistema aleatorio se vuelve inestable si:

- Aumenta el número de especies $n$ (más diversidad),
- Aumenta la conectancia $C$ (más densidad de interacciones), o
- Aumenta la fuerza de interacción $\sigma$ (más intensidad).

Esto **contradice** la hipótesis intuitiva de que "la diversidad genera estabilidad" y
desencadenó el llamado *debate complejidad-estabilidad*. La resolución posterior (Allesina
& Tang, 2012) mostró que la **estructura** de las interacciones (quién depredador de quién,
con qué asimetría) importa tanto como los parámetros agregados.

> **Nota física:** el criterio de May es análogo al criterio de Routh-Hurwitz
> aplicado a un sistema lineal con aleatoriedad en sus coeficientes. El autovalor
> dominante actúa como el "polo" más lento del sistema.
"""))

cells.append(code(r"""
# ── Construcción de la matriz comunitaria ─────────────────────────────────

def matriz_comunitaria(G, d=1.0, alpha=0.5, beta=0.3, rng=None, aleatorio=False):
    '''
    Construye la matriz comunitaria J para la red G.

    Diagonal: -d (auto-regulación)
    Para cada arista j→i (presa j, depredador i):
        J[i,j] = +alpha  (depredador se beneficia)
        J[j,i] = -beta   (presa sufre)
    Si aleatorio=True, alpha y beta se sortean de |N(0, sigma)|.
    '''
    nodos = sorted(G.nodes())
    n = len(nodos)
    idx = {v: i for i, v in enumerate(nodos)}
    J = -d * np.eye(n)
    for (j, i) in G.edges():
        ii, jj = idx[i], idx[j]
        a = abs(rng.normal(0, alpha)) if aleatorio and rng else alpha
        b = abs(rng.normal(0, beta))  if aleatorio and rng else beta
        J[ii, jj] =  a   # depredador i se beneficia de presa j
        J[jj, ii] = -b   # presa j sufre
    return J

def analizar_estabilidad(J):
    '''Autovalores, autovalor dominante, estabilidad y tiempo de recuperación.'''
    evs = np.linalg.eigvals(J)
    lmax = np.real(evs).max()
    return {
        'autovalores': evs,
        'lambda_max': lmax,
        'estable': lmax < 0,
        'tau': -1.0/lmax if lmax < 0 else np.inf
    }

# Análisis en la red mínima y en la red de nicho
for G, nombre in [(G_mini, 'Red mínima (5 sp.)'), (G_lrl, 'Red de nicho (30 sp.)')]:
    J = matriz_comunitaria(G, d=1.0, alpha=0.5, beta=0.3)
    res = analizar_estabilidad(J)
    print(f"\n{nombre}")
    print(f"  λ_max  = {res['lambda_max']:+.4f}")
    print(f"  Estable: {res['estable']}")
    if res['estable']:
        print(f"  τ (recuperación) = {res['tau']:.2f} u.t.")
    else:
        print(f"  ⚠ INESTABLE: perturbaciones crecen exponencialmente")
"""))

cells.append(code(r"""
# ── Visualización espectral ───────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

def plot_espectro(ax, G, titulo, **kw):
    J = matriz_comunitaria(G, **kw)
    evs = np.linalg.eigvals(J)
    re, im = np.real(evs), np.imag(evs)
    lmax = re.max()
    ax.axvline(0, color='red', ls='--', lw=1.8, label='Umbral Re=0')
    ax.scatter(re, im, c='steelblue', s=80, edgecolors='k', zorder=5, alpha=0.8)
    ax.scatter(lmax, im[re.argmax()], c='crimson', s=250, marker='*', zorder=6,
               label=f'$\\lambda_{{max}}={lmax:.3f}$')
    estab = 'ESTABLE' if lmax < 0 else 'INESTABLE'
    ax.text(0.05, 0.95, estab, transform=ax.transAxes, fontsize=13, fontweight='bold',
            color='seagreen' if lmax < 0 else 'crimson', va='top')
    ax.set_xlabel('Re($\\lambda$)'); ax.set_ylabel('Im($\\lambda$)')
    ax.set_title(titulo, fontweight='bold'); ax.legend(fontsize=9); ax.grid(alpha=0.3)

plot_espectro(axes[0], G_mini, 'Espectro — Red mínima (5 sp.)', d=1.0, alpha=0.5, beta=0.3)
plot_espectro(axes[1], G_lrl,  'Espectro — Red de nicho (30 sp.)', d=1.0, alpha=0.5, beta=0.3)

plt.tight_layout(); plt.savefig('fig4_espectro.png', bbox_inches='tight'); plt.show()
"""))

cells.append(code(r"""
# ── Criterio de May: probabilidad de estabilidad vs sigma y n ─────────────
# Para cada combinación (n, sigma) generamos matrices aleatorias y
# verificamos qué fracción cumple sigma*sqrt(nC) < d

def p_estable_may(n, C, sigma, d=1.0, N=80, rng=None):
    '''Fracción de redes aleatorias estables con los parámetros dados.'''
    rng = rng or np.random.default_rng()
    count = 0
    for _ in range(N):
        M = rng.normal(0, sigma, (n, n))
        M *= rng.uniform(0, 1, (n, n)) < C
        np.fill_diagonal(M, -d)
        if np.real(np.linalg.eigvals(M)).max() < 0:
            count += 1
    return count / N

sigmas = np.linspace(0.05, 0.9, 22)
tamaños = [10, 20, 40]
C0, d0 = 0.15, 1.0

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Izquierda: P(estable) vs sigma para distintos n
ax = axes[0]
for n in tamaños:
    ps = [p_estable_may(n, C0, s, d0, N=70, rng=RNG) for s in sigmas]
    umbral = d0 / np.sqrt(n * C0)
    ax.plot(sigmas, ps, lw=2, label=f'n={n}  (umbral σ*={umbral:.2f})')
    ax.axvline(umbral, ls=':', alpha=0.5)
ax.set_xlabel('Fuerza de interacción σ')
ax.set_ylabel('P(estable)')
ax.set_title(f'Criterio de May\n$\\sigma\\sqrt{{nC}} < d$  (C={C0}, d={d0})', fontweight='bold')
ax.legend(); ax.grid(alpha=0.3)

# Derecha: mapa de calor P(estable) en el plano n-C
sigma0 = 0.3
ns_grid = [5, 10, 20, 30, 50]
Cs_grid = [0.05, 0.10, 0.15, 0.20, 0.30]
Z = np.array([[p_estable_may(n, c, sigma0, d0, N=60, rng=RNG)
               for c in Cs_grid] for n in ns_grid])

ax2 = axes[1]
im = ax2.imshow(Z, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')
ax2.set_xticks(range(len(Cs_grid))); ax2.set_xticklabels([f'{c:.2f}' for c in Cs_grid])
ax2.set_yticks(range(len(ns_grid))); ax2.set_yticklabels(ns_grid)
ax2.set_xlabel('Conectancia C'); ax2.set_ylabel('N° de especies n')
ax2.set_title(f'P(estable) en plano n-C\n(σ={sigma0}, d={d0})', fontweight='bold')
plt.colorbar(im, ax=ax2, label='P(estable)')
for i in range(len(ns_grid)):
    for j in range(len(Cs_grid)):
        ax2.text(j, i, f'{Z[i,j]:.2f}', ha='center', va='center', fontsize=9,
                 color='white' if Z[i,j] < 0.4 else 'black')

plt.tight_layout(); plt.savefig('fig5_criterio_may.png', bbox_inches='tight'); plt.show()
print("Al aumentar n o C, el ecosistema se vuelve estadísticamente inestable — criterio de May.")
"""))

cells.append(code(r"""
# ── Ralentización crítica: τ = -1/λ_max diverge en la bifurcación ─────────
G_test = modelo_nicho(S=20, C=0.12, rng=RNG)
sigmas_bar = np.linspace(0.1, 1.5, 40)
lambdas, taus, estables = [], [], []
for s in sigmas_bar:
    J = matriz_comunitaria(G_test, d=1.0, alpha=s, beta=s*0.6, rng=RNG, aleatorio=True)
    res = analizar_estabilidad(J)
    lambdas.append(res['lambda_max'])
    taus.append(res['tau'])
    estables.append(res['estable'])

idx_bif = next((i for i, e in enumerate(estables) if not e), len(estables))

fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

ax = axes[0]
col_pts = ['steelblue' if e else 'crimson' for e in estables]
ax.scatter(sigmas_bar, lambdas, c=col_pts, s=55, zorder=5)
ax.plot(sigmas_bar, lambdas, color='gray', alpha=0.4)
ax.axhline(0, color='red', ls='--', lw=1.8, label='Umbral de estabilidad')
ax.axvline(sigmas_bar[idx_bif], color='black', ls=':', label='Bifurcación')
ax.set_ylabel('$\\lambda_{max}$')
ax.set_title('Transición de estabilidad\n(azul=estable, rojo=inestable)', fontweight='bold')
ax.legend(); ax.grid(alpha=0.3)

ax2 = axes[1]
tau_clip = np.clip(taus, 0, 25)
ax2.fill_between(sigmas_bar, 0, tau_clip, where=estables, color='steelblue', alpha=0.3)
ax2.plot(sigmas_bar, tau_clip, color='steelblue', lw=2)
ax2.axvline(sigmas_bar[idx_bif], color='black', ls=':', label='Bifurcación (τ→∞)')
ax2.set_xlabel('Fuerza de interacción σ (presión ambiental)')
ax2.set_ylabel('$\\tau = -1/\\lambda_{max}$')
ax2.set_title('Tiempo de recuperación — Ralentización crítica', fontweight='bold')
ax2.legend(); ax2.grid(alpha=0.3)

plt.tight_layout(); plt.savefig('fig6_critical_slowing.png', bbox_inches='tight'); plt.show()
print("τ → ∞ cuando λ_max → 0⁻: señal de alerta temprana antes del colapso.")
"""))

# ─── SECCIÓN 5: CASCADAS ─────────────────────────────────────────────────────
cells.append(md(r"""
---
## 5. Cascadas de Extinción y Robustez

### 5.1 El mecanismo de las cascadas

Una **cascada de extinción** (*extinction cascade*) es un proceso en el que la pérdida
de una especie provoca extinciones **secundarias** en otras que dependían de ella.
En redes tróficas, hay dos mecanismos principales:

1. **Colapso por falta de recursos:** un consumidor obligado pierde a su única o única
   fuente de energía ($k^{in} \to 0$). Sin presas, su población colapsa.

2. **Colapso por liberación de presa:** la desaparición de un depredador tope puede
   causar sobreabundancia de sus presas, que pueden a su vez extinguir por competencia a
   otras especies del mismo nivel (efecto trófico en cascada hacia abajo).

### 5.2 Protocolo de simulación

**Extinción primaria:** se elimina una especie de la red (puede ser dirigida por centralidad
o aleatoria).

**Propagación secundaria:** se eliminan iterativamente todos los consumidores que han
quedado sin ninguna presa (se vuelven *obligate predators* sin recurso):

$$\text{Extinguir } i \text{ si } k_i^{in} = 0 \text{ y } k_i^{out} > 0$$

Este proceso se repite hasta que la red es estable (ya no hay más extinciones
secundarias). El algoritmo tiene complejidad $O(n^2)$ en el peor caso.

### 5.3 Curva de robustez e índice $R$

La **curva de robustez** grafica la fracción de especies supervivientes $S/S_0$ contra la
fracción acumulada de extinciones primarias $x$. El **índice de robustez** $R$ es el área
bajo esta curva:

$$R = \int_0^1 \frac{S(x)}{S_0}\,dx \approx \sum_k \frac{S_k}{S_0} \Delta x$$

- $R \to 1$: la red es muy resiliente (las cascadas son mínimas).
- $R \to 0.5$: umbral convencional de vulnerabilidad.
- $R \to 0$: la remoción de cualquier especie colapsa el sistema.

### 5.4 Comparación: remoción dirigida vs. aleatoria

- **Dirigida (betweenness o grado):** se elimina primero la especie con mayor
  centralidad. Simula el escenario de pérdida selectiva de las especies más importantes
  (sobreexplotación de depredadores tope, tala de plantas clave).

- **Aleatoria:** el orden de extinción es aleatorio. Simula pérdida de biodiversidad
  no selectiva (eventos estocásticos, clima).

La diferencia entre las curvas mide la **vulnerabilidad a perturbaciones selectivas**:
cuanto mayor la brecha, más depende el ecosistema de sus especies clave.
"""))

cells.append(code(r"""
# ── Simulación de cascadas de extinción ──────────────────────────────────

def propagar_extinciones(G):
    '''
    Elimina iterativamente consumidores que se quedan sin presas.
    Un nodo es 'consumidor huérfano' si k_in=0 pero k_out>0
    (tenía depredadores en la red original pero ya no tiene presas).
    '''
    G = G.copy()
    changed = True
    while changed:
        changed = False
        huerfanos = [n for n in G.nodes()
                     if G.in_degree(n) == 0 and G.out_degree(n) > 0]
        if huerfanos:
            G.remove_nodes_from(huerfanos)
            changed = True
    return G

def cascada_extincion(G0, orden='betweenness', rng=None):
    '''
    Simula cascada eliminando especies en el orden indicado.
    Retorna (fraccion_eliminada, fraccion_superviviente).
    '''
    G = G0.copy()
    S0 = G.number_of_nodes()

    if orden == 'betweenness':
        cent = nx.betweenness_centrality(G)
        seq  = sorted(G.nodes(), key=lambda x: -cent[x])
    elif orden == 'grado':
        seq = sorted(G.nodes(), key=lambda x: -G.degree(x))
    elif orden == 'aleatorio':
        seq = list(G.nodes())
        (rng or np.random).shuffle(seq)
    else:
        raise ValueError(orden)

    fx, fy = [], []
    for nodo in seq:
        if nodo not in G:
            continue
        G.remove_node(nodo)
        G = propagar_extinciones(G)
        fx.append((S0 - G.number_of_nodes()) / S0)
        fy.append(G.number_of_nodes() / S0)

    return fx, fy

# Simular en una red de 40 especies
G_casc = modelo_nicho(S=40, C=0.15, rng=RNG)

estrategias = {
    'Betweenness (dirigida)': cascada_extincion(G_casc, 'betweenness'),
    'Grado (dirigida)':       cascada_extincion(G_casc, 'grado'),
    'Aleatoria':              cascada_extincion(G_casc, 'aleatorio', rng=RNG),
}

fig, ax = plt.subplots(figsize=(9, 5))
for (nombre, (fx, fy)), col in zip(estrategias.items(), ['crimson','darkorange','steelblue']):
    R = np.trapz(fy, fx) if len(fx) > 1 else 0
    ax.plot(fx, fy, color=col, lw=2.5, label=f'{nombre}   R = {R:.3f}')

ax.axhline(0.5, color='gray', ls='--', alpha=0.6, label='Umbral 50% supervivencia')
ax.fill_between([0,1], 0, 0.5, color='red', alpha=0.04)
ax.set_xlabel('Fracción de extinciones primarias')
ax.set_ylabel('Fracción superviviente $S/S_0$')
ax.set_title('Cascadas de extinción — Red de nicho (40 esp.)\n'
             'Área bajo la curva = Índice de robustez $R$', fontweight='bold')
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout(); plt.savefig('fig7_cascadas.png', bbox_inches='tight'); plt.show()
"""))

# ─── SECCIÓN 6: BOW-TIE ──────────────────────────────────────────────────────
cells.append(md(r"""
---
## 6. Estructura Bow-Tie y Componentes Fuertemente Conectados

### 6.1 Componentes Fuertemente Conectados (SCC)

En un dígrafo $G$, un **componente fuertemente conectado** (SCC) es un subconjunto
maximal $S \subseteq V$ tal que para cualquier par $i, j \in S$ existe un camino
dirigido de $i$ a $j$ **y** de $j$ a $i$.

El algoritmo de Kosaraju o Tarjan calcula todos los SCCs en tiempo $O(n + m)$.
En redes tróficas, la existencia de un SCC con más de un nodo indica la presencia de
**ciclos tróficos** (omnívoros que consumen presas de múltiples niveles, creando bucles
de retroalimentación).

### 6.2 Descomposición Bow-Tie

La estructura de **corbatín** (*bow-tie*) de Broder et al. (2000), desarrollada
originalmente para la Web, se aplica naturalmente a redes ecológicas dirigidas.
Se define respecto al **SCC principal** (el más grande):

| Componente | Definición formal | Rol ecológico |
|---|---|---|
| **Núcleo** (SCC) | Mayor componente fuertemente conectado | Omnívoros, bucles de retroalimentación |
| **IN** | Nodos que alcanzan el núcleo pero no son alcanzados desde él | Productores primarios y sus consumidores exclusivos |
| **OUT** | Nodos alcanzados desde el núcleo pero que no lo alcanzan | Depredadores tope especializados |
| **Tentáculos** | Nodos que no pertenecen a ninguno de los anteriores | Especies aisladas o en compartimentos separados |

Formalmente, usando el SCC principal $K$:

$$\text{IN} = \{v \notin K : \exists \text{ camino de } v \text{ a } K\} \setminus \text{OUT}$$
$$\text{OUT} = \{v \notin K : \exists \text{ camino de } K \text{ a } v\} \setminus \text{IN}$$
$$\text{Tentáculos} = V \setminus (K \cup \text{IN} \cup \text{OUT})$$

### 6.3 Implicaciones para la robustez

La estructura bow-tie tiene consecuencias directas para la estabilidad:

- El **núcleo SCC** es el más robusto: tiene ciclos internos que amortiguan
  perturbaciones.
- El componente **IN** es el más vulnerable a sobreexplotación ascendente: si se eliminan
  los productores primarios, toda la red colapsa.
- El componente **OUT** (depredadores tope) es vulnerable a la sobreexplotación
  descendente (pesca excesiva, caza).
- La **disasortatividad** ($r < 0$) observada en redes tróficas implica que los hubs
  del núcleo están mayormente conectados a especies periféricas, lo que **limita la
  propagación** de cascadas desde la periferia hacia el núcleo.
"""))

cells.append(code(r"""
# ── Análisis Bow-Tie ───────────────────────────────────────────────────────

def bow_tie(G):
    '''Descompone G en núcleo SCC, IN, OUT y tentáculos.'''
    sccs = list(nx.strongly_connected_components(G))
    nucleo = max(sccs, key=len)

    alcanza_nucleo = set()
    desde_nucleo   = set()
    for v in nucleo:
        alcanza_nucleo.update(nx.ancestors(G, v))
        desde_nucleo.update(nx.descendants(G, v))

    IN         = alcanza_nucleo - nucleo - desde_nucleo
    OUT        = desde_nucleo   - nucleo - alcanza_nucleo
    tentaculos = set(G.nodes()) - nucleo - IN - OUT
    return {'nucleo': nucleo, 'IN': IN, 'OUT': OUT, 'tentaculos': tentaculos}

G_bt = modelo_nicho(S=30, C=0.18, rng=RNG)
bt   = bow_tie(G_bt)
S0   = G_bt.number_of_nodes()

print("=== Estructura Bow-Tie ===")
for comp, nodos in bt.items():
    print(f"  {comp:<12}: {len(nodos):>3} nodos  ({len(nodos)/S0*100:.1f}%)")

color_map = {n: ('crimson' if n in bt['nucleo'] else
                 'seagreen' if n in bt['IN'] else
                 'steelblue' if n in bt['OUT'] else 'gold')
             for n in G_bt.nodes()}

fig, ax = plt.subplots(figsize=(10, 7))
pos = nx.spring_layout(G_bt, seed=7, k=1.1)
nx.draw_networkx(G_bt, pos=pos, ax=ax,
                 node_color=[color_map[n] for n in G_bt.nodes()],
                 node_size=400, with_labels=False,
                 edge_color='gray', arrows=True, arrowsize=10, alpha=0.85, width=0.7)

from matplotlib.patches import Patch
leyenda = [
    Patch(color='crimson',   label=f'Núcleo SCC ({len(bt["nucleo"])} sp.) — ciclos internos'),
    Patch(color='seagreen',  label=f'IN  ({len(bt["IN"])} sp.) — productores primarios'),
    Patch(color='steelblue', label=f'OUT ({len(bt["OUT"])} sp.) — depredadores tope'),
    Patch(color='gold',      label=f'Tentáculos ({len(bt["tentaculos"])} sp.)'),
]
ax.legend(handles=leyenda, loc='upper left', framealpha=0.95, fontsize=9)
ax.set_title('Estructura Bow-Tie de la Red Trófica\n'
             '(flecha j→i: j es presa de i)', fontweight='bold', fontsize=12)
ax.axis('off')
plt.tight_layout(); plt.savefig('fig8_bowtie.png', bbox_inches='tight'); plt.show()
"""))

# ─── SECCIÓN 7: GLV ──────────────────────────────────────────────────────────
cells.append(md(r"""
---
## 7. Dinámica de Poblaciones: Modelo de Lotka-Volterra Generalizado (GLV)

### 7.1 Formulación del modelo

El modelo **Lotka-Volterra Generalizado** (GLV) es la extensión canónica de la ecuación
logística a comunidades de $n$ especies con interacciones mutuas:

$$\frac{dN_i}{dt} = r_i N_i \left(1 + \sum_{j=1}^n \alpha_{ij} N_j\right), \quad i = 1, \ldots, n$$

donde:
- $N_i(t) \geq 0$: abundancia de la especie $i$.
- $r_i > 0$: tasa de crecimiento intrínseca.
- $\alpha_{ij}$: coeficiente de interacción entre $i$ y $j$:
  - $\alpha_{ii} < 0$: auto-limitación por capacidad de carga ($K_i = -1/\alpha_{ii}$).
  - $\alpha_{ij} > 0$ y $\alpha_{ji} < 0$: $i$ depredador de $j$.
  - $\alpha_{ij} > 0$ y $\alpha_{ji} > 0$: mutualismo.
  - $\alpha_{ij} < 0$ y $\alpha_{ji} < 0$: competencia.

### 7.2 Conexión con la estabilidad lineal

El equilibrio $\mathbf{N}^*$ satisface $r_i N_i^*(1 + \sum_j \alpha_{ij} N_j^*) = 0$.
La Jacobiana en ese punto tiene entradas:

$$J_{ij}\big|_{\mathbf{N}^*} = r_i \alpha_{ij} N_i^*, \quad i \neq j; \qquad J_{ii} = r_i \left(1 + \sum_j \alpha_{ij}N_j^*\right) + r_i\alpha_{ii}N_i^*$$

En el equilibrio interior ($N_i^* > 0$), el primer término se anula y
$J_{ii} = r_i \alpha_{ii} N_i^* < 0$, que es la auto-regulación del criterio de May.

### 7.3 Integración numérica

El sistema GLV es un sistema de EDOs no lineales. Se integra con un método
de Runge-Kutta de orden 4-5 adaptativo (`scipy.integrate.solve_ivp` con `method='RK45'`).

Las condiciones de estabilidad numérica requieren:
- Forzar $N_i \geq 0$ (las abundancias no pueden ser negativas — extinción).
- Usar tolerancias relativas y absolutas ajustadas al rango de valores esperados.
"""))

cells.append(code(r"""
# ── Implementación del GLV ────────────────────────────────────────────────

def construir_alpha(G, alpha_presa=0.1, alpha_dep=-0.05, K_inv=-0.1):
    '''
    Construye la matriz de interacciones α a partir de la topología de G.
    Arista j→i: α[i,j] = +alpha_presa, α[j,i] = alpha_dep
    Diagonal: α[i,i] = K_inv (= -1/K_i)
    '''
    nodos = sorted(G.nodes())
    n = len(nodos)
    idx = {v: i for i, v in enumerate(nodos)}
    Alpha = np.zeros((n, n))
    np.fill_diagonal(Alpha, K_inv)
    for (j, i) in G.edges():
        ii, jj = idx[i], idx[j]
        Alpha[ii, jj] =  alpha_presa  # depredador i se beneficia
        Alpha[jj, ii] =  alpha_dep    # presa j sufre
    return Alpha

def simular_glv(G, t_max=60, dt=0.1, r=1.0,
                alpha_presa=0.1, alpha_dep=-0.05, K_inv=-0.1,
                N0=None, rng=None):
    '''Integra el GLV en la red G con RK45.'''
    nodos = sorted(G.nodes())
    n = len(nodos)
    Alpha = construir_alpha(G, alpha_presa, alpha_dep, K_inv)
    if N0 is None:
        N0 = (rng or np.random).uniform(0.5, 2.0, n) if rng else np.ones(n)
    r_vec = r * np.ones(n)

    def glv(t, N):
        N = np.clip(N, 0, None)
        return r_vec * N * (1.0 + Alpha @ N)

    sol = solve_ivp(glv, (0, t_max), N0, method='RK45',
                    t_eval=np.arange(0, t_max + dt, dt),
                    max_step=dt, rtol=1e-6, atol=1e-9)
    return sol.t, sol.y.T  # (n_t, n_especies)


# Simulación en la red mínima
t, Nt = simular_glv(G_mini, t_max=70, r=0.8, alpha_presa=0.15,
                     alpha_dep=-0.08, K_inv=-0.1, rng=RNG)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
ax = axes[0]
cols = plt.cm.tab10(np.linspace(0, 1, 5))
for i, n_sp in enumerate(sorted(G_mini.nodes())):
    ax.plot(t, Nt[:, i], label=especies[n_sp], color=cols[i], lw=2)
ax.set_xlabel('Tiempo'); ax.set_ylabel('Abundancia $N_i(t)$')
ax.set_title('Dinámica GLV — Red mínima (5 sp.)', fontweight='bold')
ax.legend(); ax.grid(alpha=0.3)

ax2 = axes[1]
N_eq = Nt[-1, :]
nombres = [especies[n] for n in sorted(G_mini.nodes())]
bars = ax2.bar(nombres, N_eq, color=cols, edgecolor='k')
for bar, val in zip(bars, N_eq):
    ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.02,
             f'{val:.2f}', ha='center', fontsize=10)
ax2.set_ylabel('Abundancia en equilibrio $N_i^*$')
ax2.set_title('Equilibrio del sistema GLV', fontweight='bold')
ax2.tick_params(axis='x', rotation=30); ax2.grid(axis='y', alpha=0.3)

plt.tight_layout(); plt.savefig('fig9_glv_dinamica.png', bbox_inches='tight'); plt.show()
"""))

cells.append(code(r"""
# ── Perturbación impulsiva y recuperación ─────────────────────────────────
# Perturbación a t=25: reducción brusca del 70% de todas las abundancias

def simular_perturbacion(G, t_perturb=25, factor=0.3, t_max=80, **kw):
    t1, N1 = simular_glv(G, t_max=t_perturb, **kw)
    N_pert = N1[-1, :] * factor
    kw2 = {k: v for k, v in kw.items() if k != 'rng'}
    t2, N2 = simular_glv(G, t_max=t_max-t_perturb, N0=N_pert, **kw2)
    return np.concatenate([t1, t2+t_perturb]), np.vstack([N1, N2]), t_perturb

t_tot, N_tot, t_p = simular_perturbacion(
    G_mini, t_perturb=25, factor=0.3, t_max=90,
    r=0.8, alpha_presa=0.15, alpha_dep=-0.08, K_inv=-0.1, rng=RNG)

fig, ax = plt.subplots(figsize=(11, 5))
for i, n_sp in enumerate(sorted(G_mini.nodes())):
    ax.plot(t_tot, N_tot[:, i], label=especies[n_sp], color=cols[i], lw=2)
ax.axvline(t_p, color='black', ls='--', lw=2, label=f'Perturbación (t={t_p})')
ax.fill_betweenx([0, N_tot.max()*1.1], t_p, t_p+3, color='red', alpha=0.12)
ax.set_xlabel('Tiempo'); ax.set_ylabel('$N_i(t)$')
ax.set_title('Perturbación impulsiva (−70%) y recuperación — GLV', fontweight='bold')
ax.legend(loc='upper right'); ax.grid(alpha=0.3)
plt.tight_layout(); plt.savefig('fig10_perturbacion_glv.png', bbox_inches='tight'); plt.show()
"""))

# ─── SECCIÓN 8: DASHBOARD ────────────────────────────────────────────────────
cells.append(md(r"""
---
## 8. Dashboard Integrado: Análisis Completo de una Red de 50 Especies

Esta sección integra todas las herramientas desarrolladas en un pipeline de análisis
aplicado a una red de nicho de 50 especies, produciendo un informe estructurado y un
panel visual de síntesis.
"""))

cells.append(code(r"""
# ── Pipeline de análisis completo ─────────────────────────────────────────

def informe_completo(G, nombre='Red'):
    n_sp = G.number_of_nodes()
    print(f"\n{'='*55}\n INFORME: {nombre}\n{'='*55}")

    mg = metricas_globales(G)
    print("\n--- Métricas Globales ---")
    for k, v in mg.items():
        print(f"  {k:<22} = {v:.4f}" if isinstance(v, float) else f"  {k:<22} = {v}")

    bt_res = bow_tie(G)
    print("\n--- Estructura Bow-Tie ---")
    for comp, nds in bt_res.items():
        print(f"  {comp:<14}: {len(nds):>3} ({len(nds)/n_sp*100:.1f}%)")

    J = matriz_comunitaria(G, d=1.0, alpha=0.4, beta=0.25)
    res = analizar_estabilidad(J)
    print("\n--- Estabilidad Dinámica ---")
    print(f"  λ_max  = {res['lambda_max']:+.4f}")
    print(f"  Estable: {res['estable']}")
    if res['estable']:
        print(f"  τ ≈ {res['tau']:.2f} u.t.")

    bet = nx.betweenness_centrality(G)
    niv = nivel_trofico(G)
    print("\n--- Top 5 Especies Puente (betweenness) ---")
    for nd, val in sorted(bet.items(), key=lambda x:-x[1])[:5]:
        print(f"  Nodo {nd:3d}: B={val:.4f}  k_in={G.in_degree(nd)}  "
              f"k_out={G.out_degree(nd)}  nivel={niv[nd]:.2f}")
    return mg, bt_res, res

G_main = modelo_nicho(S=50, C=0.12, rng=RNG)
mg_m, bt_m, res_m = informe_completo(G_main, "Red de Nicho — 50 especies")
"""))

cells.append(code(r"""
# ── Dashboard visual ──────────────────────────────────────────────────────
from matplotlib.patches import Patch

fig = plt.figure(figsize=(16, 12))
gs  = fig.add_gridspec(2, 3, hspace=0.42, wspace=0.35)

# A) Grafo coloreado por nivel trófico
ax_a = fig.add_subplot(gs[0, :2])
niv_m = nivel_trofico(G_main)
vals_m = np.array([niv_m[n] for n in sorted(G_main.nodes())])
nc_m   = plt.cm.YlOrRd((vals_m - 1) / (vals_m.max() - 1 + 1e-9))
pos_m  = nx.spring_layout(G_main, seed=2, k=0.9)
nx.draw_networkx(G_main, pos=pos_m, ax=ax_a,
                 node_color=list(nc_m), node_size=180,
                 with_labels=False, edge_color='gray',
                 arrows=True, arrowsize=7, alpha=0.85, width=0.4)
sm = plt.cm.ScalarMappable(cmap='YlOrRd', norm=mcolors.Normalize(1, vals_m.max()))
plt.colorbar(sm, ax=ax_a, label='Nivel trófico')
ax_a.set_title('Red trófica — 50 especies  (color = nivel trófico)', fontweight='bold')
ax_a.axis('off')

# B) Espectro de la matriz comunitaria
ax_b = fig.add_subplot(gs[0, 2])
J_m  = matriz_comunitaria(G_main, d=1.0, alpha=0.4, beta=0.25)
evs_m = np.linalg.eigvals(J_m)
re_m, im_m = np.real(evs_m), np.imag(evs_m)
lmax_m = re_m.max()
ax_b.axvline(0, color='red', ls='--', lw=1.5)
ax_b.scatter(re_m, im_m, c='steelblue', s=40, edgecolors='k', alpha=0.8)
ax_b.scatter(lmax_m, im_m[re_m.argmax()], c='crimson', s=200, marker='*', zorder=6,
             label=f'$\\lambda_{{max}}={lmax_m:.3f}$')
ax_b.set_xlabel('Re(λ)'); ax_b.set_ylabel('Im(λ)')
ax_b.set_title('Espectro de $J$', fontweight='bold'); ax_b.legend(fontsize=9); ax_b.grid(alpha=0.3)

# C) Distribución de grado
ax_c = fig.add_subplot(gs[1, 0])
gds = [G_main.degree(n) for n in G_main.nodes()]
ax_c.hist(gds, bins=15, color='seagreen', edgecolor='white', rwidth=0.85)
ax_c.set_xlabel('Grado total'); ax_c.set_ylabel('Frecuencia')
ax_c.set_title('Distribución de grado', fontweight='bold'); ax_c.grid(alpha=0.3)

# D) Betweenness (top 20)
ax_d = fig.add_subplot(gs[1, 1])
bvals = sorted(nx.betweenness_centrality(G_main).values(), reverse=True)
ax_d.bar(range(len(bvals)), bvals, color='darkorange', edgecolor='white')
ax_d.set_xlabel('Especie (rank)'); ax_d.set_ylabel('Betweenness')
ax_d.set_title('Centralidad de intermediación', fontweight='bold'); ax_d.grid(axis='y', alpha=0.3)

# E) Bow-Tie
ax_e = fig.add_subplot(gs[1, 2])
bt_m2 = bow_tie(G_main)
sizes_bt = [len(bt_m2[k]) for k in ['nucleo','IN','OUT','tentaculos']]
lbls_bt  = ['Núcleo SCC','IN (productores)','OUT (dep. tope)','Tentáculos']
cols_bt  = ['crimson','seagreen','steelblue','gold']
ax_e.pie(sizes_bt, labels=lbls_bt, colors=cols_bt, autopct='%1.1f%%',
         startangle=90, wedgeprops={'edgecolor':'white','linewidth':1.5})
ax_e.set_title('Estructura Bow-Tie', fontweight='bold')

plt.suptitle('Dashboard — Red Ecológica (50 especies, C≈0.12)',
             fontsize=14, fontweight='bold', y=1.01)
plt.savefig('fig11_dashboard.png', bbox_inches='tight', dpi=130); plt.show()
"""))

# ─── CONCLUSIONES ─────────────────────────────────────────────────────────────
cells.append(md(r"""
---
## 9. Síntesis y Conclusiones

### Tabla resumen de conceptos

| Concepto | Expresión matemática | Interpretación ecológica |
|---|---|---|
| Matriz de adyacencia | $A_{ij}=1$ si $j\to i$ | Quién come a quién |
| Conectancia | $C = m/n^2$ | Densidad de interacciones |
| Nivel trófico | $\mathbf{x} = (D-A)^{-1}D\mathbf{1}$ | Posición en la cadena alimentaria |
| Betweenness | $B_i = \sum_{s,t}\sigma_{st}(i)/\sigma_{st}$ | Especie puente (vulnerabilidad) |
| Modularidad | $Q = \frac{1}{2m}\sum_{ij}[A_{ij}-\frac{k_ik_j}{2m}]\delta(c_i,c_j)$ | Compartimentación |
| Asortatividad | $r = $ corr. de Pearson de grados | Arquitectura hub-satélite |
| Autovalor dominante | $\lambda_1 = \max_k \mathrm{Re}(\lambda_k)$ | Estabilidad del equilibrio |
| Tiempo de relajación | $\tau = -1/\lambda_1$ | Velocidad de recuperación |
| Criterio de May | $\sigma\sqrt{nC} < d$ | Umbral complejidad-estabilidad |
| GLV | $\dot{N}_i = r_i N_i(1+\sum_j\alpha_{ij}N_j)$ | Dinámica de abundancias |

### Hallazgos principales

1. **El criterio de May es cuantitativo:** a medida que $n$, $C$ o $\sigma$ aumentan,
   la probabilidad de estabilidad cae abruptamente en torno al umbral $\sigma\sqrt{nC}=d$.
   Este umbral es una predicción exacta de teoría de matrices aleatorias.

2. **Las redes tróficas son disasortativas ($r<0$):** los generalistas (hubs) se
   conectan a especialistas. Esta arquitectura limita la propagación de cascadas desde
   la periferia hacia el núcleo.

3. **La ralentización crítica es detectable antes del colapso:** el tiempo de relajación
   $\tau \to \infty$ cuando $\lambda_1 \to 0^-$. Esto es un indicador de alerta temprana
   que puede detectarse empíricamente antes de que el ecosistema colapase.

4. **La remoción dirigida (betweenness) es mucho más destructiva que la aleatoria:**
   la brecha entre ambas curvas de robustez cuantifica la vulnerabilidad a pérdida
   selectiva de especies clave.

5. **La estructura Bow-Tie organiza el flujo de energía:** el núcleo SCC amortigua
   perturbaciones internas; los componentes IN y OUT son los puntos de entrada y salida
   del flujo energético y los más susceptibles a perturbaciones externas.

---

### Referencias

- May, R.M. (1972). *Will a large complex system be stable?* Nature, 238, 413–414.
- Williams, R.J. & Martinez, N.D. (2000). *Simple rules yield complex food webs.* Nature, 404, 180–183.
- Dunne, J.A., Williams, R.J. & Martinez, N.D. (2002). *Network structure and biodiversity loss in food webs.* Ecology Letters, 5, 558–567.
- Allesina, S. & Tang, S. (2012). *Stability criteria for complex ecosystems.* Nature, 483, 205–208.
- Newman, M.E.J. (2010). *Networks: An Introduction.* Oxford University Press.
"""))

cells.append(code(r"""
# Figuras generadas
figs = sorted(f for f in os.listdir('.') if f.startswith('fig') and f.endswith('.png'))
print("Figuras generadas:")
for f in figs:
    print(f"  {f}")
"""))

# ─── ESCRIBIR EL NOTEBOOK ────────────────────────────────────────────────────
nb = {
    "nbformat": 4, "nbformat_minor": 5,
    "metadata": {
        "colab": {"name": "redes_ecologicas_parcial2.ipynb"},
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.0"}
    },
    "cells": cells
}

# Añadir IDs únicos a cada celda
for i, cell in enumerate(nb["cells"]):
    cell["id"] = f"cell-{i:04d}"
    if cell["cell_type"] == "markdown":
        cell.pop("execution_count", None)
        cell.pop("outputs", None)

with open("redes_ecologicas_parcial2.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print("Notebook generado: redes_ecologicas_parcial2.ipynb")
print(f"Total de celdas: {len(cells)}")
