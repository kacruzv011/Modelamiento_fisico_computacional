"""
Análisis paso 2: f₀ como función de la longitud vibrante L — ley f₀ ∝ L⁻¹.

Variables utilizadas: únicamente f₀ (medida, FFT) y L (medida, Distancias.txt).
No se usa μ ni T — análisis estrictamente bivariado.

Procedimiento:
  1. Ajuste potencial:   f₀ = a · L^b  (regresión log-log)
  2. Linealización:      log(f₀) = b·log(L) + log(a)  →  Y = bX + A
  3. Outliers por desviación > UMBRAL_OUTLIER respecto al modelo f₀·L = cte

Salidas:
  figuras/fL_cuerda{n}.png     — f₀ vs L + linealización log-log por cuerda
  figuras/fL_normalizado.png   — f₀/f₀(T0) vs L/L₀ (colapso sobre y = 1/x)
  figuras/fL_constante.png     — f₀·L vs traste (constante ⟺ f₀ ∝ 1/L)
  datos_fL.csv                 — (cuerda, traste, f₀_Hz, L_cm, f₀·L, outlier)

Estilo: [SCI]
"""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# ── Configuración ─────────────────────────────────────────────────────────────

FIGURAS_DIR      = Path("figuras")
UMBRAL_OUTLIER   = 0.15    # fracción: |f_med - f_pred| / f_pred > umbral → outlier

# ── Datos de entrada ──────────────────────────────────────────────────────────

# Longitudes vibrantes L (cm) por traste — Distancias.txt
# Cada valor es la longitud de la cuerda desde el puente hasta el traste presionado.
DISTANCIAS_CM = {
    0: 65.2,  1: 63.2,  2: 59.7,  3: 56.3,  4: 53.2,
    5: 50.2,  6: 46.7,  7: 44.7,  8: 42.2,  9: 39.7,
   10: 37.4, 11: 35.4, 12: 33.4, 13: 31.4, 14: 29.7,
   15: 28.1, 16: 26.4, 17: 24.9, 18: 23.5, 19: 22.4,
}

NOMBRE_CUERDA = {
    1: "Mi4 (E4)", 2: "Si3 (B3)", 3: "Sol3 (G3)",
    4: "Re3 (D3)", 5: "La2 (A2)", 6: "Mi2 (E2)",
}
COLORES = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00", "#a65628"]

# ── Carga de datos ────────────────────────────────────────────────────────────

def cargar_f0(csv_path: Path = Path("frecuencias_fundamentales.csv")) -> dict[int, dict[int, float]]:
    tabla: dict[int, dict[int, float]] = {c: {} for c in range(1, 7)}
    with open(csv_path, encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            tabla[int(row["cuerda"])][int(row["traste"])] = float(row["f0_Hz"])
    return tabla

# ── Detección de outliers ─────────────────────────────────────────────────────

def separar_outliers(
    trastes: list[int],
    f0s: list[float],
    L0: float,
    f0_0: float,
    umbral: float = UMBRAL_OUTLIER,
) -> tuple[list[int], list[float], list[int], list[float]]:
    """Separa puntos en (buenos, outliers) usando como referencia f₀·L = constante."""
    buenos_t, buenos_f, malos_t, malos_f = [], [], [], []
    for t, f in zip(trastes, f0s):
        L = DISTANCIAS_CM[t]
        f_pred = f0_0 * L0 / L
        if abs(f - f_pred) / f_pred > umbral:
            malos_t.append(t); malos_f.append(f)
        else:
            buenos_t.append(t); buenos_f.append(f)
    return buenos_t, buenos_f, malos_t, malos_f

# ── Ajuste potencial en escala log ────────────────────────────────────────────

def ajuste_potencial(
    L_arr: np.ndarray, f0_arr: np.ndarray
) -> tuple[float, float, float]:
    """Ajusta f₀ = a · L^b en escala log-log.

    Returns:
        a, b, R²   (con R² calculado en espacio log)
    """
    log_L  = np.log10(L_arr)
    log_f  = np.log10(f0_arr)
    coeffs = np.polyfit(log_L, log_f, 1)
    b, log_a = coeffs
    a = 10 ** log_a
    log_pred = np.polyval(coeffs, log_L)
    ss_res = np.sum((log_f - log_pred) ** 2)
    ss_tot = np.sum((log_f - log_f.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return float(a), float(b), float(r2)

# ── Gráficas por cuerda ───────────────────────────────────────────────────────

def graficar_fL_cuerda(
    cuerda: int,
    buenos_t: list[int], buenos_f: list[float],
    malos_t: list[int],  malos_f: list[float],
    ruta: Path,
) -> tuple[float, float, float]:
    """Genera figura con dos subplots: f₀ vs L y linealización log-log.

    Returns (a, b, R²) del ajuste potencial sobre puntos buenos.
    """
    color = COLORES[cuerda - 1]
    L_b   = np.array([DISTANCIAS_CM[t] for t in buenos_t])
    f0_b  = np.array(buenos_f)

    a, b, r2 = ajuste_potencial(L_b, f0_b)

    L_fino = np.linspace(DISTANCIAS_CM[19] * 0.95, DISTANCIAS_CM[0] * 1.02, 300)
    f_fit  = a * L_fino ** b

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # ── Subplot izquierdo: f₀ vs L ──────────────────────────────────────────
    ax1.plot(L_fino, f_fit, "-", color=color, lw=1.8, alpha=0.7,
             label=f"Ajuste  f₀ = {a:.2f}·L^{{{b:.3f}}}  (R²={r2:.4f})")
    ax1.scatter(L_b, f0_b, color=color, s=45, zorder=5, label="Medido")
    if malos_t:
        L_m = [DISTANCIAS_CM[t] for t in malos_t]
        ax1.scatter(L_m, malos_f, marker="x", color="gray", s=60, zorder=5,
                    linewidths=1.5, label="Outlier")

    ax1.set_xlabel("L  (cm)")
    ax1.set_ylabel("f₀  (Hz)")
    ax1.set_title(f"C{cuerda} — {NOMBRE_CUERDA[cuerda]}  |  f₀ vs L")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # ── Subplot derecho: log-log (linealización) ─────────────────────────────
    log_L_b = np.log10(L_b)
    log_f_b = np.log10(f0_b)
    log_L_fino = np.log10(L_fino)

    ax2.plot(log_L_fino, np.log10(f_fit), "-", color=color, lw=1.8, alpha=0.7,
             label=f"Y = {b:.3f}·X + {np.log10(a):.3f}")
    ax2.scatter(log_L_b, log_f_b, color=color, s=45, zorder=5)
    if malos_t:
        L_m = [DISTANCIAS_CM[t] for t in malos_t]
        ax2.scatter(np.log10(L_m), np.log10(malos_f),
                    marker="x", color="gray", s=60, linewidths=1.5, zorder=5)

    ax2.set_xlabel("log₁₀(L)")
    ax2.set_ylabel("log₁₀(f₀)")
    ax2.set_title(f"Linealización  |  pendiente b = {b:.3f}  (teórico −1)")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    fig.suptitle(
        f"C{cuerda} — {NOMBRE_CUERDA[cuerda]}",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(ruta, dpi=110)
    plt.close(fig)

    return a, b, r2

# ── Gráfica normalizada (colapso) ─────────────────────────────────────────────

def graficar_normalizado(
    tabla: dict[int, dict[int, float]],
    ruta: Path,
) -> None:
    """f₀/f₀(T0) vs L/L₀ para todas las cuerdas.

    Si f₀ ∝ 1/L, todos los puntos deberían caer sobre la curva y = 1/x.
    """
    L0 = DISTANCIAS_CM[0]
    x_teo = np.linspace(0.30, 1.05, 300)
    y_teo = 1.0 / x_teo

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x_teo, y_teo, "k--", lw=1.5, alpha=0.6, label="Teórico  y = 1/x")

    for cuerda in range(1, 7):
        f0_T0 = tabla[cuerda].get(0)
        if f0_T0 is None:
            continue
        trastes = sorted(tabla[cuerda])
        L_norm  = [DISTANCIAS_CM[t] / L0 for t in trastes]
        f_norm  = [tabla[cuerda][t] / f0_T0 for t in trastes]
        ax.scatter(L_norm, f_norm, color=COLORES[cuerda - 1], s=25,
                   label=f"C{cuerda} {NOMBRE_CUERDA[cuerda]}", zorder=4)

    ax.set_xlabel("L / L₀")
    ax.set_ylabel("f₀ / f₀(T0)")
    ax.set_title("Colapso  f₀/f₀(T0) vs L/L₀  —  todas las cuerdas")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(ruta, dpi=110)
    plt.close(fig)

# ── Gráfica f₀·L ≈ constante ─────────────────────────────────────────────────

def graficar_fL_constante(
    tabla: dict[int, dict[int, float]],
    ruta: Path,
) -> None:
    """f₀·L vs traste para cada cuerda.

    Si f₀ ∝ 1/L, el producto f₀·L = (1/2)·√(T/μ) = constante por cuerda.
    Desviaciones revelan outliers o variación de tensión.
    """
    fig, ax = plt.subplots(figsize=(11, 5))

    for cuerda in range(1, 7):
        trastes = sorted(tabla[cuerda])
        fL_vals = [tabla[cuerda][t] * DISTANCIAS_CM[t] for t in trastes]
        ax.plot(trastes, fL_vals, "o-", color=COLORES[cuerda - 1],
                lw=1.2, ms=5, label=f"C{cuerda} {NOMBRE_CUERDA[cuerda]}")

    ax.set_xlabel("Traste")
    ax.set_ylabel("f₀ · L  (Hz·cm)")
    ax.set_title("f₀ · L por cuerda y traste  |  constante ⟺ f₀ ∝ 1/L")
    ax.set_xticks(range(20))
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(ruta, dpi=110)
    plt.close(fig)

# ── Pipeline ──────────────────────────────────────────────────────────────────

def main() -> None:
    FIGURAS_DIR.mkdir(exist_ok=True)
    tabla = cargar_f0()

    resultados: list[dict] = []   # para CSV y resumen
    L0 = DISTANCIAS_CM[0]

    print(f"\n{'─'*62}")
    print(f"{'Cuerda':<22} {'a':>9} {'b':>8} {'R²':>8}  {'b_teo=-1':>9}")
    print(f"{'─'*62}")

    for cuerda in range(1, 7):
        trastes = sorted(tabla[cuerda].keys())
        f0s     = [tabla[cuerda][t] for t in trastes]
        f0_T0   = tabla[cuerda].get(0)

        if f0_T0 is None or len(trastes) < 4:
            print(f"C{cuerda}: datos insuficientes")
            continue

        buenos_t, buenos_f, malos_t, malos_f = separar_outliers(
            trastes, f0s, L0, f0_T0
        )

        a, b, r2 = graficar_fL_cuerda(
            cuerda, buenos_t, buenos_f, malos_t, malos_f,
            ruta=FIGURAS_DIR / f"fL_cuerda{cuerda}.png",
        )

        print(
            f"C{cuerda} {NOMBRE_CUERDA[cuerda]:<16}"
            f"  {a:>9.3f}  {b:>8.4f}  {r2:>8.5f}"
            f"  Δb={b-(-1):>+7.4f}"
        )

        for t, f in zip(trastes, f0s):
            resultados.append({
                "cuerda":    cuerda,
                "traste":    t,
                "f0_Hz":     f,
                "L_cm":      DISTANCIAS_CM[t],
                "f0_L_Hzcm": f * DISTANCIAS_CM[t],
                "outlier":   1 if t in malos_t else 0,
            })

    print(f"{'─'*62}")

    graficar_normalizado(tabla, ruta=FIGURAS_DIR / "fL_normalizado.png")
    graficar_fL_constante(tabla, ruta=FIGURAS_DIR / "fL_constante.png")

    # CSV
    csv_path = Path("datos_fL.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        campos = ["cuerda", "traste", "f0_Hz", "L_cm", "f0_L_Hzcm", "outlier"]
        writer = csv.DictWriter(fh, fieldnames=campos)
        writer.writeheader()
        writer.writerows(resultados)

    print(f"\nFiguras:  {FIGURAS_DIR}/fL_*.png")
    print(f"CSV:      {csv_path}  (columnas: cuerda, traste, f0_Hz, L_cm, f0·L, outlier)")


if __name__ == "__main__":
    main()
