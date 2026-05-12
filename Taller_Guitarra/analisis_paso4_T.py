"""
Análisis paso 4: f₀ como función de la tensión T — ley f₀ ∝ T^(1/2).

Variables utilizadas: únicamente f₀ (medida por FFT) y T (medida, DCL con masas).
No se usa μ ni L — análisis estrictamente bivariado.

El valor de T se obtuvo experimentalmente colgando masas en la cuerda y midiendo
el ángulo de deflexión; el DCL da T = mg / (2·sin θ).  Se midió únicamente la
cuerda 5 (La2 / A2) con dos masas diferentes (220 g y 150 g) a distintos ángulos.

Procedimiento:
  1. Lectura de Datos_de_Tension.csv (columnas T_N y f0_Hz únicamente)
  2. Ajuste potencial:   f₀ = a · T^b  (regresión log-log)
  3. Linealización:      log(f₀) = b·log(T) + log(a)  →  Y = bX + A
  4. Outliers por desviación > UMBRAL_OUTLIER respecto al modelo ajustado (2 pasadas)

Salidas:
  figuras/fT_ajuste.png      — f₀ vs T + linealización log-log
  figuras/fT_normalizado.png — f₀/f₀_ref vs T/T_ref (colapso sobre y = √x)
  datos_fT.csv               — (cuerda, punto, f0_Hz, T_N, outlier)

Estilo: [SCI]
"""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# ── Configuración ─────────────────────────────────────────────────────────────

FIGURAS_DIR    = Path("figuras")
CSV_TENSION    = Path("Datos_de_Tension.csv")
UMBRAL_OUTLIER = 0.10    # fracción: |f_med - f_pred| / f_pred > umbral → outlier
CUERDA         = 5       # La2 / A2
COLOR          = "#ff7f00"

# ── Carga de datos ────────────────────────────────────────────────────────────

def cargar_datos() -> tuple[np.ndarray, np.ndarray]:
    """Lee CSV experimental y retorna (T_N, f0_Hz) como arrays numpy.

    Solo se usan las columnas T_N y f0_Hz; masa y ángulo se ignoran
    (fueron el medio de obtener T, no variables del análisis).
    """
    T_vals:  list[float] = []
    f0_vals: list[float] = []
    with open(CSV_TENSION, encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            T_vals.append(float(row["T_N"]))
            f0_vals.append(float(row["f0_Hz"]))
    return np.array(T_vals), np.array(f0_vals)

# ── Ajuste potencial ──────────────────────────────────────────────────────────

def ajuste_potencial(
    T_arr: np.ndarray, f0_arr: np.ndarray
) -> tuple[float, float, float]:
    """Ajusta f₀ = a · T^b en escala log-log.

    Returns:
        a, b, R²   (R² calculado en espacio logarítmico)
    """
    logT = np.log10(T_arr)
    logf = np.log10(f0_arr)
    coeffs = np.polyfit(logT, logf, 1)
    b, loga = coeffs
    a = 10 ** loga
    logf_pred = np.polyval(coeffs, logT)
    ss_res = np.sum((logf - logf_pred) ** 2)
    ss_tot = np.sum((logf - logf.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return float(a), float(b), float(r2)

# ── Detección de outliers ─────────────────────────────────────────────────────

def mascara_outliers(
    T_arr: np.ndarray,
    f0_arr: np.ndarray,
    a: float,
    b: float,
    umbral: float = UMBRAL_OUTLIER,
) -> np.ndarray:
    """Retorna máscara booleana: True = punto bueno."""
    f_pred = a * T_arr ** b
    return np.abs(f0_arr - f_pred) / f_pred <= umbral

# ── Gráficas ──────────────────────────────────────────────────────────────────

def graficar_ajuste(
    T_ok: np.ndarray, f0_ok: np.ndarray,
    T_out: np.ndarray, f0_out: np.ndarray,
    a: float, b: float, r2: float,
    ruta: Path,
) -> None:
    """Dos subplots: f₀ vs T (escala lineal) y linealización log-log."""
    T_fino = np.linspace(T_ok.min() * 0.85, T_ok.max() * 1.05, 300)
    f_fit  = a * T_fino ** b

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # ── f₀ vs T ─────────────────────────────────────────────────────────────
    ax1.plot(T_fino, f_fit, "-", color=COLOR, lw=1.8, alpha=0.7,
             label=f"Ajuste  f₀ = {a:.3f}·T^{{{b:.4f}}}  (R²={r2:.4f})")
    ax1.scatter(T_ok, f0_ok, color=COLOR, s=50, zorder=5, label="Medido")
    if len(T_out) > 0:
        ax1.scatter(T_out, f0_out, marker="x", color="gray", s=65,
                    linewidths=1.6, zorder=5, label="Outlier")
    ax1.set_xlabel("T  (N)")
    ax1.set_ylabel("f₀  (Hz)")
    ax1.set_title("C5 — La2 (A2)  |  f₀ vs T")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # ── Linealización log-log ────────────────────────────────────────────────
    ax2.plot(np.log10(T_fino), np.log10(f_fit), "-", color=COLOR, lw=1.8, alpha=0.7,
             label=f"Y = {b:.4f}·X + {np.log10(a):.4f}")
    ax2.scatter(np.log10(T_ok), np.log10(f0_ok), color=COLOR, s=50, zorder=5)
    if len(T_out) > 0:
        ax2.scatter(np.log10(T_out), np.log10(f0_out),
                    marker="x", color="gray", s=65, linewidths=1.6, zorder=5)
    ax2.set_xlabel("log₁₀(T)")
    ax2.set_ylabel("log₁₀(f₀)")
    ax2.set_title(f"Linealización  |  pendiente b = {b:.4f}  (teórico +0.5000)")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    fig.suptitle("C5 — La2 (A2)  |  f₀ ∝ T^(1/2)", fontsize=11)
    fig.tight_layout()
    fig.savefig(ruta, dpi=110)
    plt.close(fig)


def graficar_normalizado(
    T_ok: np.ndarray, f0_ok: np.ndarray, ruta: Path
) -> None:
    """f₀/f₀_ref vs T/T_ref — colapso sobre y = √x si f₀ ∝ √T."""
    idx_ref = np.argmax(T_ok)   # referencia: punto de mayor tensión
    T_ref   = T_ok[idx_ref]
    f0_ref  = f0_ok[idx_ref]

    T_norm  = T_ok  / T_ref
    f0_norm = f0_ok / f0_ref

    x_teo = np.linspace(T_norm.min() * 0.85, T_norm.max() * 1.05, 300)
    y_teo = np.sqrt(x_teo)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(x_teo, y_teo, "k--", lw=1.5, alpha=0.6, label="Teórico  y = √x")
    ax.scatter(T_norm, f0_norm, color=COLOR, s=55, zorder=4, label="C5 La2 (A2)")
    ax.set_xlabel("T / T_ref")
    ax.set_ylabel("f₀ / f₀(T_ref)")
    ax.set_title("Colapso  f₀/f₀_ref vs T/T_ref  —  C5 La2 (A2)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(ruta, dpi=110)
    plt.close(fig)

# ── Pipeline ──────────────────────────────────────────────────────────────────

def main() -> None:
    FIGURAS_DIR.mkdir(exist_ok=True)

    T_all, f0_all = cargar_datos()

    # Primera pasada: ajuste completo para identificar outliers
    a0, b0, _ = ajuste_potencial(T_all, f0_all)
    mask = mascara_outliers(T_all, f0_all, a0, b0)

    T_ok,  f0_ok  = T_all[mask],  f0_all[mask]
    T_out, f0_out = T_all[~mask], f0_all[~mask]

    # Segunda pasada: ajuste sobre puntos limpios
    a, b, r2 = ajuste_potencial(T_ok, f0_ok)

    print(f"\n{'─'*60}")
    print(f"C5 La2 (A2)  —  f₀ = {a:.3f} · T^{b:.4f}   R² = {r2:.5f}")
    print(f"Exponente teórico: b = +0.5000   Δb = {b - 0.5:+.4f}")
    print(f"Puntos totales: {len(T_all)}   buenos: {len(T_ok)}   outliers: {len(T_out)}")
    if len(T_out) > 0:
        print("Outliers  T(N) → f₀(Hz):")
        for T, f in zip(T_out, f0_out):
            print(f"  {T:.3f} N  →  {f:.3f} Hz")
    print(f"{'─'*60}")

    graficar_ajuste(T_ok, f0_ok, T_out, f0_out, a, b, r2,
                    ruta=FIGURAS_DIR / "fT_ajuste.png")
    graficar_normalizado(T_ok, f0_ok, ruta=FIGURAS_DIR / "fT_normalizado.png")

    # CSV — estrictamente bivariado: solo f0_Hz y T_N (más identificadores)
    csv_path = Path("datos_fT.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        campos = ["cuerda", "punto", "f0_Hz", "T_N", "outlier"]
        writer = csv.DictWriter(fh, fieldnames=campos)
        writer.writeheader()
        for i, (T, f0, ok) in enumerate(zip(T_all, f0_all, mask), start=1):
            writer.writerow({
                "cuerda":  CUERDA,
                "punto":   i,
                "f0_Hz":   round(float(f0), 4),
                "T_N":     round(float(T), 4),
                "outlier": 0 if ok else 1,
            })

    print(f"\nFiguras:  {FIGURAS_DIR}/fT_*.png")
    print(f"CSV:      {csv_path}  (columnas: cuerda, punto, f0_Hz, T_N, outlier)")


if __name__ == "__main__":
    main()
