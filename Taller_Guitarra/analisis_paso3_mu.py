#!/usr/bin/env python3
"""
Análisis f(μ): f₀ vs densidad lineal de masa — ley parcial de Mersenne

Variables utilizadas: únicamente f₀ (medida, FFT) y μ (D'Addario EJ16 specs).
No se usa L ni T — análisis estrictamente bivariado.

Procedimiento:
  1. Ajuste potencial:  f₀ = a · μ^b  (regresión log-log por traste)
  2. Linealización:     log(f₀) = b·log(μ) + log(a)  →  Y = bX + A
  3. Distribución de exponentes b sobre los 20 trastes
  4. Normalizado:       f₀/f₀(C1) vs μ/μ(C1) — colapso sobre μ^(-0.5)

Salidas:
  figuras/fmu_por_traste.png   — f₀ vs μ por traste (log-log + lineal)
  figuras/fmu_exponentes.png   — exponente b vs traste + R²
  figuras/fmu_normalizado.png  — f₀ normalizada vs μ normalizada
  datos_fmu.csv                — (cuerda, traste, f₀_Hz, μ_kg/m)

[SCI] D'Addario EJ16 light gauge phosphor bronze, 6 cuerdas × 20 trastes.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ─── Rutas ────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
FIG_DIR  = BASE_DIR / "figuras"
CSV_F0   = BASE_DIR / "frecuencias_fundamentales.csv"
CSV_FMU  = BASE_DIR / "datos_fmu.csv"
FIG_DIR.mkdir(exist_ok=True)

# ─── Constantes ───────────────────────────────────────────────────────────────
# D'Addario EJ16 — unit weights oficiales (lb/in) × 17.858 = kg/m
MU_KG_M = {
    1: 3.90e-4,   # Mi4  E4  .012" acero plain
    2: 7.05e-4,   # Si3  B3  .016" acero plain
    3: 1.59e-3,   # Sol3 G3  .024w bronce fosf.
    4: 2.84e-3,   # Re3  D3  .032w bronce fosf.
    5: 4.98e-3,   # La2  A2  .042w bronce fosf.
    6: 7.97e-3,   # Mi2  E2  .053w bronce fosf.
}

NOMBRES = {
    1: 'C1 E4 .012"',
    2: 'C2 B3 .016"',
    3: 'C3 G3 .024w',
    4: 'C4 D3 .032w',
    5: 'C5 A2 .042w',
    6: 'C6 E2 .053w',
}

COLORES = {1: "#e41a1c", 2: "#ff7f00", 3: "#4daf4a",
           4: "#377eb8", 5: "#984ea3", 6: "#a65628"}

# ─── Carga y construcción del dataset ─────────────────────────────────────────

def cargar_datos() -> pd.DataFrame:
    df = pd.read_csv(CSV_F0)
    df["mu_kg_m"] = df["cuerda"].map(MU_KG_M)
    df["nombre"]  = df["cuerda"].map(NOMBRES)
    return df.sort_values(["cuerda", "traste"]).reset_index(drop=True)


# ─── Ajuste potencial ─────────────────────────────────────────────────────────

def ajuste_potencial(
    x: np.ndarray, y: np.ndarray
) -> tuple[float, float, float]:
    """Regresión log-log: y = a·x^b  →  (a, b, R²)."""
    lx = np.log10(x)
    ly = np.log10(y)
    b, log_a = np.polyfit(lx, ly, 1)
    y_pred = np.polyval([b, log_a], lx)
    ss_res = np.sum((ly - y_pred) ** 2)
    ss_tot = np.sum((ly - ly.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return float(10 ** log_a), float(b), float(r2)


# ─── Figura 1: f₀ vs μ (trastes seleccionados) ───────────────────────────────

def graficar_fmu_por_traste(df: pd.DataFrame, ruta: Path) -> None:
    """6 puntos por traste (L fijo, μ y T varían). Escala log-log + lineal."""
    trastes_sel = [0, 3, 6, 9, 12, 15, 18]
    cmap = plt.cm.viridis
    col_t = {t: cmap(i / (len(trastes_sel) - 1)) for i, t in enumerate(trastes_sel)}

    mus_todos = np.array([MU_KG_M[c] for c in range(1, 7)])
    mu_fit_range = np.logspace(np.log10(mus_todos.min() * 0.7),
                               np.log10(mus_todos.max() * 1.3), 300)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for t in trastes_sel:
        sub = df[df["traste"] == t].sort_values("cuerda")
        if len(sub) < 4:
            continue
        mu_t = sub["mu_kg_m"].values
        f0_t = sub["f0_Hz"].values
        a, b, r2 = ajuste_potencial(mu_t, f0_t)
        f_fit = a * mu_fit_range ** b
        c = col_t[t]
        lbl = f"T{t:02d}  b = {b:.3f}  R² = {r2:.4f}"
        for ax in axes:
            ax.scatter(mu_t, f0_t, color=c, s=55, zorder=4)
            ax.plot(mu_fit_range, f_fit, color=c, lw=1.3, label=lbl)

    # Referencia teórica b = -0.5 anclada al traste 0
    sub0 = df[df["traste"] == 0].sort_values("cuerda")
    if len(sub0) >= 2:
        mu_mid = np.sqrt(mus_todos.min() * mus_todos.max())
        idx = np.argmin(np.abs(sub0["mu_kg_m"].values - mu_mid))
        f_anchor = sub0["f0_Hz"].values[idx]
        mu_anchor = sub0["mu_kg_m"].values[idx]
        f_ref = f_anchor * (mu_fit_range / mu_anchor) ** (-0.5)
        for ax in axes:
            ax.plot(mu_fit_range, f_ref, "k--", lw=1.8, alpha=0.55,
                    label="Teórico: b = −0.5")

    # Marcas de cuerdas en el eje x (μ de cada cuerda)
    mu_vals = [MU_KG_M[c] for c in range(1, 7)]
    labels_mu = [f"C{c}\n{MU_KG_M[c]:.2e}" for c in range(1, 7)]

    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("μ  (kg/m)", fontsize=12)
    axes[0].set_ylabel("f₀  (Hz)", fontsize=12)
    axes[0].set_title("f₀ vs μ — log-log", fontsize=13)
    axes[0].legend(fontsize=7.5, loc="upper right", ncol=1)
    axes[0].grid(True, which="both", alpha=0.3)

    axes[1].set_xlabel("μ  (kg/m)", fontsize=12)
    axes[1].set_ylabel("f₀  (Hz)", fontsize=12)
    axes[1].set_title("f₀ vs μ — lineal", fontsize=13)
    axes[1].grid(True, alpha=0.3)
    ax2_top = axes[1].twiny()
    ax2_top.set_xlim(axes[1].get_xlim())
    ax2_top.set_xticks(mu_vals)
    ax2_top.set_xticklabels([f"C{c}" for c in range(1, 7)], fontsize=8)

    fig.suptitle(
        "Ley parcial de Mersenne: f₀ ∝ μ^b\n"
        "(cada curva = traste fijo; L igual para todas las cuerdas)",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ─── Figura 2: velocidad de onda v = 2f₀L vs μ ───────────────────────────────

def graficar_velocidad_vs_mu(df: pd.DataFrame, ruta: Path) -> None:
    """
    v = 2f₀L = √(T/μ). Elimina la dependencia en L.
    La pendiente log-log debería ser ≈ -0.5 si T fuera constante.
    La desviación cuantifica la correlación T-μ en la guitarra.
    """
    mus_med  = np.array([MU_KG_M[c] for c in range(1, 7)])
    y_med    = np.array([df[df["cuerda"] == c]["dos_f0_L"].median() for c in range(1, 7)])

    a_gl, b_gl, r2_gl = ajuste_potencial(mus_med, y_med)
    mu_fit = np.logspace(np.log10(mus_med.min() * 0.6),
                         np.log10(mus_med.max() * 1.4), 300)
    y_fit  = a_gl * mu_fit ** b_gl

    # Referencia b = -0.5
    y_mid  = np.median(y_med)
    mu_mid = np.sqrt(mus_med.min() * mus_med.max())
    y_ref  = y_mid * (mu_fit / mu_mid) ** (-0.5)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for c in range(1, 7):
        sub   = df[df["cuerda"] == c]
        mu_c  = MU_KG_M[c]
        y_c   = sub["dos_f0_L"].values
        mu_jitter = np.full(len(y_c), mu_c) * np.random.uniform(0.97, 1.03, len(y_c))
        color = COLORES[c]
        for ax in axes:
            ax.scatter(mu_jitter, y_c, color=color, s=18, alpha=0.55,
                       label=NOMBRES[c] if ax is axes[0] else None)
            ax.scatter([mu_c], [y_med[c - 1]], color=color, s=100,
                       marker="D", edgecolors="k", lw=0.8, zorder=5)

    for ax in axes:
        ax.plot(mu_fit, y_fit, "k-", lw=2.2,
                label=f"Ajuste (medianas): b = {b_gl:.3f}  R² = {r2_gl:.4f}")
        ax.plot(mu_fit, y_ref, "r--", lw=1.7, alpha=0.65, label="Teórico: b = −0.5")

    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("μ  (kg/m)", fontsize=12)
    axes[0].set_ylabel("2·f₀·L  (m/s)", fontsize=12)
    axes[0].set_title("2·f₀·L vs μ — log-log", fontsize=13)
    axes[0].legend(fontsize=8, loc="upper right")
    axes[0].grid(True, which="both", alpha=0.3)

    axes[1].set_xlabel("μ  (kg/m)", fontsize=12)
    axes[1].set_ylabel("2·f₀·L  (m/s)", fontsize=12)
    axes[1].set_title("2·f₀·L vs μ — lineal", fontsize=13)
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(
        f"2·f₀·L vs μ — elimina dependencia en L\n"
        f"Ajuste global: b = {b_gl:.3f}  (teórico −0.5)",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return b_gl, r2_gl


# ─── Figura 3: exponente b vs traste ─────────────────────────────────────────

def graficar_exponentes(df: pd.DataFrame, ruta: Path) -> dict:
    """Ajusta f₀ = a·μ^b con las 6 cuerdas para cada traste."""
    bs, r2s, trastes_ok = [], [], []

    for t in range(20):
        sub = df[df["traste"] == t].sort_values("cuerda")
        if len(sub) < 5:
            continue
        a, b, r2 = ajuste_potencial(sub["mu_kg_m"].values, sub["f0_Hz"].values)
        bs.append(b)
        r2s.append(r2)
        trastes_ok.append(t)

    bs_arr = np.array(bs)
    r2_arr = np.array(r2s)
    b_med  = float(np.mean(bs_arr))
    b_std  = float(np.std(bs_arr))

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    axes[0].plot(trastes_ok, bs_arr, "o-", color="#377eb8", lw=1.6, ms=6, label="b ajustado")
    axes[0].axhline(-0.5, color="red", ls="--", lw=1.8, label="Teórico: b = −0.5")
    axes[0].axhline(b_med, color="green", ls=":", lw=1.8,
                    label=f"Media: b = {b_med:.4f} ± {b_std:.4f}")
    axes[0].fill_between(trastes_ok, b_med - b_std, b_med + b_std,
                         alpha=0.15, color="green")
    axes[0].set_ylabel("Exponente b", fontsize=12)
    axes[0].set_title("Exponente de f₀ = a · μ^b  por traste (6 cuerdas)", fontsize=13)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(b_med - 0.15, b_med + 0.15)

    axes[1].plot(trastes_ok, r2_arr, "s-", color="#e41a1c", lw=1.6, ms=6)
    axes[1].axhline(1.0, color="gray", ls="--", lw=1, alpha=0.5)
    axes[1].set_ylabel("R²", fontsize=12)
    axes[1].set_xlabel("Traste", fontsize=12)
    axes[1].set_title("Calidad del ajuste por traste", fontsize=13)
    axes[1].set_ylim(max(0.85, r2_arr.min() - 0.02), 1.01)
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(
        "Ley f(μ): análisis del exponente por traste\n"
        "Desviación de −0.5 debida a variación de T entre cuerdas (plain vs wound)",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return {"b_medio": b_med, "b_std": b_std,
            "r2_medio": float(np.mean(r2_arr)),
            "bs": bs, "r2s": r2s, "trastes": trastes_ok}


# ─── Figura 4: normalizado f₀/f₀(C1) vs μ/μ(C1) ─────────────────────────────

def graficar_normalizado(df: pd.DataFrame, ruta: Path) -> None:
    """Colapsa todas las cuerdas a una sola curva relativa a C1 (por traste)."""
    fig, ax = plt.subplots(figsize=(9, 7))
    cmap    = plt.cm.plasma
    trastes = sorted(df["traste"].unique())
    mu_c1   = MU_KG_M[1]

    for i, t in enumerate(trastes):
        sub = df[df["traste"] == t].sort_values("cuerda")
        if 1 not in sub["cuerda"].values or len(sub) < 4:
            continue
        f_c1 = sub.loc[sub["cuerda"] == 1, "f0_Hz"].values[0]
        x = sub["mu_kg_m"].values / mu_c1
        y = sub["f0_Hz"].values / f_c1
        color = cmap(i / (len(trastes) - 1))
        ax.plot(x, y, "o-", color=color, lw=1.1, ms=4, alpha=0.8)

    x_ref = np.logspace(0, np.log10(MU_KG_M[6] / mu_c1) + 0.1, 300)
    ax.plot(x_ref, x_ref ** (-0.5), "k--", lw=2.2, label="Teórico: $(μ/μ_1)^{-0.5}$")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("μ / μ(C1)", fontsize=12)
    ax.set_ylabel("f₀ / f₀(C1)", fontsize=12)
    ax.set_title("f₀ normalizada vs μ normalizada (referencia: C1 por traste)", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, which="both", alpha=0.3)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 19))
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label="Traste")
    fig.tight_layout()
    fig.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ─── Exportar CSVs ────────────────────────────────────────────────────────────

def exportar_datasets(df: pd.DataFrame) -> None:
    df_fmu = df[["cuerda", "nombre", "traste", "f0_Hz", "mu_kg_m"]].copy()
    df_fmu.to_csv(CSV_FMU, index=False, float_format="%.6g")
    print(f"  → {CSV_FMU.name}  ({len(df_fmu)} filas)")


# ─── Resumen en consola ───────────────────────────────────────────────────────

def imprimir_resumen(df: pd.DataFrame, res_exp: dict) -> None:
    sub0 = df[df["traste"] == 0].sort_values("cuerda")

    print("\n─── f₀ vs μ — cuerdas al aire (traste 0) ────────────────────────────────")
    a0, b0, r20 = ajuste_potencial(sub0["mu_kg_m"].values, sub0["f0_Hz"].values)
    print(f"  Ajuste f₀ = {a0:.2f} · μ^{b0:.4f}   R² = {r20:.5f}")
    print(f"  Teórico: b = −0.5000    Δb = {b0 - (-0.5):+.4f}")
    print()
    print(f"  {'Cuerda':<18} {'f₀(Hz)':>9} {'μ(kg/m)':>10}")
    print("  " + "─" * 42)
    for _, row in sub0.iterrows():
        print(f"  {row['nombre']:<18} {row['f0_Hz']:>9.2f} {row['mu_kg_m']:>10.3e}")

    print("\n─── Exponentes f₀ = a·μ^b (todos los trastes) ───────────────────────────")
    print(f"  b medio:   {res_exp['b_medio']:.4f} ± {res_exp['b_std']:.4f}")
    print(f"  Teórico:   -0.5000")
    print(f"  Δb medio:  {res_exp['b_medio'] - (-0.5):+.4f}")
    print(f"  R² medio:  {res_exp['r2_medio']:.5f}")
    print(f"\n  CSV: datos_fmu.csv  (columnas: cuerda, traste, f0_Hz, mu_kg_m)")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    np.random.seed(42)  # para jitter reproducible en scatter plots

    print("Cargando datos...")
    df = cargar_datos()
    print(f"  {len(df)} registros  ({df['cuerda'].nunique()} cuerdas × {df['traste'].nunique()} trastes)")

    print("\nGenerando figuras...")
    graficar_fmu_por_traste(df, FIG_DIR / "fmu_por_traste.png")
    print("  fmu_por_traste.png")

    res_exp = graficar_exponentes(df, FIG_DIR / "fmu_exponentes.png")
    print(f"  fmu_exponentes.png  (b={res_exp['b_medio']:.4f} ± {res_exp['b_std']:.4f})")

    graficar_normalizado(df, FIG_DIR / "fmu_normalizado.png")
    print("  fmu_normalizado.png")

    print("\nExportando CSV...")
    exportar_datasets(df)

    imprimir_resumen(df, res_exp)


if __name__ == "__main__":
    main()
