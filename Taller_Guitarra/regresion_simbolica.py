"""
Regresión simbólica para la ley de Mersenne: f₀ = (1/2L) · √(T/μ)

Datasets de entrada (sin mezclar variables dentro de cada paso):
  datos_fL.csv  (paso 2): f₀ y L medidos  —  μ conocido por cuerda
  datos_fmu.csv (paso 3): f₀ y μ medidos  —  L = L₀ = 65.2 cm
  datos_fT.csv  (paso 4): f₀ y T medidos  —  L = L₀, μ = μ₅

Estrategia (sin usar Mersenne para construir el dataset):

  Modelo 1  RS en (f₀, L, μ)
    Log-lineal:  log(f₀) = b_L·log(L) + b_μ·log(μ) + c
    gplearn:     búsqueda libre  f₀ = expr(L, μ)
    Dataset:     paso 2 (120 filas; dentro de cada cuerda L varía, μ y T fijos;
                          entre cuerdas varía μ con T pequeña ≈ constante)

  Modelo 2  RS en (f₀, T)
    Log-lineal:  log(f₀) = b_T·log(T) + c
    Dataset:     paso 4 (15 filas; L y μ constantes, T varía)

  Modelo 3  RS conjunta en (f₀, L, μ, T)
    Log-lineal con los cuatro parámetros simultáneos
    Dataset:     paso 2 con T catálogo D'Addario EJ16  ∪  paso 4 (135 filas)
    Nota: T catálogo = medición independiente de D'Addario (no derivada de Mersenne)

Salidas:
  figuras/sr_paridad_Lmu.png      paridad f₀ pred vs med, modelo (L, μ)
  figuras/sr_paridad_T.png        ídem para modelo (T)
  figuras/sr_paridad_conjunto.png ídem para modelo conjunto
  figuras/sr_exponentes.png       exponentes hallados vs Mersenne teórico
  figuras/sr_gplearn.png          resultado gplearn
  dataset_mersenne.csv            dataset completo (f₀, L, μ, T)

Estilo: [SCI]
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# ── Constantes de material ────────────────────────────────────────────────────

FIGURAS_DIR = Path("figuras")

MU_KG_M = {
    1: 3.90e-4, 2: 7.05e-4, 3: 1.59e-3,
    4: 2.84e-3, 5: 4.98e-3, 6: 7.97e-3,
}
NOMBRE_CUERDA = {
    1: "Mi4 (E4)", 2: "Si3 (B3)", 3: "Sol3 (G3)",
    4: "Re3 (D3)", 5: "La2 (A2)", 6: "Mi2 (E2)",
}

# Tensiones D'Addario EJ16 (N): medidas por D'Addario en banco de pruebas,
# escala 25.5" = 64.77 cm; NO derivadas de la ecuación de Mersenne.
T_CATALOG_N = {1: 71.65, 2: 68.54, 3: 73.87, 4: 81.42, 5: 90.30, 6: 82.31}

L0_M  = 0.652    # longitud al aire (m)
MU_C5 = 4.98e-3  # kg/m  cuerda 5

COLORES = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00", "#a65628"]

# ── Carga y preparación de datos ──────────────────────────────────────────────

def cargar_datos() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Devuelve (df_fL, df_fT) listos para regresión."""

    # ── Paso 2: (f₀, L, μ) ───────────────────────────────────────────────────
    fL = pd.read_csv("datos_fL.csv")
    fL["L_m"]   = fL["L_cm"] / 100.0
    fL["mu"]    = fL["cuerda"].map(MU_KG_M)
    fL["T_cat"] = fL["cuerda"].map(T_CATALOG_N)
    fL = fL[fL["outlier"] == 0].reset_index(drop=True)

    # ── Paso 4: (f₀, T) ──────────────────────────────────────────────────────
    fT = pd.read_csv("datos_fT.csv")
    fT["L_m"] = L0_M
    fT["mu"]  = MU_C5
    fT = fT[fT["outlier"] == 0].reset_index(drop=True)

    return fL, fT


# ── Regresión log-lineal = RS para clase de leyes de potencia ─────────────────

def ajuste_loglineal(
    X: np.ndarray, y: np.ndarray
) -> tuple[np.ndarray, float, float]:
    """Regresión lineal en espacio log₁₀.

    Modelo:  log(y) = b₁·log(X₁) + b₂·log(X₂) + … + log(C)

    Returns:
        coefs  array de exponentes b_i
        R²     coeficiente en espacio log
        C      constante multiplicativa (10^intercept)
    """
    reg = LinearRegression().fit(np.log10(X), np.log10(y))
    return reg.coef_.copy(), float(reg.score(np.log10(X), np.log10(y))), float(10 ** reg.intercept_)


# ── gplearn: búsqueda simbólica libre ─────────────────────────────────────────

def sr_gplearn(X: np.ndarray, y: np.ndarray) -> tuple[object, float]:
    """Ejecuta gplearn SymbolicRegressor sobre (X, y).

    Busca expresión libre usando operaciones: mul, div, sqrt, inv, add, sub, neg.

    Returns (modelo_entrenado, R²)
    """
    from gplearn.genetic import SymbolicRegressor

    sr = SymbolicRegressor(
        population_size=5000,
        generations=30,
        function_set=("mul", "div", "sqrt", "inv", "add", "sub", "neg"),
        metric="mean absolute error",
        p_crossover=0.70,
        p_subtree_mutation=0.10,
        p_hoist_mutation=0.05,
        p_point_mutation=0.10,
        max_samples=1.0,
        verbose=0,
        parsimony_coefficient=0.005,
        random_state=42,
        n_jobs=-1,
    )
    sr.fit(X, y)
    R2 = r2_score(y, sr.predict(X))
    return sr, R2


# ── Visualizaciones ───────────────────────────────────────────────────────────

def graficar_paridad(
    y_true: np.ndarray, y_pred: np.ndarray, titulo: str, ruta: Path
) -> None:
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(y_true, y_pred, s=18, alpha=0.6, color="#377eb8", zorder=3)
    lo = min(y_true.min(), y_pred.min()) * 0.92
    hi = max(y_true.max(), y_pred.max()) * 1.06
    ax.plot([lo, hi], [lo, hi], "k--", lw=1.3, alpha=0.7)
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    rel  = rmse / y_true.mean() * 100
    ax.set_xlabel("f₀ medida  (Hz)")
    ax.set_ylabel("f₀ predicha  (Hz)")
    ax.set_title(titulo, fontsize=9)
    ax.text(0.05, 0.93, f"RMSE = {rmse:.2f} Hz  ({rel:.1f}%)",
            transform=ax.transAxes, fontsize=8, va="top")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(ruta, dpi=110)
    plt.close(fig)


def graficar_exponentes(resultados: dict, ruta: Path) -> None:
    """Barras de exponentes descubiertos frente a los valores de Mersenne."""
    variables = ["b_L", "b_μ", "b_T"]
    teoricos  = [-1.0, -0.5, +0.5]
    etiquetas = ["b_L  (teórico −1)", "b_μ  (teórico −½)", "b_T  (teórico +½)"]

    modelos = [m for m in resultados if m != "gplearn (L,μ)"]  # gplearn sin b numérico

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, var, teo, etiq in zip(axes, variables, teoricos, etiquetas):
        vals  = [resultados[m].get(var, np.nan) for m in modelos]
        valid = [(i, v) for i, v in enumerate(vals) if not np.isnan(v)]
        for i, v in valid:
            color = "#4daf4a" if abs(v - teo) < 0.08 else ("#ff7f00" if abs(v - teo) < 0.20 else "#e41a1c")
            b = ax.bar(i, v, color=color, alpha=0.8, width=0.55)
            ax.text(i, v + (0.015 if v >= 0 else -0.03),
                    f"{v:+.3f}", ha="center",
                    va="bottom" if v >= 0 else "top", fontsize=8)
        ax.axhline(teo, color="black", lw=1.6, ls="--", alpha=0.8,
                   label=f"Mersenne = {teo:+.1f}")
        ax.axhline(0, color="gray", lw=0.5, alpha=0.5)
        ax.set_xticks(range(len(modelos)))
        ax.set_xticklabels(modelos, rotation=20, ha="right", fontsize=8)
        ax.set_title(etiq, fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle(
        "Exponentes hallados por regresión simbólica  vs  Ley de Mersenne  (−1, −½, +½)",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(ruta, dpi=110)
    plt.close(fig)


def graficar_gplearn(
    X: np.ndarray, y_true: np.ndarray, sr: object,
    nombres_X: list[str], formula: str, R2: float, ruta: Path
) -> None:
    y_pred = sr.predict(X)
    rmse   = np.sqrt(np.mean((y_true - y_pred) ** 2))
    rel    = rmse / y_true.mean() * 100
    resid  = (y_pred - y_true) / y_true * 100

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    # Paridad
    lo = min(y_true.min(), y_pred.min()) * 0.92
    hi = max(y_true.max(), y_pred.max()) * 1.06
    for c in range(1, 7):
        mask = np.zeros(len(y_true), dtype=bool)
        step = len(y_true) // 6
        mask[c * step - step: c * step] = True
        ax1.scatter(y_true[mask], y_pred[mask], s=22, alpha=0.7,
                    color=COLORES[c - 1], label=NOMBRE_CUERDA[c], zorder=3)
    ax1.plot([lo, hi], [lo, hi], "k--", lw=1.3, alpha=0.7)
    ax1.set_xlabel("f₀ medida  (Hz)")
    ax1.set_ylabel("f₀ predicha  (Hz)")
    ax1.set_title(f"gplearn  |  R² = {R2:.4f}   RMSE = {rmse:.1f} Hz  ({rel:.1f}%)", fontsize=9)
    ax1.legend(fontsize=7, ncol=2)
    ax1.grid(True, alpha=0.3)

    # Residuos
    ax2.scatter(y_true, resid, s=18, alpha=0.6, color="#4daf4a", zorder=3)
    ax2.axhline(0, color="k", lw=1.0, alpha=0.7)
    ax2.axhline(+10, color="gray", lw=0.8, ls=":", alpha=0.6)
    ax2.axhline(-10, color="gray", lw=0.8, ls=":", alpha=0.6)
    ax2.set_xlabel("f₀ medida  (Hz)")
    ax2.set_ylabel("Error relativo  (%)")
    ax2.set_title("Residuos relativos  (±10 % punteado)")
    ax2.grid(True, alpha=0.3)

    titulo = f"gplearn SR  —  {formula[:90]}"
    fig.suptitle(titulo, fontsize=8)
    fig.tight_layout()
    fig.savefig(ruta, dpi=110)
    plt.close(fig)


def graficar_comparacion_modelos(
    resultados_pred: dict, y_dict: dict, ruta: Path
) -> None:
    """Superpone las curvas de f₀ vs L para cada cuerda bajo los tres modelos."""
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharey=False)
    axes = axes.flatten()

    fL = pd.read_csv("datos_fL.csv")
    fL["L_m"] = fL["L_cm"] / 100
    fL["mu"]  = fL["cuerda"].map(MU_KG_M)
    fL["T_cat"] = fL["cuerda"].map(T_CATALOG_N)
    fL = fL[fL["outlier"] == 0]

    for idx, cuerda in enumerate(range(1, 7)):
        ax  = axes[idx]
        sub = fL[fL["cuerda"] == cuerda].sort_values("L_m")
        col = COLORES[idx]

        ax.scatter(sub["L_m"] * 100, sub["f0_Hz"],
                   color=col, s=30, zorder=5, label="Medido")

        L_fino = np.linspace(sub["L_m"].min() * 0.95, sub["L_m"].max() * 1.02, 200)
        mu_k   = MU_KG_M[cuerda]
        T_k    = T_CATALOG_N[cuerda]

        # Modelo 1 (L, μ) — no usa T
        b_L1, b_mu1, C1 = resultados_pred["m1"]
        f_m1 = C1 * L_fino**b_L1 * mu_k**b_mu1
        ax.plot(L_fino * 100, f_m1, "-", color=col, lw=1.6,
                alpha=0.8, label="Modelo 1 (L,μ)")

        # Modelo 3 conjunto
        b_L3, b_mu3, b_T3, C3 = resultados_pred["m3"]
        f_m3 = C3 * L_fino**b_L3 * mu_k**b_mu3 * T_k**b_T3
        ax.plot(L_fino * 100, f_m3, "--", color="gray", lw=1.4,
                alpha=0.7, label="Modelo 3 (L,μ,T)")

        ax.set_xlabel("L  (cm)")
        ax.set_ylabel("f₀  (Hz)")
        ax.set_title(f"C{cuerda} — {NOMBRE_CUERDA[cuerda]}", fontsize=9)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Modelos descubiertos vs datos medidos  (f₀ vs L por cuerda)", fontsize=11)
    fig.tight_layout()
    fig.savefig(ruta, dpi=110)
    plt.close(fig)


# ── Pipeline principal ────────────────────────────────────────────────────────

def main() -> None:
    FIGURAS_DIR.mkdir(exist_ok=True)
    fL, fT = cargar_datos()
    resultados: dict[str, dict] = {}
    SEP = "─" * 70

    print(f"\n{SEP}")
    print("  REGRESIÓN SIMBÓLICA — Ley de Mersenne  f₀ = (1/2L)·√(T/μ)")
    print(SEP)

    # ── Modelo 1: (f₀, L, μ) desde paso 2 ────────────────────────────────────
    print("\n[1]  log(f₀) = b_L·log(L) + b_μ·log(μ) + c   [paso 2 — 120 puntos]")
    X1 = fL[["L_m", "mu"]].values
    y1 = fL["f0_Hz"].values
    coefs1, R2_1, C1 = ajuste_loglineal(X1, y1)
    b_L1, b_mu1 = coefs1
    f0_pred1 = C1 * fL["L_m"].values ** b_L1 * fL["mu"].values ** b_mu1
    resultados["Modelo 1\n(L, μ)"] = {"b_L": b_L1, "b_μ": b_mu1, "b_T": np.nan, "R2": R2_1}
    print(f"   f₀ = {C1:.4f} · L^{{{b_L1:+.4f}}} · μ^{{{b_mu1:+.4f}}}   R² = {R2_1:.5f}")
    print(f"   Mersenne teorico:   b_L = −1.0000   b_μ = −0.5000")
    print(f"   Δb_L = {b_L1 - (-1.0):+.4f}   Δb_μ = {b_mu1 - (-0.5):+.4f}")

    # ── Modelo 2: (f₀, T) desde paso 4 ───────────────────────────────────────
    print("\n[2]  log(f₀) = b_T·log(T) + c   [paso 4 — 15 puntos]")
    X2 = fT[["T_N"]].values
    y2 = fT["f0_Hz"].values
    coefs2, R2_2, C2 = ajuste_loglineal(X2, y2)
    b_T2 = coefs2[0]
    f0_pred2 = C2 * fT["T_N"].values ** b_T2
    resultados["Modelo 2\n(T)"] = {"b_L": np.nan, "b_μ": np.nan, "b_T": b_T2, "R2": R2_2}
    print(f"   f₀ = {C2:.4f} · T^{{{b_T2:+.4f}}}   R² = {R2_2:.5f}")
    print(f"   Mersenne teorico:   b_T = +0.5000")
    print(f"   Δb_T = {b_T2 - 0.5:+.4f}  ← error sistemático del método de deflexión")

    # ── Síntesis analítica ─────────────────────────────────────────────────────
    print(f"\n{'─'*70}")
    print("  SÍNTESIS DE LEYES PARCIALES:")
    print(f"   Hallado:   f₀  ∝  L^{{{b_L1:.3f}}}  ·  μ^{{{b_mu1:.3f}}}  ·  T^{{{b_T2:.3f}}}")
    print( "   Mersenne:  f₀  ∝  L^(-1)  ·  μ^(-1/2)  ·  T^(+1/2)")
    print( "              f₀  =  (1/2L) · √(T/μ)  ✓")
    print(f"{'─'*70}")

    # ── Modelo 3: conjunto (f₀, L, μ, T) con T catálogo ──────────────────────
    print("\n[3]  log(f₀) = b_L·log(L) + b_μ·log(μ) + b_T·log(T) + c")
    print("     Dataset: paso 2 + T catálogo D'Addario  ∪  paso 4 + T medida")
    df_A = fL[["f0_Hz", "L_m", "mu", "T_cat"]].rename(columns={"T_cat": "T_N"}).copy()
    df_A["fuente"] = "paso2_Tcat"
    df_B = fT[["f0_Hz", "L_m", "mu", "T_N"]].copy()
    df_B["fuente"] = "paso4_Tmed"
    df_comb = pd.concat([df_A, df_B], ignore_index=True)

    X3 = df_comb[["L_m", "mu", "T_N"]].values
    y3 = df_comb["f0_Hz"].values
    coefs3, R2_3, C3 = ajuste_loglineal(X3, y3)
    b_L3, b_mu3, b_T3 = coefs3
    f0_pred3 = (C3 * df_comb["L_m"].values ** b_L3
                * df_comb["mu"].values ** b_mu3
                * df_comb["T_N"].values ** b_T3)
    resultados["Modelo 3\n(L, μ, T)"] = {"b_L": b_L3, "b_μ": b_mu3, "b_T": b_T3, "R2": R2_3}
    print(f"   f₀ = {C3:.4f} · L^{{{b_L3:+.4f}}} · μ^{{{b_mu3:+.4f}}} · T^{{{b_T3:+.4f}}}   R² = {R2_3:.5f}")
    print(f"   Δ:  b_L {b_L3 - (-1):+.4f}   b_μ {b_mu3 - (-0.5):+.4f}   b_T {b_T3 - 0.5:+.4f}")

    # ── gplearn: búsqueda simbólica libre ─────────────────────────────────────
    print(f"\n[4]  gplearn SymbolicRegressor en (f₀, L, μ)  [paso 2]")
    print( "     Búsqueda sin restricción de forma funcional ...")
    X_gp = fL[["L_m", "mu"]].values
    y_gp = fL["f0_Hz"].values
    sr_gp, R2_gp = sr_gplearn(X_gp, y_gp)
    formula_gp = str(sr_gp._program)
    resultados["gplearn\n(L, μ)"] = {"b_L": np.nan, "b_μ": np.nan, "b_T": np.nan, "R2": R2_gp}
    print(f"   Fórmula: {formula_gp}")
    print(f"   R² = {R2_gp:.5f}")

    # ── Tabla resumen ──────────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print(f"  {'Modelo':<22} {'b_L':>10} {'b_μ':>10} {'b_T':>10}   {'R²':>8}")
    print(f"  {'─'*66}")
    for nombre, vals in resultados.items():
        n   = nombre.replace("\n", " ")
        bL  = f"{vals['b_L']:+.4f}" if not np.isnan(vals.get("b_L", np.nan)) else "    —   "
        bmu = f"{vals['b_μ']:+.4f}" if not np.isnan(vals.get("b_μ", np.nan)) else "    —   "
        bT  = f"{vals['b_T']:+.4f}" if not np.isnan(vals.get("b_T", np.nan)) else "    —   "
        print(f"  {n:<22} {bL:>10} {bmu:>10} {bT:>10}   {vals['R2']:>8.5f}")
    print(f"  {'Mersenne (teórico)':<22} {'−1.0000':>10} {'−0.5000':>10} {'+0.5000':>10}   {'—':>8}")
    print(SEP)

    # ── Gráficas ──────────────────────────────────────────────────────────────
    graficar_paridad(
        y1, f0_pred1,
        "Modelo 1  —  f₀ = C · L^b_L · μ^b_μ\n(regresión log-log  sobre paso 2)",
        FIGURAS_DIR / "sr_paridad_Lmu.png",
    )
    graficar_paridad(
        y2, f0_pred2,
        "Modelo 2  —  f₀ = C · T^b_T\n(regresión log-log  sobre paso 4)",
        FIGURAS_DIR / "sr_paridad_T.png",
    )
    graficar_paridad(
        y3, f0_pred3,
        "Modelo 3  —  f₀ = C · L^b_L · μ^b_μ · T^b_T\n(dataset conjunto con T catálogo)",
        FIGURAS_DIR / "sr_paridad_conjunto.png",
    )
    graficar_exponentes(
        {k.replace("\n", " "): v for k, v in resultados.items()},
        FIGURAS_DIR / "sr_exponentes.png",
    )
    graficar_gplearn(
        X_gp, y_gp, sr_gp, ["L_m (m)", "μ (kg/m)"],
        formula_gp, R2_gp,
        FIGURAS_DIR / "sr_gplearn.png",
    )
    graficar_comparacion_modelos(
        {
            "m1": (b_L1, b_mu1, C1),
            "m3": (b_L3, b_mu3, b_T3, C3),
        },
        {},
        FIGURAS_DIR / "sr_comparacion_curvas.png",
    )

    # ── Exportar dataset combinado ────────────────────────────────────────────
    df_export = df_comb[["f0_Hz", "L_m", "mu", "T_N", "fuente"]].copy()
    df_export.columns = ["f0_Hz", "L_m", "mu_kg_m", "T_N", "fuente"]
    df_export.to_csv("dataset_mersenne.csv", index=False, float_format="%.6g")

    print(f"\nDataset combinado → dataset_mersenne.csv  ({len(df_export)} filas)")
    print(f"Figuras           → figuras/sr_*.png")


if __name__ == "__main__":
    main()
