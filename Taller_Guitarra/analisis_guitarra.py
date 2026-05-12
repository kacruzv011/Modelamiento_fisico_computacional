"""
Análisis espectral de cuerdas de guitarra acústica.
Paso 1: FFT → detección de picos → frecuencia fundamental.

Método f₀: template matching de armónicos.
  Para cada pico candidato f_c, se cuenta cuántos picos del espectro caen
  cerca de k·f_c (k=1,2,...).  El candidato con mayor número de armónicos
  explicados es f₀.  Esto equivale al método "delta entre picos" pero es
  robusto ante picos espurios de resonancias simpáticas inter-cuerdas.

Salidas:
  figuras/C{n}_T{m:02d}_espectro.png  — espectro individual con picos y armónicos marcados
  figuras/resumen_cuerda{n}.png       — f₀ vs traste por cuerda (medido + referencia temperada)
  figuras/resumen_traste{m:02d}.png   — f₀ vs cuerda por traste (medido + ajuste exp + referencia)
  figuras/overlay_trastes.png         — todas las curvas f₀ vs cuerda superpuestas
  figuras/resumen_global.png          — heatmap f₀ todas las cuerdas
  frecuencias_fundamentales.csv       — tabla larga (cuerda, traste, f₀_Hz)
  f0_por_traste.csv                   — tabla ancha (traste, C1_Hz, …, C6_Hz)

Estilo: [SCI]
"""

import csv
import re
from pathlib import Path

import librosa
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks, peak_prominences

# ── Configuración ─────────────────────────────────────────────────────────────

AUDIO_DIR   = Path("AUDIOS GUITARRA")
FIGURAS_DIR = Path("figuras")

N_CUERDAS   = 6
N_TRASTES   = 20          # T0 (al aire) … T19

FS_TARGET   = 44100       # Hz
VENTANA_SEG = 1.0         # s de señal estacionaria para la FFT
INICIO_REL  = 0.20        # fracción del audio donde empieza la búsqueda del segmento

F_MIN       = 60.0        # Hz, límite inferior de análisis
F_MAX       = 5500.0      # Hz, límite superior
DIST_MIN_HZ = 15.0        # Hz, separación mínima entre picos vecinos
N_TOP_PICOS = 15          # máximo de picos a retener (por prominencia)
TOLERANCIA  = 0.04        # tolerancia relativa para identificar armónicos (4%)

# Afinación estándar: MIDI de la cuerda al aire (grave→aguda)
MIDI_AL_AIRE = {1: 64, 2: 59, 3: 55, 4: 50, 5: 45, 6: 40}
NOMBRE_CUERDA = {
    1: "Mi4 (E4)", 2: "Si3 (B3)", 3: "Sol3 (G3)",
    4: "Re3 (D3)", 5: "La2 (A2)", 6: "Mi2 (E2)",
}

# Controla si se guardan los 120 espectros individuales (tarda ~2 min en total).
GUARDAR_ESPECTROS = True

# ── Búsqueda de archivos ──────────────────────────────────────────────────────

def buscar_archivo(cuerda: int, traste: int) -> Path | None:
    """Localiza el m4a para (cuerda, traste) tolerando espacios irregulares.

    'C 1 T 1 .m4a' y 'C 1 T 10.m4a' se distinguen extrayendo el último número
    del stem con regex: stem 'C 1 T 1 ' → ['1','1']; 'C 1 T 10' → ['1','10'].
    """
    carpeta = AUDIO_DIR / f"CUERDA {cuerda}"
    if not carpeta.exists():
        return None
    for candidato in carpeta.glob(f"C {cuerda} T {traste}*m4a"):
        nums = re.findall(r"\d+", candidato.stem)
        if nums and int(nums[-1]) == traste:
            return candidato
    return None

# ── Carga y segmentación ──────────────────────────────────────────────────────

def cargar_audio(ruta: Path) -> tuple[np.ndarray, int]:
    señal, fs = librosa.load(str(ruta), sr=FS_TARGET, mono=True)
    return señal, fs


def segmento_estacionario(señal: np.ndarray, fs: int) -> np.ndarray:
    """Devuelve el bloque de VENTANA_SEG s con mayor energía RMS.

    Busca desde INICIO_REL hasta el 75 % de la duración para evitar
    el ataque inicial y el silencio final.
    """
    n      = int(VENTANA_SEG * fs)
    inicio = int(INICIO_REL * len(señal))
    fin    = int(0.75 * len(señal))
    region = señal[inicio:fin]

    if len(region) < n:
        return señal[:n] if len(señal) >= n else señal

    paso     = n // 4
    mejor_i  = 0
    mejor_rms = -1.0
    for i in range(0, len(region) - n, paso):
        rms = float(np.sqrt(np.mean(region[i : i + n] ** 2)))
        if rms > mejor_rms:
            mejor_rms = rms
            mejor_i   = i
    return region[mejor_i : mejor_i + n]

# ── FFT ───────────────────────────────────────────────────────────────────────

def calcular_fft(señal: np.ndarray, fs: int) -> tuple[np.ndarray, np.ndarray]:
    """Ventana Hann + rfft → (frecuencias Hz, magnitud dB)."""
    w       = np.hanning(len(señal))
    espectro = np.fft.rfft(señal * w)
    mag_db  = 20.0 * np.log10(np.abs(espectro) + 1e-10)
    freqs   = np.fft.rfftfreq(len(señal), d=1.0 / fs)
    return freqs, mag_db

# ── Detección de picos ────────────────────────────────────────────────────────

def detectar_picos(
    freqs: np.ndarray,
    mag_db: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Retorna (f_picos, mag_picos) para los N_TOP_PICOS más prominentes en [F_MIN, F_MAX].

    Usa prominencia espectral (no solo altura) para filtrar picos de ruido
    poco relevantes frente al fondo espectral local.
    """
    mask  = (freqs >= F_MIN) & (freqs <= F_MAX)
    f_roi = freqs[mask]
    m_roi = mag_db[mask]

    if len(m_roi) < 3:
        return np.array([]), np.array([])

    delta_f  = float(f_roi[1] - f_roi[0])
    dist_idx = max(1, int(DIST_MIN_HZ / delta_f))

    idx_all, _ = find_peaks(m_roi, distance=dist_idx)
    if len(idx_all) == 0:
        return np.array([]), np.array([])

    proms = peak_prominences(m_roi, idx_all)[0]

    if len(idx_all) > N_TOP_PICOS:
        top  = np.argsort(proms)[-N_TOP_PICOS:]
        idx  = np.sort(idx_all[top])      # reordenar por frecuencia
    else:
        idx  = idx_all

    return f_roi[idx], m_roi[idx]

# ── Estimación de f₀ ──────────────────────────────────────────────────────────

def frecuencia_fundamental(freqs_picos: np.ndarray) -> float | None:
    """Estima f₀ mediante template matching de armónicos.

    Para cada candidato f_c ∈ freqs_picos se cuenta cuántos picos caen
    cerca de k·f_c (tolerancia relativa TOLERANCIA).  El candidato que
    maximiza ese conteo es f₀; los empates se resuelven eligiendo la
    frecuencia más baja.

    Esto es equivalente al método delta (los armónicos están separados por f₀)
    pero es robusto ante picos espurios de resonancias simpáticas.
    """
    if len(freqs_picos) == 0:
        return None
    if len(freqs_picos) == 1:
        return float(freqs_picos[0])

    mejor_f0    = None
    mejor_score = 0

    for f_c in freqs_picos:
        if f_c < 1.0:
            continue
        score = 0
        for f_p in freqs_picos:
            ratio = f_p / f_c
            if ratio >= 0.9 and abs(ratio - round(ratio)) < TOLERANCIA:
                score += 1
        if score > mejor_score or (
            score == mejor_score and (mejor_f0 is None or f_c < mejor_f0)
        ):
            mejor_score = score
            mejor_f0    = f_c

    return float(mejor_f0) if mejor_f0 is not None else float(freqs_picos[0])

# ── Gráficas ──────────────────────────────────────────────────────────────────

def graficar_espectro(
    freqs: np.ndarray,
    mag_db: np.ndarray,
    f_picos: np.ndarray,
    m_picos: np.ndarray,
    f0: float | None,
    titulo: str,
    ruta: Path,
) -> None:
    mask = (freqs >= F_MIN) & (freqs <= F_MAX)
    fig, ax = plt.subplots(figsize=(10, 4))

    ax.plot(freqs[mask], mag_db[mask], color="steelblue", lw=0.7, label="Espectro")

    if len(f_picos):
        ax.scatter(f_picos, m_picos, color="crimson", zorder=5, s=35, label="Picos")

    if f0 is not None:
        k = 1
        while f0 * k <= F_MAX:
            ax.axvline(f0 * k, color="darkorange", lw=0.7, ls="--", alpha=0.6,
                       label=f"kf₀ (f₀={f0:.1f} Hz)" if k == 1 else None)
            k += 1
        etiqueta = f"f₀ ≈ {f0:.1f} Hz"
    else:
        etiqueta = "f₀ no detectada"

    ax.set_title(f"{titulo}  |  {etiqueta}", fontsize=11)
    ax.set_xlabel("Frecuencia (Hz)")
    ax.set_ylabel("Magnitud (dB)")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(ruta, dpi=100)
    plt.close(fig)


def graficar_resumen_cuerda(
    trastes: list[int],
    f0s: list[float],
    cuerda: int,
    ruta: Path,
) -> None:
    """f₀ vs traste con referencia de temperamento igual (A4 = 440 Hz)."""
    midi0  = MIDI_AL_AIRE.get(cuerda)
    f_ref  = [440.0 * 2 ** ((midi0 + t - 69) / 12) for t in trastes] if midi0 else []

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(trastes, f0s, "o-", color="darkorange", lw=1.5, ms=6, label="Medido")
    if f_ref:
        ax.plot(trastes, f_ref, "s--", color="gray", lw=1.0, ms=4,
                alpha=0.7, label="Referencia temperada")

    ax.set_xlabel("Traste")
    ax.set_ylabel("f₀ (Hz)")
    ax.set_title(f"Cuerda {cuerda} — {NOMBRE_CUERDA.get(cuerda, '')}  |  f₀ por traste")
    ax.set_xticks(trastes)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(ruta, dpi=100)
    plt.close(fig)


def graficar_resumen_global(
    tabla: dict[int, dict[int, float]],
    ruta: Path,
) -> None:
    """Heatmap de f₀ (Hz) para todas las cuerdas y trastes."""
    cuerdas = sorted(tabla.keys())
    matriz  = np.full((len(cuerdas), N_TRASTES), np.nan)

    for i, c in enumerate(cuerdas):
        for t, f0 in tabla[c].items():
            if 0 <= t < N_TRASTES:
                matriz[i, t] = f0

    fig, ax = plt.subplots(figsize=(15, 5))
    im = ax.imshow(matriz, aspect="auto", cmap="viridis", origin="upper")
    plt.colorbar(im, ax=ax, label="f₀ (Hz)")

    ax.set_xlabel("Traste")
    ax.set_ylabel("Cuerda")
    ax.set_xticks(range(N_TRASTES))
    ax.set_xticklabels(range(N_TRASTES))
    ax.set_yticks(range(len(cuerdas)))
    ax.set_yticklabels([f"C{c}  {NOMBRE_CUERDA.get(c,'')}" for c in cuerdas])
    ax.set_title("Frecuencia fundamental f₀ (Hz) — todas las cuerdas y trastes")

    for i in range(len(cuerdas)):
        for j in range(N_TRASTES):
            if not np.isnan(matriz[i, j]):
                ax.text(j, i, f"{matriz[i,j]:.0f}", ha="center", va="center",
                        fontsize=5.5, color="white")

    fig.tight_layout()
    fig.savefig(ruta, dpi=120)
    plt.close(fig)

def graficar_resumen_traste(
    cuerdas: list[int],
    f0s: list[float],
    traste: int,
    ruta: Path,
) -> None:
    """f₀ vs cuerda para un traste dado, con ajuste exponencial y referencia temperada.

    El ajuste se hace en escala log (regresión lineal de log f₀ vs número de cuerda),
    que es equivalente a f₀(c) = A · e^(b·c).  Se reporta R² en escala log.
    """
    c_arr  = np.array(cuerdas, dtype=float)
    f0_arr = np.array(f0s, dtype=float)
    f_ref  = [440.0 * 2 ** ((MIDI_AL_AIRE[c] + traste - 69) / 12) for c in cuerdas]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(c_arr, f0_arr, "o", color="darkorange", ms=8, zorder=5, label="Medido")
    ax.plot(cuerdas, f_ref, "s--", color="gray", lw=1.0, ms=5,
            alpha=0.7, label="Referencia temperada")

    if len(c_arr) >= 2:
        coeffs   = np.polyfit(c_arr, np.log(f0_arr), 1)
        b, log_a = coeffs
        c_fino   = np.linspace(c_arr[0], c_arr[-1], 300)
        f_fit    = np.exp(log_a) * np.exp(b * c_fino)
        log_pred = np.polyval(coeffs, c_arr)
        ss_res   = np.sum((np.log(f0_arr) - log_pred) ** 2)
        ss_tot   = np.sum((np.log(f0_arr) - np.mean(np.log(f0_arr))) ** 2)
        r2       = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
        ax.plot(c_fino, f_fit, "-", color="darkorange", lw=1.5, alpha=0.75,
                label=f"Ajuste exp.  f₀=Ae^(bc)  R²={r2:.4f}")

    titulo_traste = "Cuerda libre (T0)" if traste == 0 else f"Traste {traste}"
    ax.set_xticks(range(1, N_CUERDAS + 1))
    ax.set_xticklabels(
        [f"C{c}\n{NOMBRE_CUERDA[c]}" for c in range(1, N_CUERDAS + 1)], fontsize=7
    )
    ax.set_xlabel("Cuerda")
    ax.set_ylabel("f₀ (Hz)")
    ax.set_title(f"{titulo_traste}  |  f₀ por cuerda")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(ruta, dpi=100)
    plt.close(fig)


def graficar_overlay_trastes(
    tabla: dict[int, dict[int, float]],
    ruta: Path,
) -> None:
    """Todas las curvas f₀ vs cuerda (una por traste) superpuestas en una figura.

    El gradiente de color plasma (T0=oscuro → T19=claro) muestra la progresión
    hacia frecuencias más agudas a medida que se acorta la longitud vibrante.
    """
    cmap = plt.cm.plasma
    fig, ax = plt.subplots(figsize=(9, 5))

    for traste in range(N_TRASTES):
        cuerdas_v = [c for c in range(1, N_CUERDAS + 1) if traste in tabla[c]]
        f0s_v     = [tabla[c][traste] for c in cuerdas_v]
        if len(f0s_v) < 2:
            continue
        color = cmap(traste / (N_TRASTES - 1))
        lw    = 2.2 if traste == 0 else 1.0
        label = "T0 — cuerda libre" if traste == 0 else f"T{traste}"
        ax.plot(cuerdas_v, f0s_v, "o-", color=color, lw=lw, ms=4 if traste else 6,
                label=label, zorder=10 if traste == 0 else None)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, N_TRASTES - 1))
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label="Traste")

    ax.set_xticks(range(1, N_CUERDAS + 1))
    ax.set_xticklabels(
        [f"C{c}\n{NOMBRE_CUERDA[c]}" for c in range(1, N_CUERDAS + 1)], fontsize=7
    )
    ax.set_xlabel("Cuerda")
    ax.set_ylabel("f₀ (Hz)")
    ax.set_title("f₀ vs cuerda — todos los trastes  (T0→T19)")
    # Anotar solo T0 y T19 en la leyenda para no saturar
    handles, labels = ax.get_legend_handles_labels()
    t0_h = [(h, l) for h, l in zip(handles, labels) if "libre" in l]
    if t0_h:
        ax.legend(*zip(*t0_h), fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(ruta, dpi=120)
    plt.close(fig)


# ── Pipeline ──────────────────────────────────────────────────────────────────

def procesar_nota(cuerda: int, traste: int) -> float | None:
    ruta = buscar_archivo(cuerda, traste)
    if ruta is None:
        print(f"  [FALTA]  C{cuerda} T{traste:2d}: archivo no encontrado")
        return None

    try:
        señal, fs = cargar_audio(ruta)
    except Exception as exc:
        print(f"  [ERROR]  C{cuerda} T{traste:2d}: {exc}")
        return None

    seg              = segmento_estacionario(señal, fs)
    freqs, mag_db    = calcular_fft(seg, fs)
    f_picos, m_picos = detectar_picos(freqs, mag_db)
    f0               = frecuencia_fundamental(f_picos)

    if GUARDAR_ESPECTROS:
        graficar_espectro(
            freqs, mag_db, f_picos, m_picos, f0,
            titulo=f"Cuerda {cuerda} — Traste {traste}",
            ruta=FIGURAS_DIR / f"C{cuerda}_T{traste:02d}_espectro.png",
        )

    etiqueta = f"{f0:.1f} Hz" if f0 is not None else "N/A"
    print(f"  C{cuerda} T{traste:2d}:  {len(f_picos):2d} picos  →  f₀ = {etiqueta}")
    return f0


def main() -> None:
    FIGURAS_DIR.mkdir(exist_ok=True)

    # tabla[cuerda][traste] = f₀  (solo entradas válidas)
    tabla: dict[int, dict[int, float]] = {c: {} for c in range(1, N_CUERDAS + 1)}

    for cuerda in range(1, N_CUERDAS + 1):
        print(f"\n─── Cuerda {cuerda}  ({NOMBRE_CUERDA.get(cuerda, '')}) ───")

        for traste in range(N_TRASTES):
            f0 = procesar_nota(cuerda, traste)
            if f0 is not None:
                tabla[cuerda][traste] = f0

        t_vals  = sorted(tabla[cuerda].keys())
        f0_vals = [tabla[cuerda][t] for t in t_vals]
        if len(t_vals) >= 2:
            graficar_resumen_cuerda(
                t_vals, f0_vals, cuerda,
                ruta=FIGURAS_DIR / f"resumen_cuerda{cuerda}.png",
            )

    graficar_resumen_global(tabla, ruta=FIGURAS_DIR / "resumen_global.png")

    # ── Gráficas por traste (f₀ vs cuerda) ──────────────────────────────────
    print("\n─── Generando gráficas por traste ───")
    for traste in range(N_TRASTES):
        cuerdas_v = [c for c in range(1, N_CUERDAS + 1) if traste in tabla[c]]
        f0s_v     = [tabla[c][traste] for c in cuerdas_v]
        if len(cuerdas_v) < 2:
            continue
        graficar_resumen_traste(
            cuerdas_v, f0s_v, traste,
            ruta=FIGURAS_DIR / f"resumen_traste{traste:02d}.png",
        )
        etiq = "cuerda libre" if traste == 0 else f"traste {traste}"
        print(f"  T{traste:2d}: {cuerdas_v}  →  {[f'{f:.0f}' for f in f0s_v]} Hz")

    graficar_overlay_trastes(tabla, ruta=FIGURAS_DIR / "overlay_trastes.png")

    # ── CSV largo: (cuerda, traste, f₀) ─────────────────────────────────────
    csv_largo = Path("frecuencias_fundamentales.csv")
    with open(csv_largo, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["cuerda", "traste", "f0_Hz"])
        for cuerda in range(1, N_CUERDAS + 1):
            for traste in sorted(tabla[cuerda]):
                writer.writerow([cuerda, traste, f"{tabla[cuerda][traste]:.4f}"])

    # ── CSV ancho: filas = trastes, columnas = cuerdas ───────────────────────
    csv_ancho = Path("f0_por_traste.csv")
    with open(csv_ancho, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["traste"] + [f"C{c}_Hz" for c in range(1, N_CUERDAS + 1)])
        for traste in range(N_TRASTES):
            fila = [traste]
            for cuerda in range(1, N_CUERDAS + 1):
                val = tabla[cuerda].get(traste, "")
                fila.append(f"{val:.4f}" if isinstance(val, float) else "")
            writer.writerow(fila)

    total = sum(len(v) for v in tabla.values())
    print(f"\n{'─'*52}")
    print(f"Figuras:    {FIGURAS_DIR}/")
    print(f"CSV largo:  {csv_largo}")
    print(f"CSV ancho:  {csv_ancho}")
    print(f"Procesadas: {total} / {N_CUERDAS * N_TRASTES} notas")


if __name__ == "__main__":
    main()
