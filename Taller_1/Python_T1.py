import numpy as np
import matplotlib.pyplot as plt
import time

# --- Parámetros ---
p0 = 0.01
b = 0.02
r = 0.1
k = r * b       # k = 0.002
h = 1.0         # Paso de tiempo
t_max = 50
steps = int(t_max / h)

t_vals = np.linspace(0, t_max, steps + 1)

# --- 1. Solución Exacta ---
def get_exact(t_arr):
    return 1 - (1 - p0) * np.exp(-k * t_arr)

p_exact = get_exact(t_vals)

# --- 2. Método de Euler ---
p_euler = np.zeros(len(t_vals))
p_euler[0] = p0
for i in range(steps):
    p_euler[i+1] = p_euler[i] + h * k * (1 - p_euler[i])

# --- 3. Método de Taylor Orden 2 ---
p_taylor = np.zeros(len(t_vals))
p_taylor[0] = p0
factor_taylor = (h * k) - (0.5 * (h**2) * (k**2))
for i in range(steps):
    p_taylor[i+1] = p_taylor[i] + (1 - p_taylor[i]) * factor_taylor

# --- 4. Método del Trapecio ---
p_trap = np.zeros(len(t_vals))
p_trap[0] = p0
hk2 = (h * k) / 2.0
for i in range(steps):
    numerator = p_trap[i] + hk2 * (2 - p_trap[i])
    p_trap[i+1] = numerator / (1 + hk2)

# --- CÁLCULO DE ERRORES ABSOLUTOS ---
err_euler = np.abs(p_exact - p_euler)
err_taylor = np.abs(p_exact - p_taylor)
err_trap = np.abs(p_exact - p_trap)

# --- Gráfica Comparativa y de Errores ---
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

# Subplot 1: Soluciones
ax1.plot(t_vals, p_exact, 'k', label='Exacta', linewidth=2)
ax1.plot(t_vals, p_euler, 'o--', label='Euler', markersize=4, alpha=0.8)
ax1.plot(t_vals, p_taylor, 's:', label='Taylor 2', markersize=4, alpha=0.8)
ax1.plot(t_vals, p_trap, 'x-.', label='Trapecio', markersize=5, alpha=0.8)
ax1.set_ylabel("Proporción p(t)")
ax1.set_title("Comparación de Métodos Numéricos")
ax1.legend()
ax1.grid(True, alpha=0.3)

# Subplot 2: Errores (Escala logarítmica para ver la diferencia real)
ax2.plot(t_vals, err_euler, 'o--', label='Error Euler', markersize=4)
ax2.plot(t_vals, err_taylor, 's:', label='Error Taylor 2', markersize=4)
ax2.plot(t_vals, err_trap, 'x-.', label='Error Trapecio', markersize=5)
ax2.set_yscale('log') # Escala logarítmica esencial para errores
ax2.set_xlabel("Tiempo (años)")
ax2.set_ylabel("Error Absoluto (log)")
ax2.set_title("Evolución del Error Absoluto")
ax2.legend()
ax2.grid(True, which="both", alpha=0.3)

plt.tight_layout()
plt.savefig("analisis_metodos.png", dpi=300)
plt.show()

# Imprimir resumen de errores finales
print(f"{'Método':<12} | {'Valor p(50)':<12} | {'Error Absoluto':<15}")
print("-" * 45)
print(f"{'Euler':<12} | {p_euler[-1]:.8f} | {err_euler[-1]:.2e}")
print(f"{'Taylor 2':<12} | {p_taylor[-1]:.8f} | {err_taylor[-1]:.2e}")
print(f"{'Trapecio':<12} | {p_trap[-1]:.8f} | {err_trap[-1]:.2e}")