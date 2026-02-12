import matplotlib.pyplot as plt
import numpy as np

# --- ¡PON TUS DATOS AQUÍ! ---
# Reemplaza estos números con los que te salieron en la terminal
tiempo_python = 52.0193   # Ejemplo (pon el tuyo real)
tiempo_cpp = 2.568385      # Ejemplo
tiempo_fortran = 2.5723039999999999   # Ejemplo

lenguajes = ['Python', 'C++', 'Fortran']
tiempos = [tiempo_python, tiempo_cpp, tiempo_fortran]
colores = ['#3776ab', '#00599C', '#734f96'] # Colores oficiales aprox

plt.figure(figsize=(8, 6))

# Crear gráfico de barras
barras = plt.bar(lenguajes, tiempos, color=colores)

# Añadir etiquetas de valor encima de las barras
for barra in barras:
    yval = barra.get_height()
    plt.text(barra.get_x() + barra.get_width()/2, yval + (0.05 * max(tiempos)), 
             f'{yval:.2f} s', ha='center', va='bottom', fontweight='bold')

plt.ylabel('Tiempo de Ejecución (segundos)\n(Menor es mejor)')
plt.title(f'Comparación de Rendimiento\n(10 millones de simulaciones)')
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Escala logarítmica opcional si la diferencia es abismal
# plt.yscale('log') 

plt.tight_layout()
plt.savefig("benchmark_tiempos.png", dpi=300)
plt.show()