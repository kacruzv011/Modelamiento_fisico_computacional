import time

p0 = 0.01; k = 0.002; h = 1.0; steps = 50
iteraciones = 10_000_000 # 10 millones

print(f"Iniciando Benchmark Python ({iteraciones} iteraciones)...")
start = time.time()

for _ in range(iteraciones):
    p = p0
    for i in range(steps):
        p = p + h * k * (1.0 - p)

end = time.time()
print(f"Tiempo Total Python: {end - start:.4f} segundos")