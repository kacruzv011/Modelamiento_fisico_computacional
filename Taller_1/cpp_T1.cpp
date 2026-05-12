#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <iomanip>

using namespace std;

// Función para el método de Euler
// Se mantiene inline para máxima velocidad
int main() {
    // Parámetros
    double p0 = 0.01;
    double b = 0.02;
    double r = 0.1;
    double k = r * b; // 0.002
    double h = 1.0;
    int steps = 50;   // t_max / h
    
    // Carga de trabajo masiva para el benchmark
    // 10 millones de simulaciones completas
    long long iteraciones = 10000000; 

    cout << "Iniciando Benchmark C++ (" << iteraciones << " iteraciones)..." << endl;

    // Iniciar cronómetro
    auto start = chrono::high_resolution_clock::now();

    double p_final = 0.0;

    // Bucle masivo
    for (long long j = 0; j < iteraciones; j++) {
        double p = p0;
        // Simulación individual (Euler)
        for (int i = 0; i < steps; i++) {
            p = p + h * k * (1.0 - p);
        }
        // Guardamos el último resultado para evitar que el compilador elimine el bucle
        if (j == iteraciones - 1) p_final = p; 
    }

    // Parar cronómetro
    auto end = chrono::high_resolution_clock::now();
    
    // Calcular duración en segundos
    chrono::duration<double> duration = end - start;

    cout << fixed << setprecision(6);
    cout << "--------------------------------------" << endl;
    cout << "Valor p(50) final: " << p_final << endl;
    cout << "Tiempo Total C++ : " << duration.count() << " segundos" << endl;
    cout << "--------------------------------------" << endl;

    return 0;
}