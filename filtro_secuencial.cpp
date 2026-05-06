/**
 * ============================================================================
 * PRÁCTICA 1 - COMPUTACIÓN PARALELA
 * Convolución de Imágenes - Versión SECUENCIAL (CPU)
 * 
 * Filtro: Laplaciano (Detección de Bordes)
 * Máscaras: 15x15, 31x31, 65x65
 * 
 * Uso: ./filtro_secuencial <imagen_entrada> <tamaño_mascara> <imagen_salida>
 * Ejemplo: ./filtro_secuencial imagen.jpg 15 resultado_cpu_15.png
 * ============================================================================
 */

#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>
#include <cstdlib>
#include <cmath>
#include <cstring>

// ============================================================================
// Generación de la máscara Laplaciana normalizada
// ============================================================================
// La máscara Laplaciana detecta bordes calculando la diferencia entre
// el valor del píxel central y el promedio de sus vecinos.
// Todos los elementos = 1/(N*N - 1), centro = -1.0
// La suma total de la máscara es 0 (filtro pasa-altos).
// ============================================================================
void generarMascaraLaplaciano(float* mascara, int tamano) {
    int totalElementos = tamano * tamano;
    float valorVecino = 1.0f / (float)(totalElementos - 1);

    // Llenar toda la máscara con el valor normalizado
    for (int i = 0; i < totalElementos; i++) {
        mascara[i] = valorVecino;
    }

    // El centro tiene valor negativo para detectar bordes
    int centro = tamano / 2;
    mascara[centro * tamano + centro] = -1.0f;
}

// ============================================================================
// Función de Convolución 2D Secuencial (CPU)
// ============================================================================
// Recorre cada píxel de la imagen y aplica la convolución con la máscara.
// Usa zero-padding en los bordes (los píxeles fuera de la imagen = 0).
// ============================================================================
void convolucion2D_CPU(const unsigned char* entrada, unsigned char* salida,
                       int ancho, int alto,
                       const float* mascara, int tamMascara) {
    int mitadMascara = tamMascara / 2;

    for (int y = 0; y < alto; y++) {
        for (int x = 0; x < ancho; x++) {
            float suma = 0.0f;

            // Recorrer la ventana de la máscara
            for (int ky = -mitadMascara; ky <= mitadMascara; ky++) {
                for (int kx = -mitadMascara; kx <= mitadMascara; kx++) {
                    int px = x + kx;
                    int py = y + ky;

                    // Zero-padding: píxeles fuera de la imagen se consideran 0
                    if (px >= 0 && px < ancho && py >= 0 && py < alto) {
                        float valorPixel = (float)entrada[py * ancho + px];
                        float valorMascara = mascara[(ky + mitadMascara) * tamMascara + (kx + mitadMascara)];
                        suma += valorPixel * valorMascara;
                    }
                }
            }

            // Tomar valor absoluto y limitar al rango [0, 255]
            suma = fabs(suma);
            if (suma > 255.0f) suma = 255.0f;
            salida[y * ancho + x] = (unsigned char)suma;
        }
    }
}

// ============================================================================
// Función Principal
// ============================================================================
int main(int argc, char** argv) {
    // Validar argumentos
    if (argc != 4) {
        std::cerr << "============================================" << std::endl;
        std::cerr << "  Convolución Secuencial - Filtro Laplaciano" << std::endl;
        std::cerr << "============================================" << std::endl;
        std::cerr << "Uso: " << argv[0] << " <imagen_entrada> <tamano_mascara> <imagen_salida>" << std::endl;
        std::cerr << "Ejemplo: " << argv[0] << " foto.jpg 15 resultado_cpu_15.png" << std::endl;
        std::cerr << "Máscaras sugeridas: 15, 31, 65" << std::endl;
        return 1;
    }

    const char* rutaEntrada = argv[1];
    int tamMascara = atoi(argv[2]);
    const char* rutaSalida = argv[3];

    // Validar que la máscara sea impar
    if (tamMascara % 2 == 0 || tamMascara < 3) {
        std::cerr << "Error: El tamaño de la máscara debe ser impar y >= 3." << std::endl;
        return 1;
    }

    // ========================================================================
    // 1. Cargar imagen en escala de grises usando OpenCV
    // ========================================================================
    cv::Mat imagen = cv::imread(rutaEntrada, cv::IMREAD_GRAYSCALE);
    if (imagen.empty()) {
        std::cerr << "Error: No se pudo cargar la imagen: " << rutaEntrada << std::endl;
        return 1;
    }

    int ancho = imagen.cols;
    int alto = imagen.rows;

    std::cout << "============================================" << std::endl;
    std::cout << "  CONVOLUCIÓN SECUENCIAL (CPU)" << std::endl;
    std::cout << "  Filtro: Laplaciano" << std::endl;
    std::cout << "============================================" << std::endl;
    std::cout << "Imagen: " << rutaEntrada << std::endl;
    std::cout << "Resolución: " << ancho << " x " << alto << std::endl;
    std::cout << "Máscara: " << tamMascara << " x " << tamMascara << std::endl;
    std::cout << "Total píxeles: " << (long)ancho * alto << std::endl;
    std::cout << "--------------------------------------------" << std::endl;

    // ========================================================================
    // 2. Generar máscara Laplaciana
    // ========================================================================
    float* mascara = new float[tamMascara * tamMascara];
    generarMascaraLaplaciano(mascara, tamMascara);

    // ========================================================================
    // 3. Reservar memoria para la imagen de salida
    // ========================================================================
    unsigned char* salida = new unsigned char[ancho * alto];

    // ========================================================================
    // 4. Ejecutar convolución y medir tiempo
    // ========================================================================
    auto inicio = std::chrono::high_resolution_clock::now();

    convolucion2D_CPU(imagen.data, salida, ancho, alto, mascara, tamMascara);

    auto fin = std::chrono::high_resolution_clock::now();

    double tiempoMs = std::chrono::duration<double, std::milli>(fin - inicio).count();

    std::cout << "Tiempo de ejecución CPU: " << tiempoMs << " ms" << std::endl;
    std::cout << "Tiempo de ejecución CPU: " << tiempoMs / 1000.0 << " s" << std::endl;
    std::cout << "--------------------------------------------" << std::endl;

    // ========================================================================
    // 5. Guardar imagen resultante usando OpenCV
    // ========================================================================
    cv::Mat resultado(alto, ancho, CV_8UC1, salida);
    cv::imwrite(rutaSalida, resultado);
    std::cout << "Resultado guardado en: " << rutaSalida << std::endl;
    std::cout << "============================================" << std::endl;

    // ========================================================================
    // 6. Liberar memoria
    // ========================================================================
    delete[] mascara;
    delete[] salida;

    return 0;
}
