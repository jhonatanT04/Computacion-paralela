/**
 * ============================================================================
 * PRÁCTICA 1 - COMPUTACIÓN PARALELA
 * Convolución de Imágenes - Versión PARALELA (GPU - CUDA)
 * 
 * Filtro: Laplaciano (Detección de Bordes)
 * Máscaras: 15x15, 31x31, 65x65
 * 
 * Cada hilo CUDA se encarga de calcular un píxel de salida.
 * Los datos se leen desde la memoria global del device.
 * Se utilizan bloques de 16x16 hilos.
 * 
 * Uso: ./filtro_paralelo <imagen_entrada> <tamaño_mascara> <imagen_salida>
 * Ejemplo: ./filtro_paralelo imagen.jpg 15 resultado_gpu_15.png
 * ============================================================================
 */

#include <opencv2/opencv.hpp>
#include <iostream>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

// Tamaño del bloque de hilos (16x16 = 256 hilos por bloque)
#define BLOCK_SIZE 16

// ============================================================================
// Macro para verificar errores de CUDA
// ============================================================================
#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "Error CUDA: " << cudaGetErrorString(err) \
                  << " en línea " << __LINE__ << std::endl; \
        exit(1); \
    } \
}

// ============================================================================
// Kernel CUDA para Convolución 2D
// ============================================================================
// Cada hilo calcula el valor de un píxel de salida.
// Se accede a la memoria global para leer los píxeles vecinos.
// Se aplica zero-padding para los bordes de la imagen.
// ============================================================================
__global__ void convolucion2D_GPU(const unsigned char* entrada, unsigned char* salida,
                                   int ancho, int alto,
                                   const float* mascara, int tamMascara) {
    // Calcular la posición (x, y) del píxel que este hilo procesará
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Verificar que el hilo está dentro de los límites de la imagen
    if (x >= ancho || y >= alto) return;

    int mitadMascara = tamMascara / 2;
    float suma = 0.0f;

    // Recorrer la ventana de la máscara alrededor del píxel (x, y)
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
    suma = fabsf(suma);
    if (suma > 255.0f) suma = 255.0f;
    salida[y * ancho + x] = (unsigned char)suma;
}

// ============================================================================
// Generación de la máscara Laplaciana normalizada (Host)
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
// Función Principal
// ============================================================================
int main(int argc, char** argv) {
    // Validar argumentos
    if (argc != 4) {
        std::cerr << "============================================" << std::endl;
        std::cerr << "  Convolución Paralela CUDA - Filtro Laplaciano" << std::endl;
        std::cerr << "============================================" << std::endl;
        std::cerr << "Uso: " << argv[0] << " <imagen_entrada> <tamano_mascara> <imagen_salida>" << std::endl;
        std::cerr << "Ejemplo: " << argv[0] << " foto.jpg 15 resultado_gpu_15.png" << std::endl;
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
    int tamImagen = ancho * alto * sizeof(unsigned char);
    int tamMascaraBytes = tamMascara * tamMascara * sizeof(float);

    std::cout << "============================================" << std::endl;
    std::cout << "  CONVOLUCIÓN PARALELA (GPU - CUDA)" << std::endl;
    std::cout << "  Filtro: Laplaciano" << std::endl;
    std::cout << "============================================" << std::endl;
    std::cout << "Imagen: " << rutaEntrada << std::endl;
    std::cout << "Resolución: " << ancho << " x " << alto << std::endl;
    std::cout << "Máscara: " << tamMascara << " x " << tamMascara << std::endl;
    std::cout << "Total píxeles: " << (long)ancho * alto << std::endl;

    // Mostrar información del dispositivo GPU
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
    std::cout << "GPU: " << prop.name << std::endl;
    std::cout << "Bloques: " << BLOCK_SIZE << "x" << BLOCK_SIZE 
              << " (" << BLOCK_SIZE * BLOCK_SIZE << " hilos/bloque)" << std::endl;

    dim3 dimBloque(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid((ancho + BLOCK_SIZE - 1) / BLOCK_SIZE,
                 (alto + BLOCK_SIZE - 1) / BLOCK_SIZE);

    std::cout << "Grid: " << dimGrid.x << " x " << dimGrid.y << " bloques" << std::endl;
    std::cout << "Total hilos: " << (long)dimGrid.x * dimGrid.y * BLOCK_SIZE * BLOCK_SIZE << std::endl;
    std::cout << "--------------------------------------------" << std::endl;

    // ========================================================================
    // 2. Generar máscara Laplaciana en el host
    // ========================================================================
    float* h_mascara = new float[tamMascara * tamMascara];
    generarMascaraLaplaciano(h_mascara, tamMascara);

    // Buffer para la imagen de salida en el host
    unsigned char* h_salida = new unsigned char[ancho * alto];

    // ========================================================================
    // 3. Reservar memoria en el device (GPU)
    // ========================================================================
    unsigned char *d_entrada, *d_salida;
    float *d_mascara;

    CHECK_CUDA(cudaMalloc(&d_entrada, tamImagen));
    CHECK_CUDA(cudaMalloc(&d_salida, tamImagen));
    CHECK_CUDA(cudaMalloc(&d_mascara, tamMascaraBytes));

    // ========================================================================
    // 4. Copiar datos del host al device
    // ========================================================================
    CHECK_CUDA(cudaMemcpy(d_entrada, imagen.data, tamImagen, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_mascara, h_mascara, tamMascaraBytes, cudaMemcpyHostToDevice));

    // ========================================================================
    // 5. Crear eventos CUDA para medir el tiempo
    // ========================================================================
    cudaEvent_t inicio, fin;
    CHECK_CUDA(cudaEventCreate(&inicio));
    CHECK_CUDA(cudaEventCreate(&fin));

    // ========================================================================
    // 6. Ejecutar el kernel y medir tiempo
    // ========================================================================
    CHECK_CUDA(cudaEventRecord(inicio));

    convolucion2D_GPU<<<dimGrid, dimBloque>>>(d_entrada, d_salida,
                                               ancho, alto,
                                               d_mascara, tamMascara);

    CHECK_CUDA(cudaEventRecord(fin));
    CHECK_CUDA(cudaEventSynchronize(fin));

    // Verificar errores del kernel
    CHECK_CUDA(cudaGetLastError());

    float tiempoMs = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&tiempoMs, inicio, fin));

    std::cout << "Tiempo de ejecución GPU: " << tiempoMs << " ms" << std::endl;
    std::cout << "Tiempo de ejecución GPU: " << tiempoMs / 1000.0f << " s" << std::endl;
    std::cout << "--------------------------------------------" << std::endl;

    // ========================================================================
    // 7. Copiar resultado del device al host
    // ========================================================================
    CHECK_CUDA(cudaMemcpy(h_salida, d_salida, tamImagen, cudaMemcpyDeviceToHost));

    // ========================================================================
    // 8. Guardar imagen resultante usando OpenCV
    // ========================================================================
    cv::Mat resultado(alto, ancho, CV_8UC1, h_salida);
    cv::imwrite(rutaSalida, resultado);
    std::cout << "Resultado guardado en: " << rutaSalida << std::endl;
    std::cout << "============================================" << std::endl;

    // ========================================================================
    // 9. Liberar memoria
    // ========================================================================
    CHECK_CUDA(cudaEventDestroy(inicio));
    CHECK_CUDA(cudaEventDestroy(fin));
    CHECK_CUDA(cudaFree(d_entrada));
    CHECK_CUDA(cudaFree(d_salida));
    CHECK_CUDA(cudaFree(d_mascara));
    delete[] h_mascara;
    delete[] h_salida;

    return 0;
}
