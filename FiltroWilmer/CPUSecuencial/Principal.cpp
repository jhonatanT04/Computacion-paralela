#include <iostream>
#include <vector>
#include <chrono>
#include <opencv2/opencv.hpp>

// 1. Función para generar dinámicamente el filtro Motion Blur
// Crea un arreglo 1D que simula una matriz NxN con valores en la diagonal.
std::vector<float> generarFiltroMotionBlur(int n) {
    std::vector<float> filtro(n * n, 0.0f); // Inicializa todo en 0
    float valor = 1.0f / static_cast<float>(n);
    
    for (int i = 0; i < n; ++i) {
        filtro[i * n + i] = valor; // Asigna el valor solo a la diagonal principal
    }
    return filtro;
}

// 2. Función de Convolución Secuencial (Host - 1 Hilo)
void convolucionCPU(const unsigned char* input, unsigned char* output, int width, int height, const std::vector<float>& filter, int filterSize) {
    int offset = filterSize / 2;

    // Los bucles inician y terminan ignorando el margen (offset) de la imagen
    for (int y = offset; y < height - offset; ++y) {
        for (int x = offset; x < width - offset; ++x) {
            float suma = 0.0f;

            // Recorrido de la ventana del filtro
            for (int fy = -offset; fy <= offset; ++fy) {
                for (int fx = -offset; fx <= offset; ++fx) {
                    
                    // Coordenadas absolutas en la imagen
                    int imgY = y + fy;
                    int imgX = x + fx;
                    
                    // Coordenadas absolutas en el filtro (de 0 a filterSize-1)
                    int filterY = fy + offset;
                    int filterX = fx + offset;

                    // Mapeo a índices 1D
                    int imgIndex = imgY * width + imgX;
                    int filterIndex = filterY * filterSize + filterX;

                    suma += input[imgIndex] * filter[filterIndex];
                }
            }
            
            // Saturación y asignación al píxel central
            int valorPixel = static_cast<int>(suma);
            if (valorPixel > 255) valorPixel = 255;
            if (valorPixel < 0) valorPixel = 0;
            
            output[y * width + x] = static_cast<unsigned char>(valorPixel);
        }
    }
}

int main() {
    // 3. Lectura de imagen con OpenCV (Escala de grises para simplificar memoria de 1 canal)
    cv::Mat imagenOriginal = cv::imread("heic2007a.jpg", cv::IMREAD_GRAYSCALE);
    if (imagenOriginal.empty()) {
        std::cerr << "Error al cargar la imagen." << std::endl;
        return -1;
    }

    int width = imagenOriginal.cols;
    int height = imagenOriginal.rows;

    // Crear imagen de salida vacía (inicializada en negro/0)
    cv::Mat imagenSalida = cv::Mat::zeros(height, width, CV_8UC1);

    // Tamaños de ventana a evaluar
    std::vector<int> tamanosFiltro = {65};//9, 21, 31

    for (int size : tamanosFiltro) {
        std::cout << "Evaluando filtro Motion Blur de " << size << "x" << size << "..." << std::endl;

        // Generar filtro dinámico
        std::vector<float> filtro = generarFiltroMotionBlur(size);

        // Punteros directos a los datos puros para emular comportamiento de memoria en CUDA
        const unsigned char* d_input = imagenOriginal.ptr<unsigned char>(0);
        unsigned char* d_output = imagenSalida.ptr<unsigned char>(0);

        // Iniciar medición de tiempo
        auto inicio = std::chrono::high_resolution_clock::now();

        // Ejecución secuencial pura
        convolucionCPU(d_input, d_output, width, height, filtro, size);

        // Finalizar medición de tiempo
        auto fin = std::chrono::high_resolution_clock::now();
        
        // Cálculo del tiempo en ms
        std::chrono::duration<double, std::milli> tiempo_ms = fin - inicio;

        std::cout << "Tiempo de ejecucion (" << size << "x" << size << "): " 
                  << tiempo_ms.count() << " ms\n" << std::endl;

        // Guardar resultado individual (Opcional, verifica los bordes negros)
        std::string nombreArchivo = "resultado_cpu_" + std::to_string(size) + ".jpg";
        cv::imwrite(nombreArchivo, imagenSalida);
    }

    return 0;
}