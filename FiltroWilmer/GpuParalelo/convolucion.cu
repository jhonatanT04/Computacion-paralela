#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>

// Declaración en Memoria Constante para el filtro.
// Se reserva el tamaño máximo necesario (65x65 = 4225 elementos) para no superar los 64KB.
__constant__ float d_filtro[4225];

// 1. Función para generar dinámicamente el filtro Motion Blur en el Host
std::vector<float> generarFiltroMotionBlur(int n) {
    std::vector<float> filtro(n * n, 0.0f);
    float valor = 1.0f / static_cast<float>(n);
    for (int i = 0; i < n; ++i) {
        filtro[i * n + i] = valor;
    }
    return filtro;
}

// 2. Función Kernel de Convolución (Device - GPU)
// ADAPTACIÓN 1D: Cada hilo se encarga de un solo píxel utilizando SOLO la dimensión X.
__global__ void convolucionGPU_1D(const unsigned char* input, unsigned char* output, int width, int height, int filterSize) {
    
    // Cálculo de la coordenada global del hilo ÚNICAMENTE en la dimensión X
    long long tid = threadIdx.x + blockIdx.x * blockDim.x;
    long long totalPixels = (long long)width * height;

    // Validación: Evitar que los hilos excedentes procesen fuera de la imagen aplanada
    if (tid < totalPixels) {
        
        // Mapeo del índice unidimensional (1D) a coordenadas bidimensionales (2D)
        int x = tid % width;
        int y = tid / width;
        
        int offset = filterSize / 2;

        // Validación: Ignorar los márgenes para no procesar bordes
        if (x >= offset && x < width - offset && y >= offset && y < height - offset) {
            float suma = 0.0f;

            // Recorrido de la ventana del filtro (Memoria Constante)
            for (int fy = -offset; fy <= offset; ++fy) {
                for (int fx = -offset; fx <= offset; ++fx) {
                    
                    int imgY = y + fy;
                    int imgX = x + fx;
                    int filterY = fy + offset;
                    int filterX = fx + offset;

                    int imgIndex = imgY * width + imgX;
                    int filterIndex = filterY * filterSize + filterX;

                    // Lectura de imagen en Global, lectura de filtro en Constante
                    suma += input[imgIndex] * d_filtro[filterIndex];
                }
            }
            
            // Saturación y escritura
            int valorPixel = static_cast<int>(suma);
            if (valorPixel > 255) valorPixel = 255;
            if (valorPixel < 0) valorPixel = 0;
            
            // Se utiliza el identificador global 1D (tid) para escribir el resultado
            output[tid] = static_cast<unsigned char>(valorPixel);
        }
    }
}

int main() {
    // 3. Lectura de imagen con OpenCV
    cv::Mat imagenOriginal = cv::imread("heic2007a.jpg", cv::IMREAD_GRAYSCALE);
    if (imagenOriginal.empty()) {
        std::cerr << "Error al cargar la imagen." << std::endl;
        return -1;
    }

    int width = imagenOriginal.cols;
    int height = imagenOriginal.rows;
    long long imageSize = (long long)width * height * sizeof(unsigned char);

    cv::Mat imagenSalida = cv::Mat::zeros(height, width, CV_8UC1);

    // Asignación de Memoria Global en el Device
    unsigned char *d_input, *d_output;
    cudaMalloc((void**)&d_input, imageSize);
    cudaMalloc((void**)&d_output, imageSize);

    // Copia de la imagen original al Device
    cudaMemcpy(d_input, imagenOriginal.ptr<unsigned char>(0), imageSize, cudaMemcpyHostToDevice);

    std::vector<int> tamanosFiltro = {9, 21, 31};

    // ADAPTACIÓN 1D: Definición de la estructura de ejecución en dimensión X
    int hilosPorBloque = 256;
    long long totalPixeles = (long long)width * height;
    int bloquesPorGrid = (totalPixeles + hilosPorBloque - 1) / hilosPorBloque;

    // Creación de eventos CUDA para medir el tiempo en ms
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    for (int size : tamanosFiltro) {
        std::cout << "Evaluando GPU con filtro Motion Blur de " << size << "x" << size << "..." << std::endl;

        // Generar filtro en CPU
        std::vector<float> filtroHost = generarFiltroMotionBlur(size);
        int bytesFiltro = size * size * sizeof(float);

        // Copiar el filtro a la Memoria Constante del Device
        cudaMemcpyToSymbol(d_filtro, filtroHost.data(), bytesFiltro);

        // Limpiar el buffer de salida en la GPU antes de la nueva ejecución
        cudaMemset(d_output, 0, imageSize);

        // Registrar inicio, ejecutar kernel y registrar fin
        cudaEventRecord(start);
        
        // Lanzamiento del kernel utilizando enteros 1D en lugar de dim3
        convolucionGPU_1D<<<bloquesPorGrid, hilosPorBloque>>>(d_input, d_output, width, height, size);
        
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        // Cálculo del tiempo en ms
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);

        std::cout << "Tiempo de ejecucion en GPU (" << size << "x" << size << "): " 
                  << milliseconds << " ms\n" << std::endl;

        // Recuperar la imagen procesada al Host
        cudaMemcpy(imagenSalida.ptr<unsigned char>(0), d_output, imageSize, cudaMemcpyDeviceToHost);

        std::string nombreArchivo = "resultado_gpu_" + std::to_string(size) + ".jpg";
        cv::imwrite(nombreArchivo, imagenSalida);
    }

    // Limpieza de recursos
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}