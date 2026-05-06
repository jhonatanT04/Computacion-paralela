#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <chrono>
#include <cstdlib>
#include <cmath>

// Genera un filtro de Media (Box Blur). Todos los pesos son iguales (1/N^2)
// Produce un efecto de suavizado/desenfoque.
void generarFiltroMedia(float* mascara, int tamano) {
    int total = tamano * tamano;
    float val = 1.0f / (float)total;
    for(int i = 0; i < total; i++) {
        mascara[i] = val;
    }
}

// Genera un filtro Laplaciano normalizado.
// Centro es negativo, rodeado de pesos positivos uniformes para detectar cambios bruscos (bordes).
void generarFiltroBordes(float* mascara, int tamano) {
    int total = tamano * tamano;
    float val = 1.0f / (float)(total - 1);
    for(int i = 0; i < total; i++) {
        mascara[i] = val;
    }
    int centro = tamano / 2;
    mascara[centro * tamano + centro] = -1.0f;
}

// Genera un filtro de Enfoque (Sharpen).
// Centro fuertemente positivo y bordes negativos para resaltar detalles finos.
void generarFiltroEnfoque(float* mascara, int tamano) {
    int total = tamano * tamano;
    float valExterior = -1.0f / (float)(total - 1);
    for(int i = 0; i < total; i++) {
        mascara[i] = valExterior;
    }
    int centro = tamano / 2;
    mascara[centro * tamano + centro] = 2.0f; 
}

// Aplica la convolución secuencial iterando píxel por píxel
void convolucion_CPU(const unsigned char* entrada, unsigned char* salida,
                     int ancho, int alto, const float* mascara, int tamMascara) {
    int mitadMascara = tamMascara / 2;

    for (int y = 0; y < alto; y++) {
        for (int x = 0; x < ancho; x++) {
            float suma = 0.0f;

            // Iteración sobre la ventana de la máscara
            for (int ky = -mitadMascara; ky <= mitadMascara; ky++) {
                for (int kx = -mitadMascara; kx <= mitadMascara; kx++) {
                    int px = x + kx;
                    int py = y + ky;

                    // Zero-padding para bordes de la imagen
                    if (px >= 0 && px < ancho && py >= 0 && py < alto) {
                        suma += (float)entrada[py * ancho + px] * mascara[(ky + mitadMascara) * tamMascara + (kx + mitadMascara)];
                    }
                }
            }
            suma = fabs(suma);
            salida[y * ancho + x] = (unsigned char)(suma > 255.0f ? 255.0f : suma);
        }
    }
}

int main(int argc, char** argv) {
    if (argc != 4) {
        fprintf(stderr, "Uso: %s <imagen> <id_filtro(1=Media,2=Bordes,3=Enfoque)> <tam_mascara_impar>\n", argv[0]);
        return 1;
    }

    const char* rutaEntrada = argv[1];
    int idFiltro = atoi(argv[2]);
    int tamMascara = atoi(argv[3]);

    if (tamMascara % 2 == 0 || tamMascara < 3 || idFiltro < 1 || idFiltro > 3) return 1;

    cv::Mat imagen = cv::imread(rutaEntrada, cv::IMREAD_GRAYSCALE);
    if (imagen.empty()) return 1;

    int ancho = imagen.cols;
    int alto = imagen.rows;

    float* h_mascara = new float[tamMascara * tamMascara];
    unsigned char* h_salidaCPU = new unsigned char[ancho * alto];

    if(idFiltro == 1) generarFiltroMedia(h_mascara, tamMascara);
    else if(idFiltro == 2) generarFiltroBordes(h_mascara, tamMascara);
    else generarFiltroEnfoque(h_mascara, tamMascara);

    auto inicioCPU = std::chrono::high_resolution_clock::now();
    convolucion_CPU(imagen.data, h_salidaCPU, ancho, alto, h_mascara, tamMascara);
    auto finCPU = std::chrono::high_resolution_clock::now();
    
    double tiempoCPU_ms = std::chrono::duration<double, std::milli>(finCPU - inicioCPU).count();
    printf("CPU (%dx%d | Máscara %dx%d) - Tiempo: %.2f ms (%.2f s)\n", ancho, alto, tamMascara, tamMascara, tiempoCPU_ms, tiempoCPU_ms / 1000.0);

    std::string baseOut = (idFiltro==1?"media":idFiltro==2?"bordes":"enfoque");
    cv::imwrite(baseOut + "_cpu_mask" + std::to_string(tamMascara) + ".png", cv::Mat(alto, ancho, CV_8UC1, h_salidaCPU));

    delete[] h_mascara;
    delete[] h_salidaCPU;

    return 0;
}
