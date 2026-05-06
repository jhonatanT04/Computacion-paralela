#include <opencv2/opencv.hpp>
#include <stdio.h>

int main(int argc, char** argv) {
    if (argc != 3) {
        printf("Uso: %s <imagen_cpu.png> <imagen_gpu.png>\n", argv[0]);
        return 1;
    }

    cv::Mat imgCPU = cv::imread(argv[1], cv::IMREAD_GRAYSCALE);
    cv::Mat imgGPU = cv::imread(argv[2], cv::IMREAD_GRAYSCALE);

    if (imgCPU.empty() || imgGPU.empty()) {
        printf("Error: No se pudieron cargar las imagenes.\n");
        return 1;
    }

    if (imgCPU.rows != imgGPU.rows || imgCPU.cols != imgGPU.cols) {
        printf("Error: Las imagenes tienen distinto tamaño.\n");
        return 1;
    }

    cv::Mat diff;
    cv::absdiff(imgCPU, imgGPU, diff);

    // Debido a las diferencias de precision de coma flotante entre CPU y GPU,
    // es normal que haya variaciones de +/- 1 en el valor del pixel final.
    // Aplicamos un umbral para ignorar diferencias de 1.
    cv::Mat diff_umbral;
    cv::threshold(diff, diff_umbral, 1, 255, cv::THRESH_BINARY);

    int diferencias = cv::countNonZero(diff_umbral);

    if (diferencias == 0) {
        printf("==================================================\n");
        printf("¡COMPROBACION EXITOSA!\n");
        printf("Los resultados de la funcion del Host (CPU) \n");
        printf("y la funcion del Device (GPU) son EXACTAMENTE IGUALES.\n");
        printf("==================================================\n");
    } else {
        printf("ERROR: Hay %d pixeles diferentes entre los resultados de CPU y GPU.\n", diferencias);
    }

    return 0;
}
