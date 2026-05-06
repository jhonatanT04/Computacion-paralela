#include <cmath>
#include <cstdlib>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <stdio.h>

#define BLOCK_SIZE_1D 256

// Genera un filtro de Media (Box Blur). Todos los pesos son iguales (1/N^2)
// Produce un efecto de suavizado/desenfoque.
void generarFiltroMedia(float *mascara, int tamano) {
  int total = tamano * tamano;
  float val = 1.0f / (float)total;
  for (int i = 0; i < total; i++)
    mascara[i] = val;
}

// Genera un filtro Laplaciano normalizado.
// Centro es negativo, rodeado de pesos positivos uniformes para detectar
// cambios bruscos (bordes).
void generarFiltroBordes(float *mascara, int tamano) {
  int total = tamano * tamano;
  float val = 1.0f / (float)(total - 1);
  for (int i = 0; i < total; i++)
    mascara[i] = val;
  mascara[(tamano / 2) * tamano + (tamano / 2)] = -1.0f;
}

// Genera un filtro de Enfoque (Sharpen).
// Centro fuertemente positivo y bordes negativos para resaltar detalles finos.
void generarFiltroEnfoque(float *mascara, int tamano) {
  int total = tamano * tamano;
  float valExterior = -1.0f / (float)(total - 1);
  for (int i = 0; i < total; i++)
    mascara[i] = valExterior;
  mascara[(tamano / 2) * tamano + (tamano / 2)] = 2.0f;
}

// Kernel de convolución usando únicamente hilos y bloques en dimensión X
// (Requerimiento PDF). Mapea el ID global lineal 'tid' a coordenadas 2D (x, y).
__global__ void convolucion_GPU(const unsigned char *entrada,
                                unsigned char *salida, int ancho, int alto,
                                const float *mascara, int tamMascara) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  if (tid >= ancho * alto)
    return;

  int x = tid % ancho;
  int y = tid / ancho;

  int mitadMascara = tamMascara / 2;
  float suma = 0.0f;

  for (int ky = -mitadMascara; ky <= mitadMascara; ky++) {
    for (int kx = -mitadMascara; kx <= mitadMascara; kx++) {
      int px = x + kx;
      int py = y + ky;

      // Zero-padding
      if (px >= 0 && px < ancho && py >= 0 && py < alto) {
        suma += (float)entrada[py * ancho + px] *
                mascara[(ky + mitadMascara) * tamMascara + (kx + mitadMascara)];
      }
    }
  }
  suma = fabsf(suma);
  salida[tid] = (unsigned char)(suma > 255.0f ? 255.0f : suma);
}

int main(int argc, char **argv) {
  if (argc != 4) {
    fprintf(stderr,
            "Uso: %s <imagen> <id_filtro(1=Media,2=Bordes,3=Enfoque)> "
            "<tam_mascara_impar>\n",
            argv[0]);
    return 1;
  }

  const char *rutaEntrada = argv[1];
  int idFiltro = atoi(argv[2]);
  int tamMascara = atoi(argv[3]);

  if (tamMascara % 2 == 0 || tamMascara < 3 || idFiltro < 1 || idFiltro > 3)
    return 1;

  cv::Mat imagen = cv::imread(rutaEntrada, cv::IMREAD_GRAYSCALE);
  if (imagen.empty())
    return 1;

  int ancho = imagen.cols;
  int alto = imagen.rows;
  int tamImagenBytes = ancho * alto * sizeof(unsigned char);
  int tamMascaraBytes = tamMascara * tamMascara * sizeof(float);

  float *h_mascara = new float[tamMascara * tamMascara];
  unsigned char *h_salidaGPU = new unsigned char[ancho * alto];

  if (idFiltro == 1)
    generarFiltroMedia(h_mascara, tamMascara);
  else if (idFiltro == 2)
    generarFiltroBordes(h_mascara, tamMascara);
  else
    generarFiltroEnfoque(h_mascara, tamMascara);

  unsigned char *d_entrada, *d_salida;
  float *d_mascara;

  cudaMalloc(&d_entrada, tamImagenBytes);
  cudaMalloc(&d_salida, tamImagenBytes);
  cudaMalloc(&d_mascara, tamMascaraBytes);

  cudaMemcpy(d_entrada, imagen.data, tamImagenBytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_mascara, h_mascara, tamMascaraBytes, cudaMemcpyHostToDevice);

  // Configuración 1D para el kernel
  int totalPixeles = ancho * alto;
  dim3 dimBloque(BLOCK_SIZE_1D);
  dim3 dimGrid((totalPixeles + BLOCK_SIZE_1D - 1) / BLOCK_SIZE_1D);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);
  convolucion_GPU<<<dimGrid, dimBloque>>>(d_entrada, d_salida, ancho, alto,
                                          d_mascara, tamMascara);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  float tiempoGPU_ms = 0;
  cudaEventElapsedTime(&tiempoGPU_ms, start, stop);

  cudaMemcpy(h_salidaGPU, d_salida, tamImagenBytes, cudaMemcpyDeviceToHost);

  printf("GPU (%dx%d | Máscara %dx%d) - Tiempo: %.2f ms (%.2f s)\n", ancho,
         alto, tamMascara, tamMascara, tiempoGPU_ms, tiempoGPU_ms / 1000.0);

  std::string baseOut = (idFiltro == 1   ? "media"
                         : idFiltro == 2 ? "bordes"
                                         : "enfoque");
  cv::imwrite(baseOut + "_gpu_mask" + std::to_string(tamMascara) + ".png",
              cv::Mat(alto, ancho, CV_8UC1, h_salidaGPU));

  cudaFree(d_entrada);
  cudaFree(d_salida);
  cudaFree(d_mascara);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  delete[] h_mascara;
  delete[] h_salidaGPU;

  return 0;
}
