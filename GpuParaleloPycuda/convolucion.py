import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np
import cv2
import time

# Kernel CUDA para convolución 1D con filtro Motion Blur
codigo_kernel = """
__constant__ float d_filtro[15625];

__global__ void convolucionGPU_1D(const unsigned char* input, unsigned char* output, int width, int height, int filterSize) {
    
    long long tid = threadIdx.x + blockIdx.x * blockDim.x;
    long long totalPixels = (long long)width * height;

    if (tid < totalPixels) {
        int x = tid % width;
        int y = tid / width;
        int offset = filterSize / 2;

        if (x >= offset && x < width - offset && y >= offset && y < height - offset) {
            float suma = 0.0f;

            for (int fy = -offset; fy <= offset; ++fy) {
                for (int fx = -offset; fx <= offset; ++fx) {
                    
                    int imgY = y + fy;
                    int imgX = x + fx;
                    int filterY = fy + offset;
                    int filterX = fx + offset;

                    int imgIndex = imgY * width + imgX;
                    int filterIndex = filterY * filterSize + filterX;

                    suma += input[imgIndex] * d_filtro[filterIndex];
                }
            }
            
            int valorPixel = (int)suma;
            if (valorPixel > 255) valorPixel = 255;
            if (valorPixel < 0) valorPixel = 0;
            
            output[tid] = (unsigned char)valorPixel;
        }
    }
}
"""

# Compilar el módulo de CUDA
mod = SourceModule(codigo_kernel)
convolucionGPU_1D = mod.get_function("convolucionGPU_1D")

# Obtener la referencia a la memoria constante del filtro
d_filtro_ptr, _ = mod.get_global("d_filtro")

#  Función para generar el filtro en Python (El Arquitecto)
def generarFiltroMotionBlur(n):
    # Creamos una matriz llena de ceros
    filtro = np.zeros((n, n), dtype=np.float32)
    valor = 1.0 / n
    # Llenamos la diagonal principal
    np.fill_diagonal(filtro, valor)
    # La aplanamos a 1D para mandarla a la GPU
    return filtro.flatten()


if __name__ == "__main__":
    # Leer imagen con OpenCV en Python
    imagenOriginal = cv2.imread("heic2007a.jpg", cv2.IMREAD_GRAYSCALE)
    
    if imagenOriginal is None:
        print("Error al cargar la imagen.")
        exit()

    height, width = imagenOriginal.shape
    # Aplanamos la imagen para mandarla al Kernel 1D
    imagen_plana = imagenOriginal.flatten().astype(np.uint8)

    tamanosFiltro = [21, 75, 125]

    # Configuración de hilos y bloques
    hilosPorBloque = 256
    totalPixeles = width * height
    bloquesPorGrid = (totalPixeles + hilosPorBloque - 1) // hilosPorBloque

    for size in tamanosFiltro:
        print(f"Evaluando GPU con filtro Motion Blur de {size}x{size}...")

        # Generar filtro
        filtroHost = generarFiltroMotionBlur(size)

        # Copiar filtro a la MEMORIA CONSTANTE usando PyCUDA
        cuda.memcpy_htod(d_filtro_ptr, filtroHost)

        # Preparar arreglo vacío para el resultado
        imagenSalida_plana = np.zeros_like(imagen_plana)

        # Variables necesarias para el Kernel (convertidas a tipos C)
        width_c = np.int32(width)
        height_c = np.int32(height)
        size_c = np.int32(size)

        # Crear eventos para medir tiempo preciso en GPU
        start = cuda.Event()
        stop = cuda.Event()

        start.record()
        
        # Llamar al kernel CUDA para convolución 1D
        # Usamos cuda.In y cuda.Out que hacen el malloc y memcpy automáticamente
        convolucionGPU_1D(
            cuda.In(imagen_plana), 
            cuda.Out(imagenSalida_plana), 
            width_c, height_c, size_c,
            block=(hilosPorBloque, 1, 1), 
            grid=(bloquesPorGrid, 1)
        )
        
        stop.record()
        stop.synchronize()

        # Obtener tiempo en milisegundos
        tiempo_ms = start.time_till(stop)
        print(f"Tiempo de ejecucion en GPU ({size}x{size}): {tiempo_ms:.2f} ms\n")

        # Reconstruir la imagen 2D desde el arreglo plano
        imagenSalida = imagenSalida_plana.reshape((height, width))

        # Guardar la imagen
        nombreArchivo = f"resultado_pycuda_{size}.jpg"
        cv2.imwrite(nombreArchivo, imagenSalida)