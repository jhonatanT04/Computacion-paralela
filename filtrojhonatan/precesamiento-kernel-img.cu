/*
 * emboss_large.cu — Filtro Emboss con kernels grandes en CUDA
 *
 * Estrategia automática según tamaño de kernel:
 *   ksize <= 11  → memoria constante   (caché broadcast, óptimo)
 *   ksize <= 63  → memoria global      (simple, sin overhead de FFT)
 *   ksize >= 64  → cuFFT               (O(NlogN), único camino viable)
 *
 * Compilar:
 *   nvcc emboss_large.cu -o emboss_large -lcufft -O2
 *
 * Uso:
 *   ./emboss_large <input.png> <output.png> [kernel_size]
 *   kernel_size: 3, 5, 7, ..., 64, 128, 255, ...  (impar recomendado)
 *
 * Dependencias header-only (mismo directorio):
 *   stb_image.h        https://github.com/nothings/stb/blob/master/stb_image.h
 *   stb_image_write.h  https://github.com/nothings/stb/blob/master/stb_image_write.h
 */

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"

#include <cuda_runtime.h>
#include <cufft.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ── Límites de estrategia ────────────────────────────────────────────────── */
#define CONST_MEM_MAX_HALF   5          // hasta 11×11 en __constant__
#define GLOBAL_MEM_MAX_HALF  31         // hasta 63×63 en global
// >= 64×64 → cuFFT

#define MAX_CONST_SIDE (2*CONST_MEM_MAX_HALF+1)

/* ── Memoria constante (solo para kernels pequeños) ───────────────────────── */
__constant__ float d_kernel_const[MAX_CONST_SIDE * MAX_CONST_SIDE];
__constant__ int   d_ksize_c;
__constant__ int   d_khalf_c;

/* ── Macro de error CUDA ─────────────────────────────────────────────────── */
#define CUDA_CHECK(call) do {                                                  \
    cudaError_t _e = (call);                                                   \
    if (_e != cudaSuccess) {                                                   \
        fprintf(stderr, "CUDA error %s:%d — %s\n",                            \
                __FILE__, __LINE__, cudaGetErrorString(_e));                   \
        exit(EXIT_FAILURE);                                                    \
    }                                                                          \
} while(0)

#define CUFFT_CHECK(call) do {                                                 \
    cufftResult _e = (call);                                                   \
    if (_e != CUFFT_SUCCESS) {                                                 \
        fprintf(stderr, "cuFFT error %s:%d — code %d\n",                      \
                __FILE__, __LINE__, (int)_e);                                  \
        exit(EXIT_FAILURE);                                                    \
    }                                                                          \
} while(0)

/* ═══════════════════════════════════════════════════════════════════════════
 * ESTRATEGIA A: memoria constante (ksize <= 11)
 * ═══════════════════════════════════════════════════════════════════════════*/
__global__ void emboss_const(
    const unsigned char* __restrict__ src,
          unsigned char* __restrict__ dst,
    int W, int H)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;

    int half = d_khalf_c;
    int ksize = d_ksize_c;
    float sum = 0.f;

    for (int ky = -half; ky <= half; ky++) {
        for (int kx = -half; kx <= half; kx++) {
            int sx = min(max(x + kx, 0), W - 1);
            int sy = min(max(y + ky, 0), H - 1);
            int idx = (sy * W + sx) * 4;
            float lum = 0.299f*src[idx] + 0.587f*src[idx+1] + 0.114f*src[idx+2];
            sum += lum * d_kernel_const[(ky+half)*ksize + (kx+half)];
        }
    }

    float norm = (float)ksize * 0.8f;
    int v = min(max((int)(sum/norm + 128.5f), 0), 255);
    int o = (y * W + x) * 4;
    dst[o] = dst[o+1] = dst[o+2] = (unsigned char)v;
    dst[o+3] = 255;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * ESTRATEGIA B: memoria global (11 < ksize <= 63)
 * ═══════════════════════════════════════════════════════════════════════════*/
__global__ void emboss_global(
    const unsigned char* __restrict__ src,
          unsigned char* __restrict__ dst,
    const float*         __restrict__ kernel,
    int W, int H, int ksize)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;

    int half = ksize / 2;
    float sum = 0.f;

    for (int ky = -half; ky <= half; ky++) {
        for (int kx = -half; kx <= half; kx++) {
            int sx = min(max(x + kx, 0), W - 1);
            int sy = min(max(y + ky, 0), H - 1);
            int idx = (sy * W + sx) * 4;
            float lum = 0.299f*src[idx] + 0.587f*src[idx+1] + 0.114f*src[idx+2];
            sum += lum * kernel[(ky+half)*ksize + (kx+half)];
        }
    }

    float norm = (float)ksize * 0.8f;
    int v = min(max((int)(sum/norm + 128.5f), 0), 255);
    int o = (y * W + x) * 4;
    dst[o] = dst[o+1] = dst[o+2] = (unsigned char)v;
    dst[o+3] = 255;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * ESTRATEGIA C: cuFFT — kernels grandes (ksize >= 64)
 *
 * Algoritmo:
 *   1. Convertir imagen a float (luminancia), pad a tamaño FFT
 *   2. Colocar kernel centrado en imagen pad, con wrap-around
 *   3. FFT(imagen) × FFT(kernel) → IFFT → resultado en dominio espacial
 *   4. Sumar bias +128, clamp, escribir salida
 * ═══════════════════════════════════════════════════════════════════════════*/

/* Convierte RGBA uchar → float luminancia */
__global__ void rgba_to_lum(
    const unsigned char* __restrict__ src,
    float* __restrict__ dst,
    int W, int H, int padW, int padH)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= padW || y >= padH) { 
        if (x < padW && y < padH) dst[y*padW+x] = 0.f;
        return;
    }
    if (x < W && y < H) {
        int idx = (y*W+x)*4;
        dst[y*padW+x] = 0.299f*src[idx] + 0.587f*src[idx+1] + 0.114f*src[idx+2];
    } else {
        dst[y*padW+x] = 0.f;
    }
}

/* Coloca el kernel emboss en imagen padW×padH con wrap-around (circular shift) */
__global__ void place_kernel_wrapped(
    float* __restrict__ dst,
    int padW, int padH,
    int ksize)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= padW || y >= padH) return;

    int half = ksize / 2;
    /* Posición dentro del kernel (con wrap-around centrado) */
    int kx = x;
    int ky = y;
    /* Solo las posiciones que corresponden al kernel */
    float val = 0.f;
    /* Cuadrante superior izquierdo (valores positivos del kernel) */
    if (kx <= half && ky <= half) {
        val = (float)((kx) - (ky));               /* (c-half)-(r-half) con half=kx,ky */
    }
    /* Cuadrante inferior derecho (wrap-around de la parte negativa) */
    else if (kx >= padW - half && ky <= half) {
        int real_kx = kx - padW + ksize;
        if (real_kx < ksize)
            val = (float)((real_kx - half) - (ky));
    }
    else if (kx <= half && ky >= padH - half) {
        int real_ky = ky - padH + ksize;
        if (real_ky < ksize)
            val = (float)((kx) - (real_ky - half));
    }
    else if (kx >= padW - half && ky >= padH - half) {
        int real_kx = kx - padW + ksize;
        int real_ky = ky - padH + ksize;
        if (real_kx < ksize && real_ky < ksize)
            val = (float)((real_kx - half) - (real_ky - half));
    }
    dst[y*padW+x] = val;
}

/* Multiplicación elemento a elemento en dominio frecuencial */
__global__ void complex_multiply(
    cufftComplex* __restrict__ a,
    const cufftComplex* __restrict__ b,
    int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float re = a[i].x * b[i].x - a[i].y * b[i].y;
    float im = a[i].x * b[i].y + a[i].y * b[i].x;
    a[i].x = re;
    a[i].y = im;
}

/* Convierte float IFFT resultado → uchar RGBA (escala de grises) */
__global__ void fft_result_to_rgba(
    const float* __restrict__ src,
          unsigned char* __restrict__ dst,
    int W, int H, int padW,
    float norm_fft,    /* 1/(padW*padH)  — normalización de IFFT */
    float norm_kernel, /* ksize*0.8      — normalización del emboss */
    int ksize)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;

    float val = src[y*padW+x] * norm_fft / norm_kernel + 128.f;
    int v = min(max((int)(val + 0.5f), 0), 255);
    int o = (y*W+x)*4;
    dst[o] = dst[o+1] = dst[o+2] = (unsigned char)v;
    dst[o+3] = 255;
}

/* ─────────────────────────────────────────────────────────────────────────────
 * Siguiente potencia de 2 >= n  (para tamaño óptimo de FFT)
 * ───────────────────────────────────────────────────────────────────────────*/
static int next_pow2(int n)
{
    int p = 1;
    while (p < n) p <<= 1;
    return p;
}

/* ─────────────────────────────────────────────────────────────────────────────
 * Construye kernel emboss en CPU
 * ───────────────────────────────────────────────────────────────────────────*/
static void build_emboss_kernel(float* k, int n)
{
    int half = n / 2;
    for (int r = 0; r < n; r++)
        for (int c = 0; c < n; c++)
            k[r*n+c] = (float)((c-half) - (r-half));
}

static void print_kernel_summary(const float* k, int n)
{
    printf("Kernel emboss %dx%d (%d valores):\n", n, n, n*n);
    if (n <= 9) {
        for (int r = 0; r < n; r++) {
            for (int c = 0; c < n; c++)
                printf("%5.0f", k[r*n+c]);
            printf("\n");
        }
    } else {
        printf("  [esquina TL] %5.0f %5.0f ...  %5.0f %5.0f\n",
               k[0], k[1], k[n-2], k[n-1]);
        printf("               ...              ...\n");
        printf("  [esquina BR] %5.0f %5.0f ...  %5.0f %5.0f\n",
               k[(n-1)*n+0], k[(n-1)*n+1], k[(n-1)*n+n-2], k[(n-1)*n+n-1]);
        float vmin=k[0], vmax=k[0];
        for (int i=1;i<n*n;i++){if(k[i]<vmin)vmin=k[i];if(k[i]>vmax)vmax=k[i];}
        printf("  rango: [%.0f, %.0f]\n", vmin, vmax);
    }
}

/* ═══════════════════════════════════════════════════════════════════════════
 * MAIN
 * ═══════════════════════════════════════════════════════════════════════════*/
int main(int argc, char** argv)
{
    if (argc < 3) {
        fprintf(stderr, "Uso: %s <input.png> <output.png> [kernel_size]\n", argv[0]);
        return EXIT_FAILURE;
    }

    const char* input_path  = argv[1];
    const char* output_path = argv[2];
    int ksize = (argc >= 4) ? atoi(argv[3]) : 3;

    if (ksize < 3) {
        fprintf(stderr, "kernel_size mínimo: 3\n");
        return EXIT_FAILURE;
    }
    /* Forzar impar */
    if (ksize % 2 == 0) { ksize++; printf("Ajustando kernel a impar: %d\n", ksize); }

    /* ── 1. Cargar imagen ─────────────────────────────────────────────────── */
    int W, H, ch;
    unsigned char* h_src = stbi_load(input_path, &W, &H, &ch, 4);
    if (!h_src) {
        fprintf(stderr, "Error al cargar '%s': %s\n", input_path, stbi_failure_reason());
        return EXIT_FAILURE;
    }
    printf("Imagen: %dx%d px  |  Kernel: %dx%d\n", W, H, ksize, ksize);

    size_t img_bytes = (size_t)W * H * 4;
    unsigned char *d_src, *d_dst;
    CUDA_CHECK(cudaMalloc(&d_src, img_bytes));
    CUDA_CHECK(cudaMalloc(&d_dst, img_bytes));
    CUDA_CHECK(cudaMemcpy(d_src, h_src, img_bytes, cudaMemcpyHostToDevice));

    dim3 block(16, 16);
    dim3 grid((W+15)/16, (H+15)/16);

    /* ── Elegir estrategia ────────────────────────────────────────────────── */
    int half = ksize / 2;

    cudaEvent_t t0, t1;
    CUDA_CHECK(cudaEventCreate(&t0));
    CUDA_CHECK(cudaEventCreate(&t1));

    /* ══════════════════════════════════════════════════════════════════════
     * ESTRATEGIA A — memoria constante
     * ═════════════════════════════════════════════════════════════════════*/
    if (half <= CONST_MEM_MAX_HALF) {
        printf("Estrategia: memoria constante (__constant__)\n");

        float h_k[MAX_CONST_SIDE * MAX_CONST_SIDE];
        build_emboss_kernel(h_k, ksize);
        print_kernel_summary(h_k, ksize);

        CUDA_CHECK(cudaMemcpyToSymbol(d_kernel_const, h_k, ksize*ksize*sizeof(float)));
        CUDA_CHECK(cudaMemcpyToSymbol(d_ksize_c, &ksize, sizeof(int)));
        CUDA_CHECK(cudaMemcpyToSymbol(d_khalf_c, &half,  sizeof(int)));

        CUDA_CHECK(cudaEventRecord(t0));
        emboss_const<<<grid, block>>>(d_src, d_dst, W, H);
        CUDA_CHECK(cudaEventRecord(t1));
    }
    /* ══════════════════════════════════════════════════════════════════════
     * ESTRATEGIA B — memoria global
     * ═════════════════════════════════════════════════════════════════════*/
    else if (half <= GLOBAL_MEM_MAX_HALF) {
        printf("Estrategia: memoria global (cudaMalloc)\n");

        size_t k_bytes = (size_t)ksize * ksize * sizeof(float);
        float* h_k = (float*)malloc(k_bytes);
        build_emboss_kernel(h_k, ksize);
        print_kernel_summary(h_k, ksize);

        float* d_k;
        CUDA_CHECK(cudaMalloc(&d_k, k_bytes));
        CUDA_CHECK(cudaMemcpy(d_k, h_k, k_bytes, cudaMemcpyHostToDevice));
        free(h_k);

        CUDA_CHECK(cudaEventRecord(t0));
        emboss_global<<<grid, block>>>(d_src, d_dst, d_k, W, H, ksize);
        CUDA_CHECK(cudaEventRecord(t1));

        cudaFree(d_k);
    }
    /* ══════════════════════════════════════════════════════════════════════
     * ESTRATEGIA C — cuFFT
     * ═════════════════════════════════════════════════════════════════════*/
    else {
        printf("Estrategia: cuFFT (convolución en dominio frecuencial)\n");

        /* Tamaño del dominio FFT: potencia de 2 >= imagen + kernel - 1 */
        int padW = next_pow2(W + ksize - 1);
        int padH = next_pow2(H + ksize - 1);
        printf("Padding FFT: %dx%d\n", padW, padH);

        size_t pad_floats = (size_t)padW * padH;
        size_t pad_bytes  = pad_floats * sizeof(float);
        size_t cpx_bytes  = pad_floats * sizeof(cufftComplex);

        /* Buffers float en GPU */
        float *d_img_f, *d_ker_f;
        CUDA_CHECK(cudaMalloc(&d_img_f, pad_bytes));
        CUDA_CHECK(cudaMalloc(&d_ker_f, pad_bytes));
        CUDA_CHECK(cudaMemset(d_img_f, 0, pad_bytes));
        CUDA_CHECK(cudaMemset(d_ker_f, 0, pad_bytes));

        /* Buffers complejos en GPU */
        cufftComplex *d_img_c, *d_ker_c;
        CUDA_CHECK(cudaMalloc(&d_img_c, cpx_bytes));
        CUDA_CHECK(cudaMalloc(&d_ker_c, cpx_bytes));

        dim3 gpad((padW+15)/16, (padH+15)/16);

        CUDA_CHECK(cudaEventRecord(t0));

        /* 1. RGBA → luminancia float (con padding) */
        rgba_to_lum<<<gpad, block>>>(d_src, d_img_f, W, H, padW, padH);

        /* 2. Kernel emboss en imagen paddeada (wrap-around) */
        place_kernel_wrapped<<<gpad, block>>>(d_ker_f, padW, padH, ksize);

        /* 3. Crear planes FFT R2C */
        cufftHandle plan_fwd_img, plan_fwd_ker, plan_inv;
        CUFFT_CHECK(cufftPlan2d(&plan_fwd_img, padH, padW, CUFFT_R2C));
        CUFFT_CHECK(cufftPlan2d(&plan_fwd_ker, padH, padW, CUFFT_R2C));
        CUFFT_CHECK(cufftPlan2d(&plan_inv,     padH, padW, CUFFT_C2R));

        /* 4. FFT directa de imagen y kernel */
        CUFFT_CHECK(cufftExecR2C(plan_fwd_img, d_img_f, d_img_c));
        CUFFT_CHECK(cufftExecR2C(plan_fwd_ker, d_ker_f, d_ker_c));

        /* 5. Multiplicación en frecuencia */
        int cpx_count = padH * (padW/2 + 1);
        complex_multiply<<<(cpx_count+255)/256, 256>>>(d_img_c, d_ker_c, cpx_count);

        /* 6. IFFT */
        CUFFT_CHECK(cufftExecC2R(plan_inv, d_img_c, d_img_f));

        /* 7. Resultado → RGBA uchar */
        float norm_fft    = 1.0f / (float)(padW * padH);
        float norm_kernel = (float)ksize * 0.8f;
        fft_result_to_rgba<<<grid, block>>>(
            d_img_f, d_dst, W, H, padW, norm_fft, norm_kernel, ksize);

        CUDA_CHECK(cudaEventRecord(t1));

        /* Liberar recursos FFT */
        cufftDestroy(plan_fwd_img);
        cufftDestroy(plan_fwd_ker);
        cufftDestroy(plan_inv);
        cudaFree(d_img_f); cudaFree(d_ker_f);
        cudaFree(d_img_c); cudaFree(d_ker_c);
    }

    /* ── Medir tiempo ─────────────────────────────────────────────────────── */
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventSynchronize(t1));
    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));
    printf("Tiempo GPU: %.3f ms\n", ms);

    /* ── Copiar resultado y guardar ───────────────────────────────────────── */
    unsigned char* h_dst = (unsigned char*)malloc(img_bytes);
    CUDA_CHECK(cudaMemcpy(h_dst, d_dst, img_bytes, cudaMemcpyDeviceToHost));

    if (!stbi_write_png(output_path, W, H, 4, h_dst, W*4))
        fprintf(stderr, "Error al guardar '%s'\n", output_path);
    else
        printf("Guardado: %s\n", output_path);

    /* ── Liberar ──────────────────────────────────────────────────────────── */
    stbi_image_free(h_src);
    free(h_dst);
    cudaFree(d_src);
    cudaFree(d_dst);
    cudaEventDestroy(t0);
    cudaEventDestroy(t1);

    return EXIT_SUCCESS;
}
