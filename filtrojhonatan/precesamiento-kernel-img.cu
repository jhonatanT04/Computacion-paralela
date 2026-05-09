#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <cufft.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ═══════════════════════════════════════════════════════════════════════════
 *  memoria global
 * ═══════════════════════════════════════════════════════════════════════════*/
__global__ void emboss_global(
    const unsigned char *__restrict__ src,
    unsigned char *__restrict__ dst,
    const float *__restrict__ kernel,
    int W, int H, int ksize)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= W * H)
        return;
    int x = i % W;
    int y = i / W;

    int half = ksize / 2;
    float sum = 0.f;

    for (int ky = -half; ky <= half; ky++)
    {
        for (int kx = -half; kx <= half; kx++)
        {
            int sx = min(max(x + kx, 0), W - 1);
            int sy = min(max(y + ky, 0), H - 1);
            int idx = (sy * W + sx) * 4;
            float lum = 0.299f * src[idx] + 0.587f * src[idx + 1] + 0.114f * src[idx + 2];
            sum += lum * kernel[(ky + half) * ksize + (kx + half)];
        }
    }

    float norm = (float)ksize * 0.8f;
    int v = min(max((int)(sum / norm + 128.5f), 0), 255);
    int o = (y * W + x) * 4;
    dst[o] = dst[o + 1] = dst[o + 2] = (unsigned char)v;
    dst[o + 3] = 255;
}

__global__ void place_kernel_wrapped(
    float *__restrict__ dst,
    int padW, int padH,
    int ksize)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= padW * padH)
        return;
    int x = i % padW;
    int y = i / padW;

    int half = ksize / 2;

    int kx = x;
    int ky = y;

    float val = 0.f;

    if (kx <= half && ky <= half)
    {
        val = (float)((kx) - (ky));
    }

    else if (kx >= padW - half && ky <= half)
    {
        int real_kx = kx - padW + ksize;
        if (real_kx < ksize)
            val = (float)((real_kx - half) - (ky));
    }
    else if (kx <= half && ky >= padH - half)
    {
        int real_ky = ky - padH + ksize;
        if (real_ky < ksize)
            val = (float)((kx) - (real_ky - half));
    }
    else if (kx >= padW - half && ky >= padH - half)
    {
        int real_kx = kx - padW + ksize;
        int real_ky = ky - padH + ksize;
        if (real_kx < ksize && real_ky < ksize)
            val = (float)((real_kx - half) - (real_ky - half));
    }
    dst[y * padW + x] = val;
}

__global__ void complex_multiply(
    cufftComplex *__restrict__ a,
    const cufftComplex *__restrict__ b,
    int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    float re = a[i].x * b[i].x - a[i].y * b[i].y;
    float im = a[i].x * b[i].y + a[i].y * b[i].x;
    a[i].x = re;
    a[i].y = im;
}

__global__ void fft_result_to_rgba(
    const float *__restrict__ src,
    unsigned char *__restrict__ dst,
    int W, int H, int padW,
    float norm_fft,
    float norm_kernel,
    int ksize)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= W * H)
        return;
    int x = i % W;
    int y = i / W;

    float val = src[y * padW + x] * norm_fft / norm_kernel + 128.f;
    int v = min(max((int)(val + 0.5f), 0), 255);
    int o = (y * W + x) * 4;
    dst[o] = dst[o + 1] = dst[o + 2] = (unsigned char)v;
    dst[o + 3] = 255;
}


static void build_emboss_kernel(float *k, int n)
{
    int half = n / 2;
    for (int r = 0; r < n; r++)
        for (int c = 0; c < n; c++)
            k[r * n + c] = (float)((c - half) - (r - half));
}

int main(int argc, char **argv)
{
    if (argc < 3)
    {
        fprintf(stderr, "Uso: %s <input.png> <output.png> [kernel_size]\n", argv[0]);
        return EXIT_FAILURE;
    }

    const char *input_path = argv[1];
    const char *output_path = argv[2];
    // int ksize = 65;
    int ksize = (argc >= 4) ? atoi(argv[3]) : 65;

    cv::Mat img = cv::imread(input_path, cv::IMREAD_COLOR);
    if (img.empty())
    {
        fprintf(stderr, "Error al cargar '%s'\n", input_path);
        return EXIT_FAILURE;
    }
    cv::Mat img_rgba;
    cv::cvtColor(img, img_rgba, cv::COLOR_BGR2RGBA);
    int W = img_rgba.cols;
    int H = img_rgba.rows;
    unsigned char *h_src = img_rgba.data;

    size_t img_bytes = (size_t)W * H * 4;
    unsigned char *d_src, *d_dst;
    cudaMalloc(&d_src, img_bytes);
    cudaMalloc(&d_dst, img_bytes);
    cudaMemcpy(d_src, h_src, img_bytes, cudaMemcpyHostToDevice);

    int block = 256;
    int grid = (W * H + block - 1) / block;

    int half = ksize / 2;

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);

    size_t k_bytes = (size_t)ksize * ksize * sizeof(float);
    float *h_k = (float *)malloc(k_bytes);
    build_emboss_kernel(h_k, ksize);
    // print_kernel_summary(h_k, ksize);

    float *d_k;
    cudaMalloc(&d_k, k_bytes);
    cudaMemcpy(d_k, h_k, k_bytes, cudaMemcpyHostToDevice);
    free(h_k);

    cudaEventRecord(t0);
    emboss_global<<<grid, block>>>(d_src, d_dst, d_k, W, H, ksize);
    cudaEventRecord(t1);

    cudaFree(d_k);

    cudaGetLastError();
    cudaEventSynchronize(t1);
    float ms = 0;
    cudaEventElapsedTime(&ms, t0, t1);
    printf("Tiempo GPU: %.3f ms\n", ms);

    unsigned char *h_dst = (unsigned char *)malloc(img_bytes);
    cudaMemcpy(h_dst, d_dst, img_bytes, cudaMemcpyDeviceToHost);

    cv::Mat out(H, W, CV_8UC4, h_dst);
    cv::Mat out_bgr;
    cv::cvtColor(out, out_bgr, cv::COLOR_RGBA2BGR);
    if (!cv::imwrite(output_path, out_bgr))
        fprintf(stderr, "Error al guardar '%s'\n", output_path);
    else
        printf("Guardado: %s\n", output_path);

    free(h_dst);
    cudaFree(d_src);
    cudaFree(d_dst);
    cudaEventDestroy(t0);
    cudaEventDestroy(t1);

    return EXIT_SUCCESS;
}
