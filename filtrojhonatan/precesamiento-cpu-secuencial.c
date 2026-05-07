#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>


static void build_emboss_kernel(float* k, int n)
{
    int half = n / 2;
    for (int r = 0; r < n; r++)
        for (int c = 0; c < n; c++)
            k[r*n+c] = (float)((c - half) - (r - half));
}


static void emboss_cpu(
    const unsigned char* src,
          unsigned char* dst,
    const float*         kernel,
    int W, int H, int ksize)
{
    int half = ksize / 2;
    float norm = (float)ksize * 0.8f;

    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
            float sum = 0.f;

            for (int ky = -half; ky <= half; ky++) {
                for (int kx = -half; kx <= half; kx++) {
                    int sx = x + kx;
                    int sy = y + ky;

                    /* clamp en bordes (mirror del comportamiento CUDA) */
                    if (sx < 0)   sx = 0;
                    if (sx >= W)  sx = W - 1;
                    if (sy < 0)   sy = 0;
                    if (sy >= H)  sy = H - 1;

                    int idx = (sy * W + sx) * 4;
                    float lum = 0.299f * src[idx]
                              + 0.587f * src[idx + 1]
                              + 0.114f * src[idx + 2];

                    sum += lum * kernel[(ky + half) * ksize + (kx + half)];
                }
            }

            int v = (int)(sum / norm + 128.5f);
            if (v < 0)   v = 0;
            if (v > 255) v = 255;

            int o = (y * W + x) * 4;
            dst[o]     = (unsigned char)v;
            dst[o + 1] = (unsigned char)v;
            dst[o + 2] = (unsigned char)v;
            dst[o + 3] = 255;
        }
    }
}

int main(int argc, char** argv)
{
    if (argc < 3) {
        fprintf(stderr, "Uso: %s <input.png> <output.png>\n", argv[0]);
        return EXIT_FAILURE;
    }

    const char* input_path  = argv[1];
    const char* output_path = argv[2];
    int ksize = 63;
    // int ksize = (argc >= 4) ? atoi(argv[3]) : 3;

    int W, H, ch;
    unsigned char* h_src = stbi_load(input_path, &W, &H, &ch, 4);
    if (!h_src) {
        fprintf(stderr, "Error al cargar '%s'\n", input_path);
        return EXIT_FAILURE;
    }

    size_t img_bytes = (size_t)W * H * 4;
    unsigned char* h_dst = (unsigned char*)malloc(img_bytes);
    if (!h_dst) {
        fprintf(stderr, "Error: no hay memoria suficiente\n");
        stbi_image_free(h_src);
        return EXIT_FAILURE;
    }

    int half = ksize / 2;

    if (half <= 31) {
        size_t k_bytes = (size_t)ksize * ksize * sizeof(float);
        float* h_k = (float*)malloc(k_bytes);
        build_emboss_kernel(h_k, ksize);
        emboss_cpu(h_src, h_dst, h_k, W, H, ksize);
        free(h_k);
    } else {
        fprintf(stderr, "kernel_size demasiado grande (half > 31)\n");
        free(h_dst);
        stbi_image_free(h_src);
        return EXIT_FAILURE;
    }

    if (!stbi_write_png(output_path, W, H, 4, h_dst, W * 4))
        fprintf(stderr, "Error al guardar '%s'\n", output_path);
    else
        printf("Guardado: %s\n", output_path);

    stbi_image_free(h_src);
    free(h_dst);

    return EXIT_SUCCESS;
}
