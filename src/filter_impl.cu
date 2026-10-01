#include "filter_impl.h"
#include <cassert>
#include <chrono>
#include <thread>
#include <cstdio>
#include <cstdlib>
#include "logo.h"

#define RGB_DIFF_THRESHOLD 30
#define MAX_WEIGHTS 1000
#define K 5

#define LOOP_DEVICE_COPY 4

#define HYSTERESIS_LOW 15
#define HYSTERESIS_HIGH 50

#define MAX_PASSES 100
#define SPACES 8

#define OPENING_SIZE 2

#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)
template <typename T>
void check(T err, const char* const func, const char* const file,
           const int line)
{
    if (err != cudaSuccess)
    {
        std::fprintf(stderr, "CUDA Runtime Error at: %s: %d\n", file, line);
        std::fprintf(stderr, "%s %s\n", cudaGetErrorString(err), func);
        // We don't exit when we encounter CUDA errors in this example.
        std::exit(EXIT_FAILURE);
    }
}

struct rgb {
    uint8_t r, g, b;
};


__constant__ uint8_t* logo;

/// @brief Black out the red channel from the video and add EPITA's logo
/// @param buffer
/// @param width
/// @param height
/// @param stride
/// @param pixel_stride
/// @return 
__global__ void remove_red_channel_inp(std::byte* buffer, int width, int height, int stride)
{
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int x = blockIdx.x * blockDim.x + threadIdx.x;

    if (x >= width || y >= height)
        return; 

    rgb* lineptr = (rgb*) (buffer + y * stride);
    if (y < logo_height && x < logo_width) {
        float alpha = logo[y * logo_width + x] / 255.f;
        lineptr[x].r = 0;
        lineptr[x].g = uint8_t(alpha * lineptr[x].g + (1-alpha) * 255);
        lineptr[x].b = uint8_t(alpha * lineptr[x].b + (1-alpha) * 255);
    } else {
        lineptr[x].r = 0;
    }
}

__device__ bool has_changed = false;

// Background
__device__ static uint8_t *d_bg_r;
__device__ static uint8_t *d_bg_g;
__device__ static uint8_t *d_bg_b;

// Reservoirs
__device__ static uint8_t *d_rs_r;
__device__ static uint8_t *d_rs_g;
__device__ static uint8_t *d_rs_b;
__device__ static int *d_rs_w;

static uint8_t *d_motion_mask;
static uint8_t *d_temp_mask;
static bool *input = nullptr;
static bool *marker = nullptr;
static bool *out = nullptr;
static bool initialized = false;
static int nb_passes = 0;
static int nb_loop = 0;

// Same deterministic random algo used in ISIM
__device__ uint32_t xorshift32(uint32_t* state) {
    uint32_t x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    return x;
}

__device__ inline int safe_dist(uint8_t a, uint8_t b) {
    return b > a ? (b - a) : (a - b);
}

__global__ void masking(uint8_t* buffer, uint8_t *mask, int width, int height, int stride, int pixel_stride) {
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int x = blockIdx.x * blockDim.x + threadIdx.x;

    if (x >= width || y >= height) return;

    if (mask[y * width + x] > 0) {
        rgb* pixel_ptr = (rgb*)(buffer + y * stride + x * pixel_stride);
        auto toAdd = mask[y * width + x] / 2;
        if (255 - pixel_ptr->r < toAdd) {
            pixel_ptr->r = 255;
        } else {
            pixel_ptr->r += toAdd;
        }
    }
}

__global__ void threshold_kernel(uint8_t* diff, bool* input, bool* marker, int n) {
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p < n) {
        input[p] = diff[p] >= HYSTERESIS_LOW;
        marker[p] = diff[p] >= HYSTERESIS_HIGH;
    }
}

__global__ void to_uint8_mask(bool* out, uint8_t* mask, int n) {
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p < n) mask[p] = out[p] ? 255 : 0;
}

#define TILE 32

__global__ void reconstruction_tiled(const bool* __restrict__ input,
                                      const bool* __restrict__ marker,
                                      bool* __restrict__ out,
                                      int width, int height)
{
    __shared__ bool sh_out[TILE + 2][TILE + 2];
    __shared__ bool sh_input[TILE][TILE];
    __shared__ bool sh_marker[TILE][TILE];
    __shared__ int sh_iter_changed;
    __shared__ int sh_block_changed;

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int x = blockIdx.x * TILE + tx;
    int y = blockIdx.y * TILE + ty;
    bool valid = (x < width && y < height);
    int idx = valid ? y * width + x : 0;

    sh_input[ty][tx] = valid ? input[idx] : false;
    sh_marker[ty][tx] = valid ? marker[idx] : false;
    sh_out[ty + 1][tx + 1] = valid ? out[idx] : false;

    bool rightEdge = (tx == TILE - 1) || (x == width - 1);
    bool bottomEdge = (ty == TILE - 1) || (y == height - 1);

    if (tx == 0) {
        int gx = x - 1;
        sh_out[ty + 1][0] = (gx >= 0 && y < height) ? out[y * width + gx] : false;
    }
    if (rightEdge) {
        int gx = x + 1;
        sh_out[ty + 1][tx + 2] = (gx < width && y < height) ? out[y * width + gx] : false;
    }
    if (ty == 0) {
        int gy = y - 1;
        sh_out[0][tx + 1] = (gy >= 0 && x < width) ? out[gy * width + x] : false;
    }
    if (bottomEdge) {
        int gy = y + 1;
        sh_out[ty + 2][tx + 1] = (gy < height && x < width) ? out[gy * width + x] : false;
    }
    if (tx == 0 && ty == 0) {
        int gx = x - 1, gy = y - 1;
        sh_out[0][0] = (gx >= 0 && gy >= 0) ? out[gy * width + gx] : false;
    }
    if (rightEdge && ty == 0) {
        int gx = x + 1;
        int gy = y - 1;
        sh_out[0][tx + 2] = (gx < width && gy >= 0) ? out[gy * width + gx] : false;
    }
    if (tx == 0 && bottomEdge) {
        int gx = x - 1;
        int gy = y + 1;
        sh_out[ty + 2][0] = (gx >= 0 && gy < height) ? out[gy * width + gx] : false;
    }
    if (rightEdge && bottomEdge) {
        int gx = x + 1;
        int gy = y + 1;
        sh_out[ty + 2][tx + 2] = (gx < width && gy < height) ? out[gy * width + gx] : false;
    }

    if (tx == 0 && ty == 0) sh_block_changed = 0;
    __syncthreads();

    const int MAX_LOCAL_ITERS = 2 * TILE;
    for (int iter = 0; iter < MAX_LOCAL_ITERS; ++iter) {
        if (tx == 0 && ty == 0) sh_iter_changed = 0;
        __syncthreads();

        bool mine = false;
        if (valid && !sh_out[ty + 1][tx + 1] && sh_input[ty][tx]) {
            if (sh_marker[ty][tx]) {
                mine = true;
            } else {
                for (int dy = -1; dy <= 1 && !mine; ++dy)
                    for (int dx = -1; dx <= 1 && !mine; ++dx) {
                        if (dx == 0 && dy == 0) continue;
                        if (sh_out[ty + 1 + dy][tx + 1 + dx]) mine = true;
                    }
            }
        }
        __syncthreads();

        if (mine) {
            sh_out[ty + 1][tx + 1] = true;
            atomicOr(&sh_iter_changed, 1);
            atomicOr(&sh_block_changed, 1);
        }
        __syncthreads();

        if (sh_iter_changed == 0) break;
    }

    if (valid) out[idx] = sh_out[ty + 1][tx + 1];

    if (tx == 0 && ty == 0 && sh_block_changed) {
        has_changed = true;
    }
}

__global__ void reconstruction(bool* input, bool* marker, bool* out, int width, int height) {
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= width * height || out[p] || !input[p]) return;
    if (marker[p]) { out[p] = true; has_changed = true; return; }

    int x = p % width, y = p / width;
    for (int dy = -1; dy <= 1; ++dy)
        for (int dx = -1; dx <= 1; ++dx) {
            int nx = x + dx, ny = y + dy;
            if (nx < 0 || nx >= width || ny < 0 || ny >= height) continue;
            if (out[ny * width + nx]) { out[p] = true; has_changed = true; }
        }
}

__device__ inline int clampi(int v, int lo, int hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

__global__ void erode_h(const uint8_t* __restrict__ src, uint8_t* __restrict__ dst,
                         int width, int height, int r) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int m = 255;
    for (int dx = -r; dx <= r; ++dx) {
        int nx = clampi(x + dx, 0, width - 1);
        m = min(m, (int)src[y * width + nx]);
    }

    dst[y * width + x] = (uint8_t)m;
}

__global__ void erode_v(const uint8_t* __restrict__ src, uint8_t* __restrict__ dst,
                         int width, int height, int r) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int m = 255;
    for (int dy = -r; dy <= r; ++dy) {
        int ny = clampi(y + dy, 0, height - 1);
        m = min(m, (int)src[ny * width + x]);
    }

    dst[y * width + x] = (uint8_t)m;
}

__global__ void dilate_h(const uint8_t* __restrict__ src, uint8_t* __restrict__ dst,
                          int width, int height, int r) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int m = 0;
    for (int dx = -r; dx <= r; ++dx) {
        int nx = clampi(x + dx, 0, width - 1);
        m = max(m, (int)src[y * width + nx]);
    }

    dst[y * width + x] = (uint8_t)m;
}

__global__ void dilate_v(const uint8_t* __restrict__ src, uint8_t* __restrict__ dst,
                          int width, int height, int r) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int m = 0;
    for (int dy = -r; dy <= r; ++dy) {
        int ny = clampi(y + dy, 0, height - 1);
        m = max(m, (int)src[ny * width + x]);
    }

    dst[y * width + x] = (uint8_t)m;
}


__global__ void movement_filter(uint8_t* buffer, uint8_t *mask, int width, int height, int stride, int pixel_stride) {
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int x = blockIdx.x * blockDim.x + threadIdx.x;

    if (x >= width || y >= height) return;

    rgb* pixel_ptr = (rgb*)(buffer + y * stride + x * pixel_stride);
    int pixel_id = y * width + x;

    int dr = safe_dist(pixel_ptr->r, d_bg_r[pixel_id]);
    int dg = safe_dist(pixel_ptr->g, d_bg_g[pixel_id]);
    int db = safe_dist(pixel_ptr->b, d_bg_b[pixel_id]);

    int diff = (dr + dg + db) / 3;
    mask[pixel_id] = static_cast<uint8_t>(diff > 255 ? 255 : diff);
}

__device__ int find_matching_reservoir(const rgb& pixel, int rs_stride, int pixel_id) {
    int m_idx = -1;
    for (int i = 0; i < K; ++i) {
        int off = i * rs_stride + pixel_id;
        if (d_rs_w[off] > 0) {
            if (safe_dist(pixel.r, d_rs_r[off]) < RGB_DIFF_THRESHOLD &&
                safe_dist(pixel.g, d_rs_g[off]) < RGB_DIFF_THRESHOLD &&
                safe_dist(pixel.b, d_rs_b[off]) < RGB_DIFF_THRESHOLD) {
                return i;
            }
        } else if (m_idx == -1) {
            m_idx = i;
        }
    }
    return m_idx;
}

__global__ void load_background_img(uint8_t* buffer, int width, int height, int stride, int pixel_stride, int nb_passes) {
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int x = blockIdx.x * blockDim.x + threadIdx.x;

    if (x >= width || y >= height) return;

    rgb* pixel_ptr = (rgb*)(buffer + y * stride + x * pixel_stride);
    rgb p = *pixel_ptr;

    int pixel_id = y * width + x;
    int rs_stride = width * height;

    int m_idx = find_matching_reservoir(p, rs_stride, pixel_id);

    if (m_idx != -1) {
        int off = m_idx * rs_stride + pixel_id;
        int w = d_rs_w[off];
        if (w > 0) {
            w += 1;
            if (w > MAX_WEIGHTS) w = MAX_WEIGHTS;
            d_rs_r[off] = ((w - 1) * d_rs_r[off] + p.r) / w;
            d_rs_g[off] = ((w - 1) * d_rs_g[off] + p.g) / w;
            d_rs_b[off] = ((w - 1) * d_rs_b[off] + p.b) / w;
            d_rs_w[off] = w;
        } else {
            d_rs_r[off] = p.r;
            d_rs_g[off] = p.g;
            d_rs_b[off] = p.b;
            d_rs_w[off] = 1;
        }
    } else {
        int min_idx = 0;
        int total_weights = 0;
        for (int i = 0; i < K; ++i) {
            int off = i * rs_stride + pixel_id;
            total_weights += d_rs_w[off];
            if (d_rs_w[off] < d_rs_w[min_idx * rs_stride + pixel_id]) {
                min_idx = i;
            }
        }

        // Initialize unique state seed per pixel coordinate
        uint32_t rng_state = (pixel_id + 1) ^ (nb_passes * 0x9E3779B9u);
        float rand_val = (float)(xorshift32(&rng_state) % 10000) / 10000.0f;

        int min_off = min_idx * rs_stride + pixel_id;
        if (rand_val * total_weights >= d_rs_w[min_off]) {
            d_rs_r[min_off] = p.r;
            d_rs_g[min_off] = p.g;
            d_rs_b[min_off] = p.b;
            d_rs_w[min_off] = 1;
        }
    }

    // Cap weights to MAX_WEIGHTS and find the dominant reservoir
    int max_weight_index = 0;
    int max_weight = d_rs_w[pixel_id];
    for (int i = 1; i < K; ++i) {
        int off = i * rs_stride + pixel_id;
        if (d_rs_w[off] > max_weight) {
            max_weight = d_rs_w[off];
            max_weight_index = i;
        }
        if (d_rs_w[off] > MAX_WEIGHTS) {
            d_rs_w[off] = MAX_WEIGHTS;
        }
    }

    int max_off = max_weight_index * rs_stride + pixel_id;
    d_bg_r[pixel_id] = d_rs_r[max_off];
    d_bg_g[pixel_id] = d_rs_g[max_off];
    d_bg_b[pixel_id] = d_rs_b[max_off];
}

namespace
{
    void load_logo()
    {
        static auto buffer = std::unique_ptr<std::byte, decltype(&cudaFree)>{nullptr, &cudaFree};

        if (buffer == nullptr)
        {
            cudaError_t err;
            std::byte* ptr;
            err = cudaMalloc(&ptr, logo_width * logo_height);
            CHECK_CUDA_ERROR(err);

            err = cudaMemcpy(ptr, logo_data, logo_width * logo_height, cudaMemcpyHostToDevice);
            CHECK_CUDA_ERROR(err);

            err = cudaMemcpyToSymbol(logo, &ptr, sizeof(ptr));
            CHECK_CUDA_ERROR(err);

            buffer.reset(ptr);
        }
    }
}


extern "C" {
    void filter_impl(uint8_t* src_buffer, int width, int height, int src_stride, int pixel_stride)
    {
        cudaError_t err;
        if (!initialized) {
            err = cudaDeviceSetLimit(cudaLimitMallocHeapSize, 256 * 1024 * 1024);
            CHECK_CUDA_ERROR(err);

            // Background
            uint8_t *tmp_bg_r;
            err = cudaMalloc(&tmp_bg_r, width * height * sizeof(uint8_t));
            CHECK_CUDA_ERROR(err);
            uint8_t *tmp_bg_g;
            err = cudaMalloc(&tmp_bg_g, width * height * sizeof(uint8_t));
            CHECK_CUDA_ERROR(err);
            uint8_t *tmp_bg_b;
            err = cudaMalloc(&tmp_bg_b, width * height * sizeof(uint8_t));
            CHECK_CUDA_ERROR(err);
            err = cudaMemcpyToSymbol(d_bg_r, &tmp_bg_r, sizeof(tmp_bg_r));
            CHECK_CUDA_ERROR(err);
            err = cudaMemcpyToSymbol(d_bg_g, &tmp_bg_g, sizeof(tmp_bg_g));
            CHECK_CUDA_ERROR(err);
            err = cudaMemcpyToSymbol(d_bg_b, &tmp_bg_b, sizeof(tmp_bg_b));
            CHECK_CUDA_ERROR(err);

            err = cudaMalloc(&d_motion_mask, width * height * sizeof(uint8_t));
            CHECK_CUDA_ERROR(err);
            err = cudaMalloc(&d_temp_mask, width * height * sizeof(uint8_t));
            CHECK_CUDA_ERROR(err);
            err = cudaMalloc(&input, width * height * sizeof(bool));
            CHECK_CUDA_ERROR(err);
            err = cudaMalloc(&marker, width * height * sizeof(bool));
            CHECK_CUDA_ERROR(err);
            err = cudaMalloc(&out, width * height * sizeof(bool));
            CHECK_CUDA_ERROR(err);

            // Reservoirs
            auto size_uint8 = width * height * K * sizeof(uint8_t);

            uint8_t *tmp_rs_r;
            err = cudaMalloc(&tmp_rs_r, size_uint8);
            CHECK_CUDA_ERROR(err);
            uint8_t *tmp_rs_g;
            err = cudaMalloc(&tmp_rs_g, size_uint8);
            CHECK_CUDA_ERROR(err);
            uint8_t *tmp_rs_b;
            err = cudaMalloc(&tmp_rs_b, size_uint8);
            CHECK_CUDA_ERROR(err);
            int *tmp_rs_w;
            err = cudaMalloc(&tmp_rs_w, width * height * K * sizeof(int));
            CHECK_CUDA_ERROR(err);

            err = cudaMemcpyToSymbol(d_rs_r, &tmp_rs_r, sizeof(tmp_rs_r));
            CHECK_CUDA_ERROR(err);
            err = cudaMemcpyToSymbol(d_rs_g, &tmp_rs_g, sizeof(tmp_rs_g));
            CHECK_CUDA_ERROR(err);
            err = cudaMemcpyToSymbol(d_rs_b, &tmp_rs_b, sizeof(tmp_rs_b));
            CHECK_CUDA_ERROR(err);
            err = cudaMemcpyToSymbol(d_rs_w, &tmp_rs_w, sizeof(tmp_rs_w));
            CHECK_CUDA_ERROR(err);

            err = cudaMemset(tmp_rs_w, 0, width * height * K * sizeof(int));
            CHECK_CUDA_ERROR(err);

            initialized = true;
        }

        assert(sizeof(rgb) == pixel_stride);
        std::uint8_t* dBuffer;
        size_t pitch;

        err = cudaMallocPitch(&dBuffer, &pitch, width * sizeof(rgb), height);
        CHECK_CUDA_ERROR(err);

        err = cudaMemcpy2D(dBuffer, pitch, src_buffer, src_stride, width * sizeof(rgb), height, cudaMemcpyHostToDevice);
        CHECK_CUDA_ERROR(err);

        dim3 blockSize(32, 32);
        dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

        if (nb_passes % SPACES == 0) {
            if (nb_passes < MAX_PASSES) {
                load_background_img<<<gridSize, blockSize>>>(dBuffer, width, height, pitch, pixel_stride, nb_passes);
            }
        }
        // Compute background
        // Get the movement filter
        movement_filter<<<gridSize, blockSize>>>(dBuffer, d_motion_mask, width, height, pitch, pixel_stride);
        // Noise Suppression
        erode_h<<<gridSize, blockSize>>>(d_motion_mask, d_temp_mask,  width, height, OPENING_SIZE);
        erode_v<<<gridSize, blockSize>>>(d_temp_mask,  d_motion_mask, width, height, OPENING_SIZE);
        dilate_h<<<gridSize, blockSize>>>(d_motion_mask, d_temp_mask,  width, height, OPENING_SIZE);
        dilate_v<<<gridSize, blockSize>>>(d_temp_mask,  d_motion_mask, width, height, OPENING_SIZE);

        dim3 reconBlock(TILE, TILE);
        dim3 reconGrid((width + TILE - 1) / TILE, (height + TILE - 1) / TILE);

        int threads = 256;
        int blocks = (width * height + threads - 1) / threads;

        threshold_kernel<<<blocks, threads>>>(d_motion_mask, input, marker, width * height);
        cudaMemset(out, 0, width * height * sizeof(bool));

        bool host_changed;
        do {
            bool zero = false;
            cudaMemcpyToSymbol(has_changed, &zero, sizeof(bool));
            reconstruction_tiled<<<reconGrid, reconBlock>>>(input, marker, out, width, height);
            cudaMemcpyFromSymbol(&host_changed, has_changed, sizeof(bool));
        } while (host_changed);

        // convert out to a uint8_t mask for computations
        to_uint8_mask<<<blocks, threads>>>(out, d_motion_mask, width * height);
        // superpose mask/frame with a red color
        masking<<<gridSize, blockSize>>>(dBuffer, d_motion_mask, width, height, pitch, pixel_stride);

        err = cudaMemcpy2D(src_buffer, src_stride, dBuffer, pitch, width * sizeof(rgb), height, cudaMemcpyDefault);
        CHECK_CUDA_ERROR(err);

        cudaFree(dBuffer);

        err = cudaDeviceSynchronize();
        CHECK_CUDA_ERROR(err);

        nb_passes++;
    }
}
