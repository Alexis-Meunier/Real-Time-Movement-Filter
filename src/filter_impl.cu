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
#define MAX_WIDTH 2160
#define MAX_HEIGHT 1280

#define HYSTERESIS_LOW 12
#define HYSTERESIS_HIGH 30

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

struct reservoir {
    uint8_t r, g, b;
    int w;
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

__device__ static reservoir d_global_rs[MAX_WIDTH * MAX_HEIGHT][K]; 
__device__ static uint8_t d_background_img[MAX_WIDTH * MAX_HEIGHT * 3];
__device__ static uint8_t d_motion_mask[MAX_WIDTH * MAX_HEIGHT];
__device__ static uint8_t d_temp_mask[MAX_WIDTH * MAX_HEIGHT];

// Should use the static Image<curandState > rng_states; and
// curand_init(seed, global_pixel_pos, 0, &randState_row[x]);
// But for now
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

__device__ int find_matching_reservoir(rgb pixel, reservoir* rs, int& empty_idx) {
    empty_idx = -1;
    for (int i = 0; i < K; ++i) {
        if (rs[i].w > 0) { 
            if (safe_dist(pixel.r, rs[i].r) < RGB_DIFF_THRESHOLD &&
                safe_dist(pixel.g, rs[i].g) < RGB_DIFF_THRESHOLD &&
                safe_dist(pixel.b, rs[i].b) < RGB_DIFF_THRESHOLD) {
                return i;
            }
        } else if (empty_idx == -1) {
            empty_idx = i;
        }
    }
    return -1;
}

__global__ void load_background_img(uint8_t* buffer, int width, int height, int stride, int pixel_stride) {
    int y = blockIdx.y * blockDim.y + threadIdx.y; 
    int x = blockIdx.x * blockDim.x + threadIdx.x;

    if (x >= width || y >= height) return; 

    rgb* pixel_ptr = (rgb*)(buffer + y * stride + x * pixel_stride);
    rgb p = *pixel_ptr;

    int pixel_id = y * width + x;
    reservoir* rs = d_global_rs[pixel_id];

    int empty_idx = -1;
    int m_idx = find_matching_reservoir(p, rs, empty_idx);

    if (m_idx != -1) { 
        rs[m_idx].w += 1;
        if (rs[m_idx].w > MAX_WEIGHTS) rs[m_idx].w = MAX_WEIGHTS;

        rs[m_idx].r = ((rs[m_idx].w - 1) * rs[m_idx].r + p.r) / rs[m_idx].w;
        rs[m_idx].g = ((rs[m_idx].w - 1) * rs[m_idx].g + p.g) / rs[m_idx].w;
        rs[m_idx].b = ((rs[m_idx].w - 1) * rs[m_idx].b + p.b) / rs[m_idx].w;
    } 
    else if (empty_idx != -1) { 
        rs[empty_idx].r = p.r;
        rs[empty_idx].g = p.g;
        rs[empty_idx].b = p.b;
        rs[empty_idx].w = 1;
    } 
    else { 
        int min_idx = 0;
        int total_weights = 0;
        for (int i = 0; i < K; ++i) {
            total_weights += rs[i].w;
            if (rs[i].w < rs[min_idx].w) {
                min_idx = i;
            }
        }

        // Initialize unique state seed per pixel coordinate
        uint32_t rng_state = y * width + x + 1; 
        float rand_val = (float)(xorshift32(&rng_state) % 10000) / 10000.0f;
        
        if (rand_val * total_weights >= rs[min_idx].w) {
            rs[min_idx].r = p.r;
            rs[min_idx].g = p.g;
            rs[min_idx].b = p.b;
            rs[min_idx].w = 1;
        }
    }

    int max_weight_index = 0;
    for (int i = 1; i < K; ++i) {
        if (rs[i].w > rs[max_weight_index].w) {
            max_weight_index = i;
        }
    }
    
    int bg_idx = pixel_id * 3;
    d_background_img[bg_idx]     = rs[max_weight_index].r;
    d_background_img[bg_idx + 1] = rs[max_weight_index].g;
    d_background_img[bg_idx + 2] = rs[max_weight_index].b;
}

__global__ void movement_filter(uint8_t* buffer, int width, int height, int stride, int pixel_stride) {
    int y = blockIdx.y * blockDim.y + threadIdx.y; 
    int x = blockIdx.x * blockDim.x + threadIdx.x;

    if (x >= width || y >= height) return;

    rgb* pixel_ptr = (rgb*)(buffer + y * stride + x * pixel_stride);
    int bg_idx = (y * width + x) * 3;
    
    int dr = safe_dist(pixel_ptr->r, d_background_img[bg_idx]);
    int dg = safe_dist(pixel_ptr->g, d_background_img[bg_idx + 1]);
    int db = safe_dist(pixel_ptr->b, d_background_img[bg_idx + 2]);
    
    int diff = (dr + dg + db) / 3;
    d_motion_mask[y * width + x] = static_cast<uint8_t>(diff > 255 ? 255 : diff);
}

__global__ void erosion(const uint8_t* src, uint8_t* dst, int width, int height, int opening_size) {
    int y = blockIdx.y * blockDim.y + threadIdx.y; 
    int x = blockIdx.x * blockDim.x + threadIdx.x;

    if (x >= width || y >= height) return;

    uint8_t min_v = 255;
    for (int dy = -opening_size; dy <= opening_size; ++dy) {
        for (int dx = -opening_size; dx <= opening_size; ++dx) {
            int nx = x + dx;
            int ny = y + dy;
            if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                uint8_t val = src[ny * width + nx];
                if (val < min_v) min_v = val;
            }
        }
    }
    dst[y * width + x] = min_v;
}

__global__ void dilation(const uint8_t* src, uint8_t* dst, int width, int height, int opening_size) {
    int y = blockIdx.y * blockDim.y + threadIdx.y; 
    int x = blockIdx.x * blockDim.x + threadIdx.x;

    if (x >= width || y >= height) return;

    uint8_t max_v = 0;
    for (int dy = -opening_size; dy <= opening_size; ++dy) {
        for (int dx = -opening_size; dx <= opening_size; ++dx) {
            int nx = x + dx;
            int ny = y + dy;
            if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                uint8_t val = src[ny * width + nx];
                if (val > max_v) max_v = val;
            }
        }
    }
    dst[y * width + x] = max_v;
}

__device__ bool has_changed;

__global__ void reconstruction(input, marker, out) {
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (out[p] || !input[p])
        return;
    if (marker[p]) {
        out[p] = true;
        has_changed = true;
        return;
    }
    for (int q : neighbors(p))
    {
        if (out[q]) {
            out[p] = true;
            has_changed = true;
        }
    }
}

__global__ void masking(uint8_t* buffer, int width, int height, int stride, int pixel_stride) {
    int y = blockIdx.y * blockDim.y + threadIdx.y; 
    int x = blockIdx.x * blockDim.x + threadIdx.x;

    if (x >= width || y >= height) return;

    if (d_motion_mask[y * width + x] > 0) {
        rgb* pixel_ptr = (rgb*)(buffer + y * stride + x * pixel_stride);
        pixel_ptr->r = 255; 
    }
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
        load_logo();

        assert(sizeof(rgb) == pixel_stride);
        std::byte* dBuffer;
        size_t pitch;

        cudaError_t err;
        
        err = cudaMallocPitch(&dBuffer, &pitch, width * sizeof(rgb), height);
        CHECK_CUDA_ERROR(err);

        err = cudaMemcpy2D(dBuffer, pitch, src_buffer, src_stride, width * sizeof(rgb), height, cudaMemcpyDefault);
        CHECK_CUDA_ERROR(err);

        err = cudaMemcpy2D(dBuffer, pitch, src_buffer, src_stride, width * sizeof(rgb), height, cudaMemcpyHostToDevice);
        CHECK_CUDA_ERROR(err);

        dim3 blockSize(16, 16);
        dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

        load_background_img<<<gridSize, blockSize>>>(dBuffer, width, height, pitch, pixel_stride);
        movement_filter<<<gridSize, blockSize>>>(dBuffer, width, height, pitch, pixel_stride);
        erosion<<<gridSize, blockSize>>>(d_motion_mask, d_temp_mask, width, height, 1);
        dilation<<<gridSize, blockSize>>>(d_temp_mask, d_motion_mask, width, height, 1);

        // IDK what input and black_image are but w/e
        out = black_image;
        has_changed = false;
        while (has_changed)
            reconstruction<<<>>>(input, marker, out);

        masking<<<gridSize, blockSize>>>(dBuffer, width, height, pitch, pixel_stride);

        err = cudaMemcpy2D(src_buffer, src_stride, dBuffer, pitch, width * sizeof(rgb), height, cudaMemcpyDefault);
        CHECK_CUDA_ERROR(err);

        cudaFree(dBuffer);

        err = cudaDeviceSynchronize();
        CHECK_CUDA_ERROR(err);
    }   
}
