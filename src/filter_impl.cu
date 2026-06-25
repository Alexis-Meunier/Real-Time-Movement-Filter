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
__device__ static uint8_t *d_background_img;
__device__ static uint8_t *d_motion_mask;
__device__ static uint8_t *d_temp_mask;
__device__ bool has_changed = false;
static bool *input = nullptr;
static bool *marker = nullptr;
static bool *out = nullptr;
static bool initialized = false;

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

__global__ void threshold_kernel(uint8_t* diff, bool* input, bool* marker,
                                  int n, uint8_t th_low, uint8_t th_high) {
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p < n) {
        input[p]  = diff[p] >= th_low;
        marker[p] = diff[p] >= th_high;
    }
}

__global__ void to_uint8_mask(bool* out, uint8_t* mask, int n) {
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p < n) mask[p] = out[p] ? 255 : 0;
}

__global__ void reconstruction(bool* input, bool* marker, bool* out, int width, int height, int n) {
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= n || out[p] || !input[p]) return;
    if (marker[p]) { out[p] = true; has_changed = true; return; }

    int x = p % width, y = p / width;
    for (int dy = -1; dy <= 1; ++dy)
        for (int dx = -1; dx <= 1; ++dx) {
            int nx = x + dx, ny = y + dy;
            if (nx < 0 || nx >= width || ny < 0 || ny >= height) continue;
            if (out[ny * width + nx]) { out[p] = true; has_changed = true; }
        }
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

__device__ int find_matching_reservoir(const rgb& pixel, reservoir* rs) {
    int m_idx = -1;
    for (int i = 0; i < K; ++i) {
        if (rs[i].w > 0) { 
            if (safe_dist(pixel.r, rs[i].r) < RGB_DIFF_THRESHOLD &&
                safe_dist(pixel.g, rs[i].g) < RGB_DIFF_THRESHOLD &&
                safe_dist(pixel.b, rs[i].b) < RGB_DIFF_THRESHOLD) {
                return i;
            }
        } else if (m_idx == -1) {
            m_idx = i;
        }
    }
    return m_idx;
}

__global__ void load_background_img(uint8_t* buffer, int width, int height, int stride, int pixel_stride) {
    int y = blockIdx.y * blockDim.y + threadIdx.y; 
    int x = blockIdx.x * blockDim.x + threadIdx.x;

    if (x >= width || y >= height) return; 

    rgb* pixel_ptr = (rgb*)(buffer + y * stride + x * pixel_stride);
    rgb p = *pixel_ptr;

    int pixel_id = y * width + x;
    reservoir* rs = d_global_rs[pixel_id];

    int m_idx = find_matching_reservoir(p, rs);

    if (m_idx != -1 && rs[m_idx].w > 0) { 
        rs[m_idx].w += 1;
        if (rs[m_idx].w > MAX_WEIGHTS) rs[m_idx].w = MAX_WEIGHTS;

        rs[m_idx].r = ((rs[m_idx].w - 1) * rs[m_idx].r + p.r) / rs[m_idx].w;
        rs[m_idx].g = ((rs[m_idx].w - 1) * rs[m_idx].g + p.g) / rs[m_idx].w;
        rs[m_idx].b = ((rs[m_idx].w - 1) * rs[m_idx].b + p.b) / rs[m_idx].w;
    } 
    else if (m_idx != -1 && rs[m_idx].w == 0) { 
        rs[m_idx].r = p.r;
        rs[m_idx].g = p.g;
        rs[m_idx].b = p.b;
        rs[m_idx].w = 1;
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

    // then cap weights to MAX_WEIGHTS
    int max_weight_index = 0;
    int max_weight = rs[0].w;
    for (int i = 1; i < K; ++i) {
        if (rs[i].w > max_weight) {
            max_weight = rs[i].w;
            max_weight_index = i;
        }
        if (rs[i].w > MAX_WEIGHTS) {
            rs[i].w = MAX_WEIGHTS;
        }
    }
    
    int bg_idx = pixel_id * 3;
    d_background_img[bg_idx] = rs[max_weight_index].r;
    d_background_img[bg_idx + 1] = rs[max_weight_index].g;
    d_background_img[bg_idx + 2] = rs[max_weight_index].b;
}

static Image<curandState > rng_states;
curand_init(seed, global_pixel_pos, 0, &randState_row[x]);
float rand_val = curand_uniform(&randState_row[x]);

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
            err = cudaMalloc(&d_background_img, width * height * 3 * sizeof(uint8_t));
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
            initialized = true;
        }

        assert(sizeof(rgb) == pixel_stride);
        std::uint8_t* dBuffer;
        size_t pitch;

        
        err = cudaMallocPitch(&dBuffer, &pitch, width * sizeof(rgb), height);
        CHECK_CUDA_ERROR(err);

        err = cudaMemcpy2D(dBuffer, pitch, src_buffer, src_stride, width * sizeof(rgb), height, cudaMemcpyDefault);
        CHECK_CUDA_ERROR(err);

        err = cudaMemcpy2D(dBuffer, pitch, src_buffer, src_stride, width * sizeof(rgb), height, cudaMemcpyHostToDevice);
        CHECK_CUDA_ERROR(err);

        dim3 blockSize(16, 16);
        dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

        // Compute background
        load_background_img<<<gridSize, blockSize>>>(dBuffer, width, height, pitch, pixel_stride);
        // Get the movement filter
        movement_filter<<<gridSize, blockSize>>>(dBuffer, width, height, pitch, pixel_stride);
        // Noise Supprsion
        erosion<<<gridSize, blockSize>>>(d_motion_mask, d_temp_mask, width, height, 1);
        dilation<<<gridSize, blockSize>>>(d_temp_mask, d_motion_mask, width, height, 1);

        int threads = 256;
        int blocks = (width * height + threads - 1) / threads;
        // Compute input and marker
        threshold_kernel<<<blocks, threads>>>(d_motion_mask, input, marker, width * height, HYSTERESIS_LOW, HYSTERESIS_HIGH);
        // Set output to false everywhere
        cudaMemset(out, 0, width * height * sizeof(bool));
        // Hysteresis
        bool host_changed;
        do {
            bool zero = false;
            cudaMemcpyToSymbol(&has_changed, &zero, sizeof(bool));
            reconstruction<<<blocks, threads>>>(input, marker, out, width * height);
            cudaMemcpyFromSymbol(&host_changed, &has_changed, sizeof(bool));
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
    }   
}
