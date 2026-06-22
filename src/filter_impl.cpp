#include "filter_impl.h"
#include <random>
#include <cstdlib>
#include <cstring>

#define RGB_DIFF_THRESHOLD 30
#define MAX_WEIGHTS 1000
#define K 5

// Default subject guidelines for low/high hysteresis thresholds
#define HYSTERESIS_LOW 3
#define HYSTERESIS_HIGH 30

struct rgb {
    uint8_t r, g, b;
};

struct reservoir {
    uint8_t r, g, b;
    int w;
};

#define MAX_WIDTH 2160
#define MAX_HEIGHT 1280

static reservoir global_rs[MAX_WIDTH * MAX_HEIGHT][K] = {0}; 
static uint8_t background_img[MAX_WIDTH * MAX_HEIGHT * 3] = {0};
static uint8_t motion_mask[MAX_WIDTH * MAX_HEIGHT] = {0};
static uint8_t temp_mask[MAX_WIDTH * MAX_HEIGHT] = {0};

extern "C" {
    inline int safe_dist(uint8_t a, uint8_t b) {
        return std::abs(static_cast<int>(a) - static_cast<int>(b));
    }

    void masking(uint8_t* mask, uint8_t* buffer, int width, int height, int stride, int pixel_stride) {
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                if (mask[y * width + x] > 0) {
                    rgb* pixel_ptr = (rgb*)(buffer + y * stride + x * pixel_stride);
                    pixel_ptr->r = 255; 
                }
            }
        }
    }

    void hysteresis(uint8_t* mask, int width, int height, uint8_t th_low, uint8_t th_high) {
        std::memcpy(temp_mask, mask, width * height);
        std::memset(mask, 0, width * height);

        static int stack_x[MAX_WIDTH * MAX_HEIGHT];
        static int stack_y[MAX_WIDTH * MAX_HEIGHT];
        int top = 0;

        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                int idx = y * width + x;
                if (temp_mask[idx] >= th_high) {
                    mask[idx] = 255;
                    stack_x[top] = x;
                    stack_y[top] = y;
                    top++;
                }
            }
        }

        while (top > 0) {
            top--;
            int cx = stack_x[top];
            int cy = stack_y[top];

            for (int dy = -1; dy <= 1; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {
                    int nx = cx + dx;
                    int ny = cy + dy;

                    if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                        int n_idx = ny * width + nx;
                        if (mask[n_idx] == 0 && temp_mask[n_idx] >= th_low) {
                            mask[n_idx] = 255;
                            stack_x[top] = nx;
                            stack_y[top] = ny;
                            top++;
                        }
                    }
                }
            }
        }
    }

    void dilation(uint8_t* mask, int width, int height, int opening_size) {
        std::memcpy(temp_mask, mask, width * height);
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                uint8_t max_v = 0;
                for (int dy = -opening_size; dy <= opening_size; ++dy) {
                    for (int dx = -opening_size; dx <= opening_size; ++dx) {
                        int nx = x + dx;
                        int ny = y + dy;
                        if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                            uint8_t val = temp_mask[ny * width + nx];
                            if (val > max_v) max_v = val;
                        }
                    }
                }
                mask[y * width + x] = max_v;
            }
        }
    }

    void erosion(uint8_t* mask, int width, int height, int opening_size) {
        std::memcpy(temp_mask, mask, width * height);
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                uint8_t min_v = 255;
                for (int dy = -opening_size; dy <= opening_size; ++dy) {
                    for (int dx = -opening_size; dx <= opening_size; ++dx) {
                        int nx = x + dx;
                        int ny = y + dy;
                        if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                            uint8_t val = temp_mask[ny * width + nx];
                            if (val < min_v) min_v = val;
                        }
                    }
                }
                mask[y * width + x] = min_v;
            }
        }
    }

    void noise_suppression(uint8_t* mask, int width, int height, int opening_size) {
        erosion(mask, width, height, opening_size);
        dilation(mask, width, height, opening_size);
    }

    void movement_filter(uint8_t* buffer, uint8_t* mask, int width, int height, int stride, int pixel_stride) {
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                rgb* pixel_ptr = (rgb*)(buffer + y * stride + x * pixel_stride);
                int bg_idx = (y * width + x) * 3;
                
                int dr = safe_dist(pixel_ptr->r, background_img[bg_idx]);
                int dg = safe_dist(pixel_ptr->g, background_img[bg_idx + 1]);
                int db = safe_dist(pixel_ptr->b, background_img[bg_idx + 2]);
                
                int diff = (dr + dg + db) / 3;
                mask[y * width + x] = static_cast<uint8_t>(diff > 255 ? 255 : diff);
            }
        }
    }

    int find_matching_reservoir(rgb pixel, reservoir* rs, int& empty_idx) {
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

    void filter_impl(uint8_t* buffer, int width, int height, int stride, int pixel_stride) {
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                rgb* pixel_ptr = (rgb*)(buffer + y * stride + x * pixel_stride);
                rgb p = *pixel_ptr;

                int pixel_id = y * width + x;
                reservoir* rs = global_rs[pixel_id];

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

                    float rand_val = (float)rand() / (float)RAND_MAX;
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
                background_img[bg_idx]     = rs[max_weight_index].r;
                background_img[bg_idx + 1] = rs[max_weight_index].g;
                background_img[bg_idx + 2] = rs[max_weight_index].b;
            }
        }

        movement_filter(buffer, motion_mask, width, height, stride, pixel_stride);
        noise_suppression(motion_mask, width, height, 1);
        hysteresis(motion_mask, width, height, HYSTERESIS_LOW, HYSTERESIS_HIGH);
        masking(motion_mask, buffer, width, height, stride, pixel_stride);
    }
}