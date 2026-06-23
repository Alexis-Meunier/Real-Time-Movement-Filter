#include "filter_impl.h"
#include <random>
#include <cstdlib>
#include <cstring>

#define RGB_DIFF_THRESHOLD 30
#define MAX_WEIGHTS 1000
#define K 5

// Default subject guidelines for low/high hysteresis thresholds
#define HYSTERESIS_LOW 15
#define HYSTERESIS_HIGH 50

#define MAX_PASSES 1000
#define SPACES 1

struct rgb {
    uint8_t r, g, b;
};

struct reservoir {
    uint8_t r, g, b;
    int w;
};

static int nb_passes = 0;
static reservoir **rs; 
static uint8_t *background_img;
static uint8_t *motion_mask;
static uint8_t *temp_mask;
static bool initialized = false;
static int *stack_x = nullptr, *stack_y = nullptr;

extern "C" {
    inline int safe_dist(uint8_t a, uint8_t b) {
        return std::abs(static_cast<int>(a) - static_cast<int>(b));
    }

    void masking(uint8_t* mask, uint8_t* buffer, int width, int height, int stride, int pixel_stride) {
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                auto mask_value = mask[y * width + x];
                if (mask_value > 0) {
                    rgb* pixel_ptr = (rgb*)(buffer + y * stride + x * pixel_stride);
                    auto toAdd = mask_value / 2;
                    if (255 - pixel_ptr->r < toAdd) {
                        pixel_ptr->r = 255;
                    } else {
                        pixel_ptr->r += toAdd;
                    }
                }
            }
        }
    }

    void hysteresis(uint8_t* mask, int width, int height, uint8_t th_low, uint8_t th_high) {
        std::memcpy(temp_mask, mask, width * height * sizeof(uint8_t));
        std::memset(mask, 0, width * height * sizeof(uint8_t));

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
        std::memcpy(temp_mask, mask, width * height * sizeof(uint8_t));
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
        std::memcpy(temp_mask, mask, width * height * sizeof(uint8_t));
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

    int find_matching_reservoir(const rgb& pixel, reservoir *rs)
    {
        int m_idx = -1;
        for (int i = 0; i < K; ++i)
        {
            if (rs[i].w > 0)
            { 
                if (safe_dist(pixel.r, rs[i].r) < RGB_DIFF_THRESHOLD &&
                    safe_dist(pixel.g, rs[i].g) < RGB_DIFF_THRESHOLD &&
                    safe_dist(pixel.b, rs[i].b) < RGB_DIFF_THRESHOLD) {
                    return i;
                }
            }
            else
            {
                m_idx = i;
            }
        }
        return m_idx;
    }

    void load_background_img(uint8_t* buffer, int width, int height, int stride, int pixel_stride) {
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                rgb* pixel_ptr = (rgb*)(buffer + y * stride + x * pixel_stride);
                rgb p = *pixel_ptr;
                auto curr_rs = rs[y * width + x];

                int m_idx = find_matching_reservoir(p, curr_rs);

                if (m_idx != -1 && curr_rs[m_idx].w > 0) // matching
                {
                    curr_rs[m_idx].w += 1;

                    curr_rs[m_idx].r = ((curr_rs[m_idx].w - 1) * curr_rs[m_idx].r + p.r) / curr_rs[m_idx].w;
                    curr_rs[m_idx].g = ((curr_rs[m_idx].w - 1) * curr_rs[m_idx].g + p.g) / curr_rs[m_idx].w;
                    curr_rs[m_idx].b = ((curr_rs[m_idx].w - 1) * curr_rs[m_idx].b + p.b) / curr_rs[m_idx].w;
                } 
                else if (m_idx != -1 && curr_rs[m_idx].w == 0) // empty slot
                { 
                    curr_rs[m_idx].r = p.r;
                    curr_rs[m_idx].g = p.g;
                    curr_rs[m_idx].b = p.b;
                    curr_rs[m_idx].w = 1;
                } 
                else // no match and no empty slot, perform weighted reservoir replacement
                { 
                    int min_idx = 0;
                    int total_weights = 0;
                    for (int i = 0; i < K; ++i) {
                        total_weights += curr_rs[i].w;
                        if (curr_rs[i].w < curr_rs[min_idx].w) {
                            min_idx = i;
                        }
                    }

                    float rand_val = (float)rand() / (float)RAND_MAX;
                    if (rand_val * total_weights >= curr_rs[min_idx].w) {
                        curr_rs[min_idx].r = p.r;
                        curr_rs[min_idx].g = p.g;
                        curr_rs[min_idx].b = p.b;
                        curr_rs[min_idx].w = 1;
                    }
                }

                // then cap weights to MAX_WEIGHTS
                int max_weight_index = 0;
                int max_weight = curr_rs[0].w;
                for (int i = 1; i < K; ++i) {
                    if (curr_rs[i].w > max_weight) {
                        max_weight = curr_rs[i].w;
                        max_weight_index = i;
                    }
                    if (curr_rs[i].w > MAX_WEIGHTS) {
                        curr_rs[i].w = MAX_WEIGHTS;
                    }
                }

                // and set the background (value of the pixel to return) to the rgb with max weight
                int bg_idx = 3 * (y * width + x);
                background_img[bg_idx] = curr_rs[max_weight_index].r;
                background_img[bg_idx + 1] = curr_rs[max_weight_index].g;
                background_img[bg_idx + 2] = curr_rs[max_weight_index].b;
            }
        }
    }

    void filter_impl(uint8_t* buffer, int width, int height, int stride, int pixel_stride) {
        if (!initialized) {
            rs = new reservoir*[width * height];
            for (int i = 0; i < width * height; ++i) {
                rs[i] = new reservoir[K];
                std::memset(rs[i], 0, K * sizeof(reservoir));
            }
            stack_x = new int[width * height];
            stack_y = new int[width * height];
            background_img = new uint8_t[width * height * 3];
            motion_mask = new uint8_t[width * height];
            temp_mask = new uint8_t[width * height];
            std::memset(stack_x, 0, width * height);
            std::memset(stack_y, 0, width * height);
            std::memset(background_img, 0, width * height * 3);
            std::memset(motion_mask, 0, width * height);
            std::memset(temp_mask, 0, width * height);
            initialized = true;
        }

        if (nb_passes % SPACES == 0) {
            if (nb_passes < MAX_PASSES) {
                load_background_img(buffer, width, height, stride, pixel_stride);
            }
        }
        movement_filter(buffer, motion_mask, width, height, stride, pixel_stride);
        noise_suppression(motion_mask, width, height, 2);
        hysteresis(motion_mask, width, height, HYSTERESIS_LOW, HYSTERESIS_HIGH);
        masking(motion_mask, buffer, width, height, stride, pixel_stride);
        nb_passes++;
    }
}
