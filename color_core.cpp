#include <vector>
#include <cmath>
#include <algorithm>
#include <limits>

struct Pixel {
    unsigned char b, g, r;
};

struct Centroid {
    float b, g, r;
    int count;
};

extern "C" {

    // K-Means Clustering to find dominant color
    // Returns the centroid of the largest cluster (most common color)
    void get_dominant_color_kmeans(unsigned char* data, int width, int height, int stride, int k, int* output_bgr) {
        if (width <= 0 || height <= 0 || k <= 0) {
            output_bgr[0] = 0; output_bgr[1] = 0; output_bgr[2] = 0;
            return;
        }

        std::vector<Pixel> pixels;
        pixels.reserve(width * height);

        // 1. Flatten image data (and ignore extremely dark/bright noise if desired)
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                int idx = y * stride + x * 3;
                unsigned char b = data[idx];
                unsigned char g = data[idx + 1];
                unsigned char r = data[idx + 2];
                pixels.push_back({b, g, r});
            }
        }

        if (pixels.empty()) {
            output_bgr[0] = 0; output_bgr[1] = 0; output_bgr[2] = 0;
            return;
        }

        // 2. Initialize Centroids (simplistic initialization)
        std::vector<Centroid> centroids(k);
        for (int i = 0; i < k; ++i) {
            Pixel p = pixels[i * pixels.size() / k];
            centroids[i] = {(float)p.b, (float)p.g, (float)p.r, 0};
        }

        std::vector<int> labels(pixels.size());
        int iterations = 10; // 10 iterations is usually enough for video approximations

        // 3. K-Means Loop
        for (int iter = 0; iter < iterations; ++iter) {
            // Reset counts
            for (auto& c : centroids) c.count = 0;

            // Assignment Step
            for (size_t i = 0; i < pixels.size(); ++i) {
                float min_dist = std::numeric_limits<float>::max();
                int best_k = 0;
                
                for (int j = 0; j < k; ++j) {
                    float db = pixels[i].b - centroids[j].b;
                    float dg = pixels[i].g - centroids[j].g;
                    float dr = pixels[i].r - centroids[j].r;
                    float dist = db*db + dg*dg + dr*dr;
                    
                    if (dist < min_dist) {
                        min_dist = dist;
                        best_k = j;
                    }
                }
                labels[i] = best_k;
            }

            // Update Step
            std::vector<Centroid> new_sums(k, {0, 0, 0, 0});
            for(size_t i = 0; i < pixels.size(); ++i) {
                int label = labels[i];
                new_sums[label].b += pixels[i].b;
                new_sums[label].g += pixels[i].g;
                new_sums[label].r += pixels[i].r;
                new_sums[label].count++;
            }

            for(int j=0; j<k; ++j) {
                if(new_sums[j].count > 0) {
                    centroids[j].b = new_sums[j].b / new_sums[j].count;
                    centroids[j].g = new_sums[j].g / new_sums[j].count;
                    centroids[j].r = new_sums[j].r / new_sums[j].count;
                    centroids[j].count = new_sums[j].count;
                }
            }
        }

        // 4. Find largest cluster
        int max_count = -1;
        int best_centroid = 0;
        for (int i = 0; i < k; ++i) {
            if (centroids[i].count > max_count) {
                max_count = centroids[i].count;
                best_centroid = i;
            }
        }

        output_bgr[0] = (int)centroids[best_centroid].b;
        output_bgr[1] = (int)centroids[best_centroid].g;
        output_bgr[2] = (int)centroids[best_centroid].r;
    }
}