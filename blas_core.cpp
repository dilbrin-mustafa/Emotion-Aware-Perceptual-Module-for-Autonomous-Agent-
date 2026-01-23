#include <vector>
#include <cmath>
#include <algorithm>

extern "C" {

    // Computes Cosine Distance Matrix between two sets of vectors
    // A: [rows_a x cols]
    // B: [rows_b x cols]
    // Output: [rows_a x rows_b] where out[i][j] is distance between A[i] and B[j]
    // Distance = 1 - (dot_product / (norm_a * norm_b))
    void compute_cosine_distance(float* A, float* B, int rows_a, int rows_b, int cols, float* output) {
        
        // 1. Pre-compute norms for Matrix A
        std::vector<float> norms_a(rows_a);
        for (int i = 0; i < rows_a; ++i) {
            float sum = 0.0f;
            for (int k = 0; k < cols; ++k) {
                float val = A[i * cols + k];
                sum += val * val;
            }
            norms_a[i] = std::sqrt(sum);
        }

        // 2. Pre-compute norms for Matrix B
        std::vector<float> norms_b(rows_b);
        for (int i = 0; i < rows_b; ++i) {
            float sum = 0.0f;
            for (int k = 0; k < cols; ++k) {
                float val = B[i * cols + k];
                sum += val * val;
            }
            norms_b[i] = std::sqrt(sum);
        }

        // 3. Compute Distance Matrix (The "BLAS" Heavy Lifting)
        for (int i = 0; i < rows_a; ++i) {
            for (int j = 0; j < rows_b; ++j) {
                float dot = 0.0f;
                for (int k = 0; k < cols; ++k) {
                    dot += A[i * cols + k] * B[j * cols + k];
                }
                
                float denominator = norms_a[i] * norms_b[j];
                float cosine_sim = (denominator > 1e-6) ? (dot / denominator) : 0.0f;
                
                // Clamp to -1..1 to avoid numerical errors
                if (cosine_sim > 1.0f) cosine_sim = 1.0f;
                if (cosine_sim < -1.0f) cosine_sim = -1.0f;

                // Distance = 1 - Similarity
                output[i * rows_b + j] = 1.0f - cosine_sim;
            }
        }
    }
}