// iou_core.cpp
// Low-level C++ implementation of IoU calculation to satisfy Requirement [Ib]
#include <algorithm>

// We use extern "C" to prevent name mangling, making it easy to call from Python via ctypes
extern "C" {
    
    // Calculates Intersection over Union between two boxes
    // box format: [x1, y1, x2, y2]
    float calculate_iou(float ax1, float ay1, float ax2, float ay2,
                        float bx1, float by1, float bx2, float by2) {
        
        // 1. Calculate intersection coordinates
        // Using low-level std::max/min is faster than Python's checks
        float xi1 = (ax1 > bx1) ? ax1 : bx1;
        float yi1 = (ay1 > by1) ? ay1 : by1;
        float xi2 = (ax2 < bx2) ? ax2 : bx2;
        float yi2 = (ay2 < by2) ? ay2 : by2;

        // 2. Calculate intersection area
        float width = xi2 - xi1;
        float height = yi2 - yi1;

        // If no overlap
        if (width <= 0 || height <= 0) {
            return 0.0f;
        }

        float intersection_area = width * height;

        // 3. Calculate Union area
        float area_a = (ax2 - ax1) * (ay2 - ay1);
        float area_b = (bx2 - bx1) * (by2 - by1);
        float union_area = area_a + area_b - intersection_area;

        // 4. Return IoU
        if (union_area > 0) {
            return intersection_area / union_area;
        }
        return 0.0f;
    }
}