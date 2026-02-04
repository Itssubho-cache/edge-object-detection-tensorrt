#include <vector>

struct BoundingBox {
    float x1, y1, x2, y2;
    float score;
    int class_id;
};

std::vector<BoundingBox> decodeDetections(const float* output) {
    std::vector<BoundingBox> boxes;

    // TODO:
    // - Decode raw model outputs
    // - Apply confidence threshold
    // - Implement Non-Max Suppression (NMS)

    return boxes;
}
