#include <opencv2/opencv.hpp>

// Resize, normalize, and convert HWC → CHW
void preprocessImage(const cv::Mat& input, float* output) {
    cv::Mat resized;
    cv::resize(input, resized, cv::Size(224, 224));

    resized.convertTo(resized, CV_32F, 1.0 / 255);

    // HWC → CHW
    std::vector<cv::Mat> channels(3);
    cv::split(resized, channels);

    int channelSize = 224 * 224;
    for (int i = 0; i < 3; i++) {
        memcpy(output + i * channelSize,
               channels[i].data,
               channelSize * sizeof(float));
    }
}

// TODO:
// - Add mean/std normalization
// - Support batch preprocessing
