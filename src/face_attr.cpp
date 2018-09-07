#include <face_attr.h>

// input is a grayscale image
double GetVarianceOfLaplacianSharpness(const cv::Mat& image) {
    cv::Mat laplacian_output;
    cv::Laplacian(image, laplacian_output, CV_32F);

    cv::Scalar mean, stddev;
    meanStdDev(laplacian_output, mean, stddev);
    return stddev[0];
}
