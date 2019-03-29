#include <image_quality.h>

/*
double GetImageQuality(IplImage* img, int left, int top, int right, int bottom) {
    double temp = 0;
    double DR = 0;
    int i,j;
    int height=bottom-top;
    int width=right-left;
    int step=img->widthStep/sizeof(uchar);
    uchar *data=(uchar*)img->imageData;
    double num = width*height;
    for(i=top;i<bottom;i++) {
        for(j=left;j<right;j++) {
            temp += sqrt((pow((double)(data[(i+1)*step+j]-data[i*step+j]),2) + pow((double)(data[i*step+j+1]-data[i*step+j]),2)));
            temp += abs(data[(i+1)*step+j]-data[i*step+j])+abs(data[i*step+j+1]-data[i*step+j]);
        }
    }
    DR = temp/num;
    return DR;
}
*/

double GetNormOfDerivativesBlurriness(const cv::Mat& image) {
    cv::Mat Gx;
    cv::Mat Gy;
    cv::Sobel(image, Gx, CV_32F, 1, 0);
    cv::Sobel(image, Gy, CV_32F, 0, 1);
    double normGx = cv::norm(Gx);
    double normGy = cv::norm(Gy);
    // write_log("norm: " + std::to_string(normGx));
    // write_log("norm: " + std::to_string(normGy));
    double sumSq = normGx * normGx + normGy * normGy;
    // std::cout << "Image size: " << image.size() << std::endl;
    // std::cout << "Image size area: " + std::to_string(image.size().area()) << std::endl;
    // double blur = std::sqrt(sumSq) / image.size().area();
    double blur = std::sqrt(sumSq) / magic;
    // std::cout << "blurriness: " << blur << std::endl;
    return blur;
}

// input is a grayscale image
double GetVarianceOfLaplacianSharpness(const cv::Mat& image) {
    cv::Mat laplacian_output;
    cv::Laplacian(image, laplacian_output, CV_32F);

    cv::Scalar mean, stddev;
    cv::meanStdDev(laplacian_output, mean, stddev);
    return stddev[0];
}
