#include <iostream>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "ImageAnalyzerSingleThread.h"

int main()
{
    cv::Mat gaussianKernel = (cv::Mat_<float>(3, 3) << 1, 2, 1, 2, 4, 2, 1, 2, 1);
    gaussianKernel /= 16;

    auto img = cv::imread(".//..//Resources//testImage.jpg");
    auto imageAnalyser = ImageAnalysis::ImageAnalyzerSingleThread();
    auto imgBlurred = imageAnalyser.GaussianBlur(img, gaussianKernel);

    return 0;
}
