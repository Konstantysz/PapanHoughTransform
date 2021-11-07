#include <iostream>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "ImageAnalyzerSingleThread.h"

int main()
{
    auto img = cv::imread(".//..//Resources//testImage.jpg");
    auto imageAnalyser = ImageAnalysis::ImageAnalyzerSingleThread();
    auto imgBlurred = imageAnalyser.GaussianBlur(img, 19);

    cv::imshow("blurred difference", imgBlurred - img);
    cv::waitKey(0);

    return 0;
}
