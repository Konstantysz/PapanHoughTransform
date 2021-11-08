#include <iostream>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "ImageAnalyzerSingleThread.h"

int main()
{
    auto img = cv::imread(".//..//Resources//testImage.jpg");
    //auto img = cv::imread(".//..//Resources//gayMaria.png");
    auto imageAnalyser = ImageAnalysis::ImageAnalyzerSingleThread();

    auto imgGray = imageAnalyser.BGR2Grayscale(img);
    cv::namedWindow("grayscale image", cv::WINDOW_KEEPRATIO);
    cv::imshow("grayscale image", imgGray);
    cv::waitKey(0);

    auto imgEdges = imageAnalyser.Canny(imgGray, 0.1, 0.3);
    cv::namedWindow("edge image", cv::WINDOW_KEEPRATIO);
    cv::imshow("edge image", imgEdges);
    cv::waitKey(0);

    auto circles = imageAnalyser.CircleHoughTransform(imgEdges, 20, 50);
    for (const auto& circle : circles)
    {
        cv::circle(img, circle.second, circle.first, cv::Scalar(0, 0, 255));
    }
    cv::namedWindow("circle image", cv::WINDOW_KEEPRATIO);
    cv::imshow("circle image", img);
    cv::waitKey(0);

    return 0;
}
