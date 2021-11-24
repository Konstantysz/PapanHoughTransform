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

    cv::Mat imgGray;
    {
        [[maybe_unused]] ImageAnalysis::utils::Timer t("Color converion");
        imgGray = imageAnalyser.BGR2Grayscale(img);
    }

    cv::Mat imgBlur;
    {
        [[maybe_unused]] ImageAnalysis::utils::Timer t("Gaussian blur");
        imgBlur = imageAnalyser.GaussianBlur(imgGray, 5);
    }

    cv::Mat imgBin;
    {
        [[maybe_unused]] ImageAnalysis::utils::Timer t("Otsu thresholding");
        imgBin = imageAnalyser.OtsuThreshold(imgBlur, 27);
    }

    cv::Mat imgEdges;
    {
        [[maybe_unused]] ImageAnalysis::utils::Timer t("Canny edge filter");
        imgEdges = imageAnalyser.Canny(imgBin, 0.3, 0.6);
    }

    std::vector<ImageAnalysis::Circle> circles;
    {
        [[maybe_unused]] ImageAnalysis::utils::Timer t("Hough circle transform");
        circles = imageAnalyser.CircleHoughTransform(imgEdges, 20, 30, 10);
    }

    //!< Draw and display circles
    for (const auto& circle : circles)
    {
        cv::circle(img, cv::Point(circle.x, circle.y), circle.radius, cv::Scalar(0, 0, 255), 2);
        std::string text = "(x: " + std::to_string(circle.x) + ", y: " + std::to_string(circle.y) + ", r: " + std::to_string(circle.radius) + ")";
        cv::putText(img, text, cv::Point(circle.x - circle.radius, circle.y - circle.radius), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 2);
    }
    cv::namedWindow("circle image", cv::WINDOW_NORMAL);
    cv::resizeWindow("circle image", cv::Size(img.cols / 2, img.rows / 2));
    cv::imshow("circle image", img);
    cv::waitKey(0);

    return 0;
}
