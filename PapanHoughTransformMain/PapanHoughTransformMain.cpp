#include <iostream>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "ImageAnalyzer.h"
#include "ImageAnalyzerSingleThread.h"
#include "ImageAnalyzerMultiThreading.h"

int main()
{
    auto img = cv::imread(".//..//Resources//testImage.jpg");

    //!< Single thread functions
    ImageAnalysis::ImageAnalyzer* imageAnalyzer =new ImageAnalysis::ImageAnalyzerSingleThread();
    {
        cv::Mat imgGray;
        {
            [[maybe_unused]] ImageAnalysis::utils::Timer t("Single thread --- Color conversion");
            imgGray = imageAnalyzer->BGR2Grayscale(img);
        }

        cv::Mat imgBlur;
        {
            [[maybe_unused]] ImageAnalysis::utils::Timer t("Single thread --- Gaussian blur");
            imgBlur = imageAnalyzer->GaussianBlur(imgGray, 5);
        }

        cv::Mat imgBin;
        {
            [[maybe_unused]] ImageAnalysis::utils::Timer t("Single thread --- Otsu thresholding");
            imgBin = imageAnalyzer->OtsuThreshold(imgBlur, 27);
        }

        cv::Mat imgEdges;
        {
            [[maybe_unused]] ImageAnalysis::utils::Timer t("Single thread --- Canny edge filter");
            imgEdges = imageAnalyzer->Canny(imgBin, 0.3, 0.6);
        }

        std::vector<ImageAnalysis::Circle> circles;
        {
            [[maybe_unused]] ImageAnalysis::utils::Timer t("Single thread --- Hough circle transform");
            circles = imageAnalyzer->CircleHoughTransform(imgEdges, 25, 35, 20, 0.75);
        }
#ifndef NDEBUG
        //!< Draw and display circles
        cv::Mat resultIa;
        img.copyTo(resultIa);

        for (const auto& circle : circles)
        {
            cv::circle(resultIa, cv::Point(circle.x, circle.y), circle.radius, cv::Scalar(0, 0, 255), 2);
            std::string text = "(x: " + std::to_string(circle.x) + ", y: " + std::to_string(circle.y) + ", r: " + std::to_string(circle.radius) + ")";
            cv::putText(resultIa, text, cv::Point(circle.x - circle.radius, circle.y - circle.radius), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 2);
        }
        cv::namedWindow("circle image", cv::WINDOW_NORMAL);
        cv::resizeWindow("circle image", cv::Size(resultIa.cols / 2, resultIa.rows / 2));
        cv::imshow("circle image", resultIa);
        cv::waitKey(0);
#endif // !NDEBUG
    }

    //!< Single thread functions
    imageAnalyzer = new ImageAnalysis::ImageAnalyzerMultiThreading();
    {
        cv::Mat imgGray;
        {
            [[maybe_unused]] ImageAnalysis::utils::Timer t("Multithreading --- Color conversion");
            imgGray = imageAnalyzer->BGR2Grayscale(img);
        }

        cv::Mat imgBlur;
        {
            [[maybe_unused]] ImageAnalysis::utils::Timer t("Multithreading --- Gaussian blur");
            imgBlur = imageAnalyzer->GaussianBlur(imgGray, 5);
        }

        cv::Mat imgBin;
        {
            [[maybe_unused]] ImageAnalysis::utils::Timer t("Multithreading --- Otsu thresholding");
            imgBin = imageAnalyzer->OtsuThreshold(imgBlur, 27);
        }

        cv::Mat imgEdges;
        {
            [[maybe_unused]] ImageAnalysis::utils::Timer t("Multithreading --- Canny edge filter");
            imgEdges = imageAnalyzer->Canny(imgBin, 0.3, 0.6);
        }

        std::vector<ImageAnalysis::Circle> circles;
        {
            [[maybe_unused]] ImageAnalysis::utils::Timer t("Multithreading --- Hough circle transform");
            circles = imageAnalyzer->CircleHoughTransform(imgEdges, 25, 35, 20, 0.75);
        }
#ifndef NDEBUG
        //!< Draw and display circles
        cv::Mat resultIa;
        img.copyTo(resultIa);

        for (const auto& circle : circles)
        {
            cv::circle(resultIa, cv::Point(circle.x, circle.y), circle.radius, cv::Scalar(0, 0, 255), 2);
            std::string text = "(x: " + std::to_string(circle.x) + ", y: " + std::to_string(circle.y) + ", r: " + std::to_string(circle.radius) + ")";
            cv::putText(resultIa, text, cv::Point(circle.x - circle.radius, circle.y - circle.radius), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 2);
        }
        cv::namedWindow("circle image", cv::WINDOW_NORMAL);
        cv::resizeWindow("circle image", cv::Size(resultIa.cols / 2, resultIa.rows / 2));
        cv::imshow("circle image", resultIa);
        cv::waitKey(0);
#endif // !NDEBUG
    }

    //!< OpenCV for comparison and benchmark
    {
        std::vector<cv::Vec3f> circlesOpencCV;
        {
            [[maybe_unused]] ImageAnalysis::utils::Timer t("OpenCV --- Hough circle transform");
            cv::Mat imgGray;
            cv::cvtColor(img, imgGray, cv::COLOR_BGR2GRAY);
            cv::HoughCircles(imgGray, circlesOpencCV, cv::HOUGH_GRADIENT_ALT, 1, 20, 100.0, 0.75, 25, 35);
        }
#ifndef NDEBUG
        cv::Mat resultOpenCV;
        img.copyTo(resultOpenCV);

        for (const auto& circle : circlesOpencCV)
        {
            cv::circle(resultOpenCV, cv::Point(circle[0], circle[1]), circle[2], cv::Scalar(0, 0, 255), 2);
        }
        cv::namedWindow("circle image opencv", cv::WINDOW_NORMAL);
        cv::resizeWindow("circle image opencv", cv::Size(resultOpenCV.cols / 2, resultOpenCV.rows / 2));
        cv::imshow("circle image opencv", resultOpenCV);
        cv::waitKey(0);
#endif // !NDEBUG
    }

    return 0;
}
