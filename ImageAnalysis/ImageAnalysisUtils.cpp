#include "pch.h"
#include "ImageAnalysisUtils.h"

void ImageAnalysis::utils::SingleConvolve(const cv::Mat& input, cv::Mat& output, const cv::Mat& kernel, const int& i, const int& j)
{
    cv::Mat roi;
    input(cv::Rect(i - 1, j - 1, kernel.cols, kernel.rows)).convertTo(roi, CV_32F);
    auto newValue = roi.dot(kernel);

    switch (input.type()) {
    case CV_8U:  
        output.at<uchar>(j, i) = newValue;
        break;
    case CV_8S:  
        output.at<schar>(j, i) = newValue;
        break;
    case CV_16U: 
        output.at<ushort>(j, i) = newValue;
        break;
    case CV_16S: 
        output.at<short>(j, i) = newValue;
        break;
    case CV_32S: 
        output.at<int>(j, i) = newValue;
        break;
    case CV_32F: 
        output.at<float>(j, i) = newValue;
        break;
    case CV_64F: 
        output.at<double>(j, i) = newValue;
        break;
    default:     
        throw "KUUUUURWA";
    }

}
