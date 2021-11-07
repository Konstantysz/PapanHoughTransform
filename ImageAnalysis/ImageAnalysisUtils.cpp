#include "pch.h"
#include "ImageAnalysisUtils.h"

namespace ImageAnalysis
{
    namespace utils 
    {
        void SingleConvolve(const cv::Mat& input, cv::Mat& output, const cv::Mat& kernel, const int& i, const int& j)
        {
            cv::Mat roi;
            int halfKernelSize = (kernel.cols - 1) / 2;
            input(cv::Rect(i - halfKernelSize, j - halfKernelSize, kernel.cols, kernel.rows)).convertTo(roi, CV_32F);
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
        
        cv::Mat FilterKernelGenerator(int size)
        {
            if (size % 2 == 0)
            {
                throw "Kurwa daj nieparzystš";
            }

            // initialising standard deviation to 1.0
            double sigma = 1.0;
            double s = 2.0 * sigma * sigma;

            // sum is for normalization
            double sum = 0.0;

            int halfKernelSize = (size - 1) / 2;
            cv::Mat kernel = cv::Mat::zeros(cv::Size(size, size), CV_32F);

            for (int i = -halfKernelSize; i <= halfKernelSize; i++) {
                for (int j = -halfKernelSize; j <= halfKernelSize; j++) {
                    auto r = sqrt(i * i + j * j);
                    kernel.at<float>(j + halfKernelSize, i + halfKernelSize) = (exp(-(r * r) / s)) / (CV_PI * s);
                    sum += kernel.at<float>(j + halfKernelSize, i + halfKernelSize);
                }
            }

            // normalising the Kernel
            return kernel /= sum;
        }
    } // namespace utils
} // namespace ImageAnalysis
