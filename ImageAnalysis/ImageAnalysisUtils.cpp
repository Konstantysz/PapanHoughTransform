#include "pch.h"
#include "ImageAnalysisUtils.h"
#include <cmath>

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
                output.at<uchar>(j, i) = uchar(newValue);
                break;
            case CV_8S:
                output.at<schar>(j, i) = schar(newValue);
                break;
            case CV_16U:
                output.at<ushort>(j, i) = ushort(newValue);
                break;
            case CV_16S:
                output.at<short>(j, i) = short(newValue);
                break;
            case CV_32S:
                output.at<int>(j, i) = int(newValue);
                break;
            case CV_32F:
                output.at<float>(j, i) = float(newValue);
                break;
            case CV_64F:
                output.at<double>(j, i) = double(newValue);
                break;
            default:
                throw "KUUUUURWA";
            }


    }
        
        cv::Mat Convolve(const cv::Mat& input, const cv::Mat& kernel)
        {
            int halfKernelSize = (kernel.cols - 1) / 2;

            // Zero-pad matrix
            cv::Mat zeroPadImage = cv::Mat::zeros(cv::Size(input.cols + 2 * halfKernelSize, input.rows + 2 * halfKernelSize), input.type());
            input.copyTo(zeroPadImage(cv::Rect(halfKernelSize, halfKernelSize, input.cols, input.rows)));

            cv::Mat convolutionResult = cv::Mat::zeros(cv::Size(zeroPadImage.cols, zeroPadImage.rows), input.type());

            for (int i = halfKernelSize; i < zeroPadImage.cols - halfKernelSize; i++)
            {
                for (int j = halfKernelSize; j < zeroPadImage.rows - halfKernelSize; j++)
                {
                    ImageAnalysis::utils::SingleConvolve(zeroPadImage, convolutionResult, kernel, i, j);
                }
            }

            return convolutionResult(cv::Rect(halfKernelSize, halfKernelSize, input.cols, input.rows));
        }

        cv::Mat FilterKernelGenerator(int size)
        {
            if (size % 2 == 0)
            {
                throw "Kurwa daj nieparzyst¹";
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

        std::pair<cv::Mat, cv::Mat> Gradient(cv::Mat input)
        {
            cv::Mat input32F;
            input.convertTo(input32F, CV_32F);

            cv::Mat KernelGx = (cv::Mat_<float>(3, 3) << -1, 0, 1, -2, 0, 2, -1, 0, 1);
            cv::Mat KernelGy = (cv::Mat_<float>(3, 3) << 1, 2, 1, 0, 0, 0, -1, -2, -1);

            cv::Mat Gx = ImageAnalysis::utils::Convolve(input32F, KernelGx);
            cv::Mat Gy = ImageAnalysis::utils::Convolve(input32F, KernelGy);

            cv::Mat GxPow2, GyPow2;
            cv::pow(Gx, 2, GxPow2);
            cv::pow(Gy, 2, GyPow2);

            // Gradient intensity
            cv::Mat G;
            cv::sqrt(GxPow2 + GyPow2, G);

            // Gradient direction
            cv::Mat theta; 
            cv::phase(Gx, Gy, theta);

            // Change range from [0, 2PI] to [0, PI]
            theta /= 2;

            // Rounding direction to multiplications of 45 degree angles (namely 0, 45, 90 etc.)
            // Also change values to range [0, 255]
            for (int j = 0; j < theta.rows; j++)
            {
                for (int i = 0; i < theta.cols; i++)
                {
                    auto angle = theta.at<float>(j, i);
                    if (angle >= CV_PI / 8 && angle < 3 * CV_PI / 8)
                    {
                        theta.at<float>(j, i) = 255 / 4;
                    }
                    else if (angle >= 3 * CV_PI / 8 && angle < 5 * CV_PI / 8)
                    {
                        theta.at<float>(j, i) = 255 / 2;
                    }
                    else if (angle >= 5 * CV_PI / 8 && angle < 7 * CV_PI / 8)
                    {
                        theta.at<float>(j, i) = 3 * 255 / 4;
                    }
                    else if (angle >= 7 * CV_PI / 8 || angle < CV_PI / 8)
                    {
                        theta.at<float>(j, i) = 0;
                    }
                }
            }

            cv::Mat G8u;
            G.convertTo(G8u, CV_8U);
            cv::Mat theta8u;
            theta.convertTo(theta8u, CV_8U);

            return std::make_pair(G8u, theta8u);
        }

        cv::Mat NonMaxSuppression(const cv::Mat& gradientIntensity, const cv::Mat& gradientDirection)
        {
            cv::Mat result = cv::Mat::zeros(gradientIntensity.size(), gradientIntensity.type());

            for (int i = 1; i < gradientIntensity.cols - 1; i++)
            {
                for (int j = 1; j < gradientIntensity.rows - 1; j++)
                {
                    int q = 255;
                    int r = 255;

                    if (gradientDirection.at<uchar>(j, i) == 255)
                    {
                        q = gradientIntensity.at<uchar>(j + 1, i);
                        r = gradientIntensity.at<uchar>(j - 1, i);
                    }
                    else if (gradientDirection.at<uchar>(j, i) == 63)
                    {
                        q = gradientIntensity.at<uchar>(j - 1, i + 1);
                        r = gradientIntensity.at<uchar>(j + 1, i - 1);
                    }
                    else if (gradientDirection.at<uchar>(j, i) == 127)
                    {
                        q = gradientIntensity.at<uchar>(j, i + 1);
                        r = gradientIntensity.at<uchar>(j, i - 1);
                    }
                    else if (gradientDirection.at<uchar>(j, i) == 191)
                    {
                        q = gradientIntensity.at<uchar>(j - 1, i - 1);
                        r = gradientIntensity.at<uchar>(j + 1, i + 1);
                    }

                    if (gradientIntensity.at<uchar>(j, i) >= q && gradientIntensity.at<uchar>(j, i) >= r)
                    {
                        result.at<uchar>(j, i) = gradientIntensity.at<uchar>(j, i);
                    }
                }
            }

            return result;
        }

        cv::Mat DoubleThreshold(const cv::Mat& input, float lowThresholdRatio, float highThresholdRatio)
        {
            double inputMin, inputMax;
            cv::minMaxLoc(input, &inputMin, &inputMax);

            int highThreshold = int(inputMax * highThresholdRatio);
            int lowThreshold = int(highThreshold * lowThresholdRatio);

            cv::Mat result = cv::Mat::zeros(input.size(), input.type());

            for (int i = 0; i < input.cols; i++)
            {
                for (int j = 0; j < input.rows; j++)
                {
                    result.at<uchar>(j, i) = (input.at<uchar>(j, i) >= lowThreshold) ? ((input.at<uchar>(j, i) >= highThreshold) ? 255 : 25) : 0;
                }
            }

            return result;
        }

    } // namespace utils
} // namespace ImageAnalysis
