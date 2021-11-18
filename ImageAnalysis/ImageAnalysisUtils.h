#pragma once

namespace ImageAnalysis
{
	namespace utils
	{
		void SingleConvolve(const cv::Mat& input, cv::Mat& output, const cv::Mat& kernel, const int& i, const int& j);
		
		cv::Mat ConvolveZeroPad(const cv::Mat& input, const cv::Mat& kernel);

		cv::Mat Convolve(const cv::Mat& input, const cv::Mat& kernel);

		cv::Mat GaussianKernelGenerator(int size, double sigma);

		std::pair<cv::Mat, cv::Mat> Gradient(const cv::Mat& input);

		cv::Mat NonMaxSuppression(const cv::Mat& gradientIntensity, const cv::Mat& gradientDirection);

		cv::Mat DoubleThreshold(const cv::Mat& input, float lowThresholdRatio = 0.05, float highThresholdRatio = 0.09);

		cv::Mat HysteresisThresholding(const cv::Mat& input);
	} // namespace utils
} // namespace ImageAnalysis
