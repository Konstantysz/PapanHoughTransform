#pragma once
#include "ImageAnalyzer.h"
#include "ImageAnalysisUtils.h"

namespace ImageAnalysis
{
	class ImageAnalyzerSingleThread : public ImageAnalyzer
	{
	public:
		cv::Mat GaussianBlur(const cv::Mat& input, const int& kernelSize, const double& sigma = 1.0);
		cv::Mat BGR2Grayscale(const cv::Mat& input);
		cv::Mat Canny(const cv::Mat& input, float lowThresholdRatio = 0.05, float highThresholdRatio = 0.09);
		std::vector<CircleParameters> CircleHoughTransform(const cv::Mat& input, const int& lowRadiusThreshold, const int& highRadiusThreshold);
	};
} // namespace ImageAnalysis
