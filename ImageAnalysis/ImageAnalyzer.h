#pragma once

namespace ImageAnalysis 
{
	typedef std::pair<int, cv::Point2i> CircleParameters;

	class ImageAnalyzer
	{
	public:
		virtual cv::Mat GaussianBlur(const cv::Mat& input, const int& kernelSize) = 0;
		virtual cv::Mat BGR2Grayscale(const cv::Mat& input) = 0;
		virtual cv::Mat Canny(const cv::Mat& input, float lowThresholdRatio = 0.05, float highThresholdRatio = 0.09) = 0;
		virtual std::vector<CircleParameters> CircleHoughTransform(const cv::Mat& input, const int& lowRadiusThreshold, const int& highRadiusThreshold) = 0;
	};
} // namespace ImageAnalysis
