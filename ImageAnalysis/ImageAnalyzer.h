#pragma once

namespace ImageAnalysis 
{
	struct Circle
	{
		Circle(int x, int y, int r, int votes) : x(x), y(y), radius(r)
		{
			float diameter = 2 * CV_PI * r;
			probability = (r > 0) ? static_cast<float>(votes) / diameter : 0.F;
			filtered = false;
		}
		float probability;
		int x;
		int y;
		int radius;
		bool filtered;
	};

	class ImageAnalyzer
	{
	public:
		virtual cv::Mat GaussianBlur(
			const cv::Mat& input, 
			const int& kernelSize, 
			const double& sigma = 1.0
		) = 0;

		virtual cv::Mat OtsuThreshold(
			const cv::Mat& input, 
			const int& thresholdValue
		) = 0;
		virtual cv::Mat BGR2Grayscale(const cv::Mat& input) = 0;

		virtual cv::Mat Canny(
			const cv::Mat& input, 
			const float& lowThresholdRatio = 0.05, 
			const float& highThresholdRatio = 0.09
		) = 0;

		virtual std::vector<Circle> CircleHoughTransform(
			const cv::Mat& input, 
			const int& lowRadiusThreshold, 
			const int& highRadiusThreshold,
			const int& minDistance,
			const float& circulairty
		) = 0;
	};
} // namespace ImageAnalysis
