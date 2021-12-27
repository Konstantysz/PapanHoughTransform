#pragma once
#include "ImageAnalyzer.h"
#include "ImageAnalysisUtils.h"

namespace ImageAnalysis
{
	class ImageAnalyzerMultiThreading : public ImageAnalyzer
	{
	public:
		cv::Mat GaussianBlur(
			const cv::Mat& input,
			int kernelSize,
			double sigma = 1.0
		);

		cv::Mat OtsuThreshold(
			const cv::Mat& input,
			int thresholdValue
		);

		cv::Mat BGR2Grayscale(const cv::Mat& input);

		cv::Mat Canny(
			const cv::Mat& input,
			float lowThresholdRatio = 0.05,
			float highThresholdRatio = 0.09
		);

		std::vector<Circle> CircleHoughTransform(
			const cv::Mat& input,
			int lowRadiusThreshold,
			int highRadiusThreshold,
			int minDistance,
			float circulatity
		);
	};
} // namespace ImageAnalysis

