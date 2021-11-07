#pragma once
#include "ImageAnalyzer.h"
#include "ImageAnalysisUtils.h"

namespace ImageAnalysis
{
	class ImageAnalyzerSingleThread : public ImageAnalyzer
	{
	public:
		cv::Mat GaussianBlur(const cv::Mat& input, const int& kernelSize);
		cv::Mat Image2Grayscale(const cv::Mat& input);
		cv::Mat Canny(const cv::Mat& input);
		cv::Mat CircleHoughTransform(const cv::Mat& input);
	};
} // namespace ImageAnalysis
