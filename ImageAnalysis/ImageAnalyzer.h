#pragma once

namespace ImageAnalysis {
	class ImageAnalyzer
	{
	public:
		virtual cv::Mat GaussianBlur(const cv::Mat& input, const int& kernelSize) = 0;
		virtual cv::Mat Image2Grayscale(const cv::Mat& input) = 0;
		virtual cv::Mat Canny(const cv::Mat& input) = 0;
		virtual cv::Mat CircleHoughTransform(const cv::Mat& input) = 0;
	};
} // namespace ImageAnalysis
