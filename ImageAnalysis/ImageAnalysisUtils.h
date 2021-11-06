#pragma once

namespace ImageAnalysis 
{
	namespace utils
	{
		void SingleConvolve(const cv::Mat& input, cv::Mat& output, const cv::Mat& kernel, const int& i, const int& j);
	} // namespace utils
} // namespace ImageAnalysis
