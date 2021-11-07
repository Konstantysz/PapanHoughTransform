#pragma once

namespace ImageAnalysis
{
	namespace utils
	{
		void SingleConvolve(const cv::Mat& input, cv::Mat& output, const cv::Mat& kernel, const int& i, const int& j);

		cv::Mat FilterKernelGenerator(int size);
	} // namespace utils
} // namespace ImageAnalysis
