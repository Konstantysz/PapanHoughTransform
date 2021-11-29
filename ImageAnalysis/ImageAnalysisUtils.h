#pragma once
#include <chrono>
#include <windows.h>

namespace ImageAnalysis
{
	namespace utils
	{
		class Timer
		{
		public:
			Timer(std::string name)
			{
				functionName = name;
				tic = std::chrono::steady_clock::now();
			}

			~Timer()
			{
				HANDLE hConsole = GetStdHandle(STD_OUTPUT_HANDLE);
				auto toc = std::chrono::steady_clock::now();
				auto time = std::chrono::duration_cast<std::chrono::milliseconds>(toc - tic).count();

				SetConsoleTextAttribute(hConsole, 11);
				std::cout << "[" + functionName + "] ";
				SetConsoleTextAttribute(hConsole, 15);
				std::cout << "Time elapsed: " << time << "[ms]" << std::endl;
			}
		private:
			std::string functionName;
			std::chrono::steady_clock::time_point tic;
		};

		std::array<int, 256> Histogram(const cv::Mat& input);

		void SingleConvolve(const cv::Mat& input, cv::Mat& output, const cv::Mat& kernel, const int& i, const int& j);
		
		cv::Mat ConvolveZeroPad(const cv::Mat& input, const cv::Mat& kernel);

		cv::Mat Convolve(const cv::Mat& input, const cv::Mat& kernel);

		cv::Mat ConvolveMT(const cv::Mat& input, const cv::Mat& kernel);

		cv::Mat GaussianKernelGenerator(int size, double sigma);

		std::pair<cv::Mat, cv::Mat> Gradient(const cv::Mat& input);

		std::pair<cv::Mat, cv::Mat> GradientMT(const cv::Mat& input);

		cv::Mat NonMaxSuppression(const cv::Mat& gradientIntensity, const cv::Mat& gradientDirection);

		cv::Mat DoubleThreshold(const cv::Mat& input, float lowThresholdRatio = 0.05, float highThresholdRatio = 0.09);

		cv::Mat HysteresisThresholding(const cv::Mat& input);

	} // namespace utils
} // namespace ImageAnalysis
