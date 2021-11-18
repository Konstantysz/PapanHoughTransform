#include "pch.h"
#include "ImageAnalyzerSingleThread.h"

namespace ImageAnalysis
{
	cv::Mat ImageAnalyzerSingleThread::GaussianBlur(const cv::Mat& input, const int& kernelSize, const double& sigma)
	{
		// Generate kernel
		cv::Mat kernel = ImageAnalysis::utils::GaussianKernelGenerator(kernelSize, sigma);
		int halfKernelSize = (kernelSize - 1) / 2;

		// Split image to channels to itarate over pixels image channel independently
		auto channelVector = std::vector<cv::Mat>();
		cv::split(input, channelVector);
		auto convolutionResultVector = channelVector;

		for (int k = 0; k < channelVector.size(); k++)
		{
			convolutionResultVector[k] = ImageAnalysis::utils::Convolve(channelVector[k], kernel);
		}

		cv::Mat output = cv::Mat(cv::Size(input.cols, input.rows), input.type());
		cv::merge(convolutionResultVector, output);

		// Delete zero-pad and return
		return output;
	}

	cv::Mat ImageAnalyzerSingleThread::BGR2Grayscale(const cv::Mat& input)
	{
		if (input.channels() != 3) return input;

		// http://ninghang.blogspot.com/2012/11/list-of-mat-type-in-opencv.html
		int enum3ChannelToSingleDiff = 16;
		cv::Mat grayscaleOutput = cv::Mat::zeros(cv::Size(input.cols, input.rows), input.type() - enum3ChannelToSingleDiff);

		switch (input.type())
		{
		case CV_8UC3:
			for (int j = 0; j < input.rows; j++)
			{
				for (int i = 0; i < input.cols; i++)
				{
					auto bgrPixel = input.at<cv::Vec<uchar, 3>>(j, i);
					grayscaleOutput.at<uchar>(j, i) = (bgrPixel.val[0] + bgrPixel.val[1] + bgrPixel.val[2]) / 3;
				}
			}
			break;
		case CV_8SC3:
			for (int j = 0; j < input.rows; j++)
			{
				for (int i = 0; i < input.cols; i++)
				{
					auto bgrPixel = input.at<cv::Vec<schar, 3>>(j, i);
					grayscaleOutput.at<schar>(j, i) = (bgrPixel.val[0] + bgrPixel.val[1] + bgrPixel.val[2]) / 3;
				}
			}
			break;
		case CV_16UC3:
			for (int j = 0; j < input.rows; j++)
			{
				for (int i = 0; i < input.cols; i++)
				{
					auto bgrPixel = input.at<cv::Vec<ushort, 3>>(j, i);
					grayscaleOutput.at<ushort>(j, i) = (bgrPixel.val[0] + bgrPixel.val[1] + bgrPixel.val[2]) / 3;
				}
			}
			break;
		case CV_16SC3:
			for (int j = 0; j < input.rows; j++)
			{
				for (int i = 0; i < input.cols; i++)
				{
					auto bgrPixel = input.at<cv::Vec<short, 3>>(j, i);
					grayscaleOutput.at<short>(j, i) = (bgrPixel.val[0] + bgrPixel.val[1] + bgrPixel.val[2]) / 3;
				}
			}
			break;
		case CV_32SC3:
			for (int j = 0; j < input.rows; j++)
			{
				for (int i = 0; i < input.cols; i++)
				{
					auto bgrPixel = input.at<cv::Vec<int, 3>>(j, i);
					grayscaleOutput.at<int>(j, i) = (bgrPixel.val[0] + bgrPixel.val[1] + bgrPixel.val[2]) / 3;
				}
			}
			break;
		case CV_32FC3:
			for (int j = 0; j < input.rows; j++)
			{
				for (int i = 0; i < input.cols; i++)
				{
					auto bgrPixel = input.at<cv::Vec<float, 3>>(j, i);
					grayscaleOutput.at<float>(j, i) = (bgrPixel.val[0] + bgrPixel.val[1] + bgrPixel.val[2]) / 3;
				}
			}
			break;
		case CV_64FC3:
			for (int j = 0; j < input.rows; j++)
			{
				for (int i = 0; i < input.cols; i++)
				{
					auto bgrPixel = input.at<cv::Vec<double, 3>>(j, i);
					grayscaleOutput.at<double>(j, i) = (bgrPixel.val[0] + bgrPixel.val[1] + bgrPixel.val[2]) / 3;
				}
			}
			break;
		default:
			throw "Type invlid!";
			break;
		}

		return grayscaleOutput;
	}

	cv::Mat ImageAnalyzerSingleThread::Canny(const cv::Mat& input, float lowThresholdRatio, float highThresholdRatio)
	{
		cv::Mat imgGrayscale;
		if (input.channels() != 1)
		{
			std::cout << "Image had to be converted into grayscale!" << std::endl;
			imgGrayscale = BGR2Grayscale(input);
		}
		else
		{
			imgGrayscale = input;
		}

		// 1. Noise reduction
		auto imgBlurred = GaussianBlur(imgGrayscale, 5);

		// 2. Gradients
		auto gradientsInfo = ImageAnalysis::utils::Gradient(imgBlurred);
		auto gradIntensity = gradientsInfo.first;
		auto gradOrientation = gradientsInfo.second;

		// 3. Non-Maximum Suppression
		auto nonMaxSuppressed = ImageAnalysis::utils::NonMaxSuppression(gradientsInfo.first, gradientsInfo.second);

		// 4. Double threshold
		auto thresholdedEdges = ImageAnalysis::utils::DoubleThreshold(nonMaxSuppressed, lowThresholdRatio, highThresholdRatio);

		// 5. Hysteresis Thresholding
		auto cannyEdges = ImageAnalysis::utils::HysteresisThresholding(thresholdedEdges);

		return cannyEdges;
	}

	std::vector<CircleParameters> ImageAnalyzerSingleThread::CircleHoughTransform(const cv::Mat& input, const int& lowRadiusThreshold, const int& highRadiusThreshold)
	{
		auto circles = std::vector<CircleParameters>();

		cv::Mat accumulator = cv::Mat::zeros(input.size(), CV_32F);

		for (int r = lowRadiusThreshold; r <= highRadiusThreshold; r++)
		{
			for (int t = 0; t < 360; t++)
			{
				for (int i = 0; i < input.cols; i++)
				{
					for (int j = 0; j < input.rows; j++)
					{
						if (input.at<uchar>(j, i) == 255)
						{
							int b = j - r * sin(t * CV_PI / 180);
							int a = i - r * cos(t * CV_PI / 180);
							if (a > 0 && a < input.cols && b > 0 && b < input.rows)
							{
								accumulator.at<float>(b, a) += 1.F;
							}
						}
					}
				}
			}

			//for (int i = 0; i < accumulator.cols; i++)
			//{
			//	for (int j = 0; j < accumulator.rows; j++)
			//	{
			//		if (accumulator.at<float>(j, i) > 2 * CV_PI * r * 0.9)
			//		{
			//			circles.push_back(std::make_pair(r, cv::Point2i(i, j)));
			//		}
			//	}
			//}
		}

		double min, max;
		cv::minMaxLoc(accumulator, &min, &max);
		accumulator /= max;
		accumulator *= 255;

		cv::Mat accumulator8u;
		accumulator.convertTo(accumulator8u, CV_8U);

		return circles;
	}

}
