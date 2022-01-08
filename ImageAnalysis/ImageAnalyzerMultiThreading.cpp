#include "pch.h"
#include "ImageAnalyzerMultiThreading.h"

namespace ImageAnalysis
{
	cv::Mat ImageAnalyzerMultiThreading::GaussianBlur(
		const cv::Mat& input,
		int kernelSize,
		double sigma
	)
	{
		//!< Generate kernel
		cv::Mat kernel = ImageAnalysis::utils::GaussianKernelGenerator(kernelSize, sigma);
		int halfKernelSize = (kernelSize - 1) / 2;

		//!< Split image to channels to itarate over pixels image channel independently,
		//!< if grayscale iamge vector size is 1
		auto channelVector = std::vector<cv::Mat>();
		cv::split(input, channelVector);
		auto convolutionResultVector = channelVector;

		for (int k = 0; k < channelVector.size(); k++)
		{
			convolutionResultVector[k] = ImageAnalysis::utils::ConvolveMT(channelVector[k], kernel);
		}

		cv::Mat output = cv::Mat(cv::Size(input.cols, input.rows), input.type());
		cv::merge(convolutionResultVector, output);

		return output;
	}

	cv::Mat ImageAnalyzerMultiThreading::OtsuThreshold(
		const cv::Mat& input
	)
	{
		int N = input.rows * input.cols;
		auto histogram = utils::Histogram(input);

		double sum = 0;
		for (int v = 0; v < 256; v++)
		{
			sum += v * histogram[v];
		}

		double q1 = 0;
		double q2 = 0;
		double u1 = 0;
		double u2 = 0;
		double sumB = 0;
		double varMax = 0;
		int threshold = 0;
		for (int v = 0; v < 256; v++)
		{
			q1 += histogram[v];
			if (q1 == 0) continue;
			q2 = N - q1;
			sumB += v * histogram[v];
			u1 = sumB / q1;
			u2 = (sum - sumB) / q2;
			double var = q1 * q2 * (u1 - u2) * (u1 - u2);

			if (var > varMax)
			{
				threshold = v;
				varMax = var;
			}
		}

		cv::Mat output;
		input.copyTo(output);
		for (int i = 0; i < input.cols; i++)
		{
			for (int j = 0; j < input.rows; j++)
			{
				output.at<uchar>(j, i) = (output.at<uchar>(j, i) > threshold) ? 255 : 0;
			}
		}

		return output;
	}

	cv::Mat ImageAnalyzerMultiThreading::BGR2Grayscale(const cv::Mat& input)
	{
		if (input.channels() != 3) return input;

		//!< Function supports converting to grayscale all types of cv::Mat listed below:
		//!< http://ninghang.blogspot.com/2012/11/list-of-mat-type-in-opencv.html
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

	cv::Mat ImageAnalyzerMultiThreading::Canny(
		const cv::Mat& input,
		float lowThresholdRatio,
		float highThresholdRatio
	)
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

		//!< 1. Noise reduction
		auto imgBlurred = GaussianBlur(imgGrayscale, 5);

		//!< 2. Gradients
		auto gradientsInfo = ImageAnalysis::utils::GradientMT(imgBlurred);
		auto gradIntensity = gradientsInfo.first;
		auto gradOrientation = gradientsInfo.second;

		//!< 3. Non-Maximum Suppression
		auto nonMaxSuppressed = ImageAnalysis::utils::NonMaxSuppression(gradientsInfo.first, gradientsInfo.second);

		//!< 4. Double threshold
		auto thresholdedEdges = ImageAnalysis::utils::DoubleThreshold(nonMaxSuppressed, lowThresholdRatio, highThresholdRatio);

		//!< 5. Hysteresis Thresholding
		auto cannyEdges = ImageAnalysis::utils::HysteresisThresholding(thresholdedEdges);

		return cannyEdges;
	}

	std::vector<Circle> ImageAnalyzerMultiThreading::CircleHoughTransform(
		const cv::Mat& input,
		int lowRadiusThreshold,
		int highRadiusThreshold,
		int minDistance,
		float circularity
	)
	{
		const int jobsNumber = highRadiusThreshold - lowRadiusThreshold + 1;
		auto jobs = std::vector<std::future<std::vector<Circle>>>(jobsNumber);
		for (int k = 0; k < jobsNumber; k++)
		{
			jobs[k] = std::async(
				std::launch::async,
				[&](int k)
				{
					cv::Mat accumulatorLayer = cv::Mat::zeros(input.size(), CV_32S);
					const int r = lowRadiusThreshold + k;

					//!< Voting
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
										accumulatorLayer.at<int>(b, a)++;
									}
								}
							}
						}
					}

					//!< Extracting circles data
					auto circlesSingleRadius = std::vector<Circle>();
					const int minimalVotes = 2 * CV_PI * r * circularity;
					for (int i = 0; i < accumulatorLayer.cols; i++)
					{
						for (int j = 0; j < accumulatorLayer.rows; j++)
						{
							if (accumulatorLayer.at<int>(j, i) > minimalVotes)
							{
								circlesSingleRadius.push_back(Circle(i, j, r, accumulatorLayer.at<int>(j, i)));
							}
						}
					}

					return circlesSingleRadius;
				},
				k
			);
		}

		auto circlesForEveryRadius = std::vector<std::vector<Circle>>(jobsNumber);
		for (auto& job : jobs)
		{
			circlesForEveryRadius.push_back(job.get());
		}

		auto circles = std::vector<Circle>();
		for (auto& singleRadiusCircles : circlesForEveryRadius)
		{
			for (auto& circle : singleRadiusCircles)
			{
				circles.push_back(circle);
			}
		}

		//!< Circles filtration to remove repeating circles
		std::vector<Circle> resultCircles;
		for (auto circle = circles.begin(); circle != circles.end(); ++circle)
		{
			if (!circle->filtered)
			{
				std::vector<Circle> neighbours;
				neighbours.push_back(*circle);
				circle->filtered = true;
				for (auto otherCircle = circle; otherCircle != circles.end(); ++otherCircle)
				{
					int centerDistance = static_cast<int>(std::sqrt(
						(circle->x - otherCircle->x) * (circle->x - otherCircle->x) + (circle->y - otherCircle->y) * (circle->y - otherCircle->y)
					));
					if (centerDistance < minDistance)
					{
						neighbours.push_back(*otherCircle);
						otherCircle->filtered = true;
					}
				}

				auto bestCircle = std::max_element(neighbours.begin(), neighbours.end(),
					[](const Circle& a, const Circle& b)
					{
						return a.probability < b.probability;
					});
				resultCircles.push_back(*bestCircle);
			}
		}

		return resultCircles;
	}
}
