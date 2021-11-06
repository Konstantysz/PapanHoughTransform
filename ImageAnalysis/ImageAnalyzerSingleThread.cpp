#include "pch.h"
#include "ImageAnalyzerSingleThread.h"

cv::Mat ImageAnalysis::ImageAnalyzerSingleThread::GaussianBlur(const cv::Mat& input, const cv::Mat& kernel)
{
	// Zero-pad matrix
	cv::Mat zeroPadImage = cv::Mat::zeros(cv::Size(input.cols + 2, input.rows + 2), input.type());
	input.copyTo(zeroPadImage(cv::Rect(1, 1, input.cols, input.rows)));

	// Split image to channels to itarate over pixels image channel independently
	auto channelVector = std::vector<cv::Mat>();
	cv::split(zeroPadImage, channelVector);
	auto convolutionResult = channelVector;

	for (int k = 0; k < channelVector.size(); k++) 
	{
		for (int i = 1; i < channelVector[k].cols - 1; i++)
		{
			for (int j = 1; j < channelVector[k].rows - 1; j++)
			{
				ImageAnalysis::utils::SingleConvolve(channelVector[k], convolutionResult[k], kernel, i, j);
			}
		}
	}

	cv::Mat outputZeroPad = cv::Mat(cv::Size(input.cols + 2, input.rows + 2), input.type());
	cv::merge(channelVector, outputZeroPad);

	// Delete zero-pad and return
    return outputZeroPad(cv::Rect(1, 1, input.cols, input.rows));;
}

cv::Mat ImageAnalysis::ImageAnalyzerSingleThread::Image2Grayscale(const cv::Mat& input)
{
	return cv::Mat();
}

cv::Mat ImageAnalysis::ImageAnalyzerSingleThread::Canny(const cv::Mat& input)
{
	return cv::Mat();
}

cv::Mat ImageAnalysis::ImageAnalyzerSingleThread::CircleHoughTransform(const cv::Mat& input)
{
	return cv::Mat();
}
