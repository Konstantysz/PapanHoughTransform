#include "pch.h"

#include "gtest/gtest.h"

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "ImageAnalyzerSingleThread.h"

class TestImageAnalyzerSingleThread : public ::testing::Test
{
protected:
	void SetUp() 
	{
		testImage = cv::imread(".//..//Resources//testImage.jpg");
		imageAnalyzer = std::make_shared<ImageAnalysis::ImageAnalyzerSingleThread>();
		cv::cvtColor(testImage, testImageGrayscale, cv::COLOR_BGR2GRAY);
		cv::threshold(testImageGrayscale, testImageBinarized, 27, 255, cv::THRESH_OTSU);
	}

	void TearDown() 
	{
	}

	cv::Mat testImage;
	cv::Mat testImageGrayscale;
	cv::Mat testImageBinarized;
	std::shared_ptr<ImageAnalysis::ImageAnalyzerSingleThread> imageAnalyzer;
	
	double GetSimilarityRMS(const cv::Mat& A, const cv::Mat& B) {
		if (A.rows > 0 && A.rows == B.rows && A.cols > 0 && A.cols == B.cols) {
			double errorL2 = norm(A, B, cv::NORM_L2);

			return std::sqrt(errorL2 / double(A.rows * A.cols));
		}
		else {
			//Images have a different size
			return 256.0;  // Return a bad value
		}
	}

};

TEST_F(TestImageAnalyzerSingleThread, OtsuThreshold)
{
	cv::Mat cvResult;
	{
		[[maybe_unused]] ImageAnalysis::utils::Timer t("Threshold Otsu OpenCV");
		cv::threshold(testImageGrayscale, cvResult, 127, 255, cv::THRESH_OTSU);
	}

	cv::Mat iaResult;
	{
		[[maybe_unused]] ImageAnalysis::utils::Timer t("Threshold Otsu single thread");
		iaResult = imageAnalyzer->OtsuThreshold(testImageGrayscale, 127);
	}
	double error = GetSimilarityRMS(cvResult, iaResult);

	EXPECT_TRUE(error < 0.1);
}

TEST_F(TestImageAnalyzerSingleThread, BGR2Gray) 
{
	cv::Mat cvResult; 
	{
		[[maybe_unused]] ImageAnalysis::utils::Timer t("Color converion OpenCV");
		cv::cvtColor(testImage, cvResult, cv::COLOR_BGR2GRAY);
	}
	
	cv::Mat iaResult;
	{
		[[maybe_unused]] ImageAnalysis::utils::Timer t("Color converion single thread");
		iaResult = imageAnalyzer->BGR2Grayscale(testImage);
	}
	double error = GetSimilarityRMS(cvResult, iaResult);

	EXPECT_TRUE(error < 0.1);
}

TEST_F(TestImageAnalyzerSingleThread, GaussianBlur) 
{	
	std::vector<double> errors;
	for (int n = 3; n <= 9; n += 2)
	{
		cv::Mat cvResult;
		{
			[[maybe_unused]] ImageAnalysis::utils::Timer t("GaussianBlur " + std::to_string(n) + "x" + std::to_string(n) + " OpenCV");
			cv::GaussianBlur(testImage, cvResult, cv::Size(n, n), 1);
		}

		cv::Mat iaResult;
		{
			[[maybe_unused]] ImageAnalysis::utils::Timer t("GaussianBlur " + std::to_string(n) + "x" + std::to_string(n) + " single thread");
			iaResult = imageAnalyzer->GaussianBlur(testImage, n, 1);
		}

		errors.push_back(GetSimilarityRMS(cvResult, iaResult));
	}

	auto maxError = *std::max_element(errors.begin(), errors.end());

	EXPECT_TRUE(maxError < 0.1);
}

TEST_F(TestImageAnalyzerSingleThread, Canny)
{
	double lowThres = 0.3;
	double highThres = 0.6;

	cv::Mat cvResult;
	{ 
		[[maybe_unused]] ImageAnalysis::utils::Timer t("Canny OpenCV");
		cv::Canny(testImageBinarized, cvResult, lowThres, highThres);
	}

	cv::Mat iaResult;
	{
		[[maybe_unused]] ImageAnalysis::utils::Timer t("Canny single thread");
		iaResult = imageAnalyzer->Canny(testImageBinarized, lowThres, highThres);
	}
	auto error = GetSimilarityRMS(cvResult, iaResult);

	EXPECT_TRUE(error < 0.1);
}

int main(int argc, char** argv) {
	::testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}