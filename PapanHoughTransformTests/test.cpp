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
		testImage = std::make_shared<cv::Mat>(cv::imread(".//..//Resources//gayMaria.png"));
		imageAnalyzer = std::make_shared<ImageAnalysis::ImageAnalyzerSingleThread>();
	}

	void TearDown() 
	{
	}

	std::shared_ptr<cv::Mat> testImage;
	std::shared_ptr<ImageAnalysis::ImageAnalyzerSingleThread> imageAnalyzer;
	
	double GetSimilarity(const cv::Mat& A, const cv::Mat& B) {
		if (A.rows > 0 && A.rows == B.rows && A.cols > 0 && A.cols == B.cols) {
			double errorL2 = norm(A, B, cv::NORM_L2);

			return errorL2 / double(A.rows * A.cols);
		}
		else {
			//Images have a different size
			return 256.0;  // Return a bad value
		}
	}

};

TEST_F(TestImageAnalyzerSingleThread, BGR2Gray) {
	cv::Mat cvResult; 
	cv::cvtColor(*testImage, cvResult, cv::COLOR_BGR2GRAY);

	cv::Mat iaResult = imageAnalyzer->BGR2Grayscale(*testImage);
	double similarity = GetSimilarity(cvResult, iaResult);

	EXPECT_TRUE(similarity < 0.01);
}

TEST_F(TestImageAnalyzerSingleThread, GaussianBlur) {
	int n = 3;
	cv::Mat cvResult;
	cv::GaussianBlur(*testImage, cvResult, cv::Size(n, n), 1);

	cv::Mat iaResult = imageAnalyzer->GaussianBlur(*testImage, n, 1);
	double similarity = GetSimilarity(cvResult, iaResult);

	EXPECT_TRUE(similarity < 0.01);
}

int main(int argc, char** argv) {
	::testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}