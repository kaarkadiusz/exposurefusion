#include "opencv2/photo.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include <opencv2/core/utils/logger.hpp>

#include <vector>
#include <iostream>
#include <fstream>
#include <filesystem>

using namespace cv;
using namespace std;
namespace fs = std::filesystem;

double roundoff(double value, unsigned char prec)
{
	double pow_10 = pow(10.0f, (double)prec);
	return round(value * pow_10) / pow_10;
}

void load_images2(String, vector<Mat>&);
void load_images(String, vector<Mat>&);
void contrast(vector<Mat>, vector<Mat>&);
void saturation(vector<Mat>, vector<Mat>&);
void well_exposedness(vector<Mat>, vector<Mat>&);
void gaussian_pyramid(Mat, vector<Mat>&);
void laplacian_pyramid(Mat, vector<Mat>&);
void downsample(Mat, Mat&);
void upsample(Mat, Mat&, int[2]);
void normalise_weights(vector<Mat>, vector<Mat>&);
void reconstruct_laplacian_pyramid(vector<Mat>, Mat&);

int main(int argc, char** argv)
{
	cv::utils::logging::setLogLevel(cv::utils::logging::LogLevel::LOG_LEVEL_SILENT);
	const String keys =
		"{@input | | Input directory that contains images and exposure times. }"
		"{@contrast | 1 | Contrast parameter }"
		"{@saturation | 1 | Saturation parameter }"
		"{@well_exposedness | 1 | Well exposedness parameter }"
		;
	CommandLineParser parser(argc, argv, keys);

	vector<Mat> images255;
	vector<Mat> images1;

	vector<Mat> W;
	double contrast_parm = parser.get<double>("@contrast");
	double saturation_parm = parser.get<double>("@saturation");
	double well_exposedness_parm = parser.get<double>("@well_exposedness");

	cout << "Reading sequence...";
	//load_images(parser.get<String>("@input"), images255);
	load_images(parser.get<String>("@input"), images255);
	int r = images255[0].size().height;
	int c = images255[0].size().width;
	int N = images255.size();

	for (int i = 0; i < N; i++) {
		Mat result = images255[i];
		result.convertTo(result, CV_64FC3);
		result /= 255.0;
		images1.push_back(result);
	}
	cout << " Done" << endl;

	cout << r << "x" << c << " N: " << N << endl;

	for (int i = 0; i < N; i++) {
		Mat result = Mat(r, c, CV_64F, Scalar(1));
		W.push_back(result);
	}
	cout << "Contrast parameter...";
	if (contrast_parm > 0) {
		vector<Mat> C;
		contrast(images255, C);
		for (int i = 0;i < N;i++) {
			for (int x = 0;x < r;x++) {
				for (int y = 0; y < c;y++) {
					W[i].at<double>(x, y) = W[i].at<double>(x, y) * pow(C[i].at<double>(x, y), contrast_parm);
				}
			}
		}
	}
	cout << " Done" << endl;
	cout << "Saturation parameter...";
	if (saturation_parm > 0) {
		vector<Mat> S;
		saturation(images255, S);
		for (int i = 0;i < N;i++) {
			for (int x = 0;x < r;x++) {
				for (int y = 0; y < c;y++) {
					W[i].at<double>(x, y) = W[i].at<double>(x, y) * pow(S[i].at<double>(x, y), saturation_parm);
				}
			}
		}
	}
	cout << " Done" << endl;
	cout << "Well exposedness parameter...";
	if (well_exposedness_parm > 0) {
		vector<Mat> WE;
		well_exposedness(images255, WE);
		for (int i = 0;i < N;i++) {
			for (int x = 0;x < r;x++) {
				for (int y = 0; y < c;y++) {
					W[i].at<double>(x, y) = W[i].at<double>(x, y) * pow(WE[i].at<double>(x, y), well_exposedness_parm);
				}
			}
		}
	}
	cout << " Done" << endl;
	cout << "Weight normalising...";
	Mat sum = Mat::zeros(r, c, CV_64F);
	for (int i = 0;i < N;i++) {
		for (int x = 0;x < r;x++) {
			for (int y = 0; y < c;y++) {
				sum.at<double>(x, y) += W[i].at<double>(x, y);
			}
		}
	}
	for (int i = 0;i < N;i++) {
		for (int x = 0;x < r;x++) {
			for (int y = 0; y < c;y++) {
				sum.at<double>(x, y) += 0.000000000001;
				W[i].at<double>(x, y) = W[i].at<double>(x, y) / sum.at<double>(x, y);
			}
		}
	}
	cout << " Done" << endl;

	cout << "Multiresolution blending...";
	Mat input = Mat::zeros(r, c, CV_64FC3);
	vector<Mat> pyr;
	gaussian_pyramid(input, pyr);
	int nlev = pyr.size();

	for (int i = 0;i < N;i++) {
		vector<Mat> pyrW;
		vector<Mat> pyrI;

		Mat pom = W[i];
		gaussian_pyramid(W[i], pyrW);
		pyrW[0] = pom;
		laplacian_pyramid(images1[i], pyrI);

		for (int j = 0;j < nlev;j++) {
			Mat repmat;
			Mat in[] = { pyrW[j],pyrW[j],pyrW[j] };
			merge(in, 3, repmat);
			Mat result;
			multiply(repmat, pyrI[j], result);
			pyr[j] = pyr[j] + result;
		}
	}
	cout << " Done" << endl;

	cout << "Reconstructing pyrmid...";
	Mat R;
	reconstruct_laplacian_pyramid(pyr, R);
	cout << " Done" << endl;

	cout << "Writing image...";
	imwrite(parser.get<String>("@input") + "/fusion.png", R * 255);
	cout << " Done" << endl;

	return 0;
}

void load_images(String path, vector<Mat>& images)
{
	for (const auto& entry : fs::directory_iterator(path + "sequence\\"))
	{
		if (entry.is_regular_file()) 
		{
			//cout << entry.path().string() << endl;
			Mat img = imread(entry.path().string(), IMREAD_COLOR);
			images.push_back(img);
		}
		//String s = entry.path().string();
		//std::cout << s<<" "<< s.compare(s.length() - 3, 3, "jpg") << " " << std::endl;
	}
}
void load_images2(String path, vector<Mat>& images)
{
	path = path + "/";
	ifstream list_file((path + "list.txt").c_str());
	string name;
	while (list_file >> name) {
		cout << path + name << endl;
		Mat img = imread(path + name, IMREAD_COLOR);
		images.push_back(img);
	}
	list_file.close();
}
void contrast(vector<Mat> images, vector<Mat>& contrast) // zwraca wartości w zakresie 0..1
{
	int h[3][3] = { {0,1,0}, {1,-4,1}, {0,1,0} };
	Mat kernel = Mat(3, 3, CV_32SC1, h);
	int N = images.size();
	int r = images[0].rows;
	int c = images[0].cols;
	for (int i = 0;i < N;i++)
	{
		Mat result = Mat::zeros(r, c, CV_64F);
		for (int y = 0;y < r;y++)
		{
			for (int x = 0;x < c;x++)
			{
				Vec3b& color = images[i].at<Vec3b>(y, x);
				double B = (double)color[0];
				double G = (double)color[1];
				double R = (double)color[2];
				double value = 0.2989 * R + 0.5870 * G + 0.1140 * B;
				value = roundoff(value / 255.0, 4);
				result.at<double>(y, x) = value;
			}
		}
		filter2D(result, result, result.depth(), kernel, Point(-1, -1), (0, 0), BORDER_REPLICATE);
		result = abs(result);
		contrast.push_back(result);
	}
}
void saturation(vector<Mat> images, vector<Mat>& saturation) // zwraca wartości w zakresie 0..1
{
	int N = images.size();
	int r = images[0].rows;
	int c = images[0].cols;
	for (int i = 0;i < N;i++)
	{
		Mat result = Mat::zeros(r, c, CV_64F);
		for (int y = 0;y < r;y++)
		{
			for (int x = 0;x < c;x++)
			{
				Vec3b& color = images[i].at<Vec3b>(y, x);
				double B = (double)color[0] / 255.0;
				double G = (double)color[1] / 255.0;
				double R = (double)color[2] / 255.0;
				double mu = (R + G + B) / 3.0;
				double value = sqrt(((R - mu) * (R - mu) + (G - mu) * (G - mu) + (B - mu) * (B - mu)) / 3.0);
				value = roundoff(value, 4);
				result.at<double>(y, x) = value;
			}
		}
		saturation.push_back(result);
	}
}
void well_exposedness(vector<Mat> images, vector<Mat>& well_exposedness) // zwraca wartości w zakresie 0..1
{
	int N = images.size();
	int r = images[0].rows;
	int c = images[0].cols;
	for (int i = 0;i < N;i++)
	{
		Mat result = Mat::zeros(r, c, CV_64F);
		for (int y = 0;y < r;y++)
		{
			for (int x = 0;x < c;x++)
			{
				Vec3b& color = images[i].at<Vec3b>(y, x);
				double B = (double)color[0] / 255.0;
				double G = (double)color[1] / 255.0;
				double R = (double)color[2] / 255.0;
				R = exp(-0.5 * pow((R - 0.5), 2.0) / pow(0.2, 2.0));
				G = exp(-0.5 * pow((G - 0.5), 2.0) / pow(0.2, 2.0));
				B = exp(-0.5 * pow((B - 0.5), 2.0) / pow(0.2, 2.0));
				double value = R * G * B;
				value = roundoff(value, 4);
				result.at<double>(y, x) = value;
			}
		}
		well_exposedness.push_back(result);
	}
}
void gaussian_pyramid(Mat image, vector<Mat>& pyr)
{
	int r = image.rows;
	int c = image.cols;
	int nlev = floor(log(min(r, c)) / log(2));
	pyr.clear();
	Mat I;
	image.copyTo(I);
	pyr.push_back(I);
	for (int i = 0; i < nlev - 1; i++) {
		downsample(I, I);
		pyr.push_back(I);
	}
}
void laplacian_pyramid(Mat image, vector<Mat>& pyr)
{
	int r = image.rows;
	int c = image.cols;
	int nlev = floor(log(min(r, c)) / log(2));
	Mat J;
	image.copyTo(J);
	Mat I;
	for (int i = 0; i < nlev - 1; i++) {
		downsample(J, I);
		int odd[2] = { 0 , 0 };
		odd[0] = 2 * I.rows - J.rows;
		odd[1] = 2 * I.cols - J.cols;
		Mat us;
		upsample(I, us, odd);
		//cout << "[ " << (double)us.at<Vec3d>(0, 0)[2] << " " << (double)us.at<Vec3d>(0, 0)[1] << " " << (double)us.at<Vec3d>(0, 0)[0] << " ]" << endl;
		us = J - us;
		pyr.push_back(us);
		J = I;
	}
	pyr.push_back(J);
}
void downsample(Mat InputImage, Mat& OutputImage)
{
	//cout << "[ " << (double)InputImage.at<Vec3d>(0, 0)[2] << " " << (double)InputImage.at<Vec3d>(0, 0)[1] << " " << (double)InputImage.at<Vec3d>(0, 0)[0] << " ]" << endl;
	double horizontal[1][5] = { 0.0625, 0.25, 0.375, 0.25, 0.0625 };
	double vertical[5][1] = { {0.0625}, {0.25}, {0.375}, {0.25}, {0.0625} };
	Mat kernel_horizontal = Mat(1, 5, CV_64FC1, horizontal);
	Mat kernel_vertical = Mat(5, 1, CV_64FC1, vertical);
	divide(kernel_horizontal, sum(kernel_horizontal), kernel_horizontal);
	divide(kernel_vertical, sum(kernel_vertical), kernel_vertical);
	Mat R;
	InputImage.copyTo(R);
	filter2D(R, R, R.depth(), kernel_horizontal, Point(-1, -1), 0.0, BORDER_REFLECT);
	filter2D(R, R, R.depth(), kernel_vertical, Point(-1, -1), 0.0, BORDER_REFLECT);
	resize(R, R, Size((R.cols - 1) / 2 + 1, (R.rows - 1) / 2 + 1), 0.0, 0.0, INTER_NEAREST_EXACT);
	//pyrDown(R, R, Size((R.cols - 1) / 2 + 1, (R.rows - 1) / 2 + 1), BORDER_REFLECT);
	//for (int x = 0;x < R.rows;x++) {
	//	for (int y = 0; y < R.cols;y++) {
	//		R.at<Vec3d>(x, y)[0] = (double)R.at<Vec3d>(x, y)[0];
	//		R.at<Vec3d>(x, y)[1] = (double)R.at<Vec3d>(x, y)[1];
	//		R.at<Vec3d>(x, y)[2] = (double)R.at<Vec3d>(x, y)[2];
	//	}
	//}
	OutputImage = R;
}
void upsample(Mat InputImage, Mat& OutputImage, int odd[2])
{
	double horizontal[1][5] = { 0.0625, 0.25, 0.375, 0.25, 0.0625 };
	double vertical[5][1] = { {0.0625}, {0.25}, {0.375}, {0.25}, {0.0625} };
	Mat kernel_horizontal = Mat(1, 5, CV_64FC1, horizontal);
	Mat kernel_vertical = Mat(5, 1, CV_64FC1, vertical);
	divide(kernel_horizontal, sum(kernel_horizontal), kernel_horizontal);
	divide(kernel_vertical, sum(kernel_vertical), kernel_vertical);
	int r = (InputImage.rows + 2) * 2;
	int c = (InputImage.cols + 2) * 2;
	int k = InputImage.channels();
	Mat I;
	copyMakeBorder(InputImage, I, 1, 1, 1, 1, BORDER_REPLICATE);
	Mat result = Mat::zeros(r, c, CV_64FC3);
	for (int x = 0;x < I.rows;x++) {
		for (int y = 0; y < I.cols;y++) {
			Vec3d color = I.at<Vec3d>(x, y);
			result.at<Vec3d>(x * 2, y * 2) = I.at<Vec3d>(x, y) * 4;
		}
	}
	filter2D(result, result, result.depth(), kernel_horizontal, Point(-1, -1), 0.0, BORDER_REFLECT);
	filter2D(result, result, result.depth(), kernel_vertical, Point(-1, -1), 0.0, BORDER_REFLECT);
	result = result(Range(2, r - 2 - odd[0]), Range(2, c - 2 - odd[1]));
	OutputImage = result;
}
void reconstruct_laplacian_pyramid(vector<Mat> pyr, Mat& image)
{
	int r = pyr[0].rows;
	int c = pyr[0].cols;
	int nlev = pyr.size();
	Mat R;
	pyr[nlev - 1].copyTo(R);
	for (int i = nlev - 2;i >= 0;i--) {
		int odd[2] = { 0 , 0 };
		odd[0] = 2 * R.rows - pyr[i].rows;
		odd[1] = 2 * R.cols - pyr[i].cols;
		Mat us;
		upsample(R, us, odd);
		Mat py;
		pyr[i].copyTo(py);
		R = py + us;
	}
	R.copyTo(image);
}