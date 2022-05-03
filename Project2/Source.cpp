#include "opencv2/photo.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include <opencv2/core/utils/logger.hpp>

#include <vector>
#include <iostream>
#include <fstream>

using namespace cv;
using namespace std;

void loadExposureSeq(String, vector<Mat>&, vector<float>&);
void load_images(String, vector<Mat>&);
void contrast(vector<Mat>, vector<Mat>&);
void saturation(vector<Mat>, vector<Mat>&);
void well_exposedness(vector<Mat>, vector<Mat>&);
void gaussian_pyramid(vector<Mat>, vector<vector<Mat>>&);

int main(int argc, char** argv)
{
	cv::utils::logging::setLogLevel(cv::utils::logging::LogLevel::LOG_LEVEL_SILENT);
	CommandLineParser parser(argc, argv, "{@input | | Input directory that contains images and exposure times. }");

	//! [Load images and exposure times]

	vector<Mat> images255;
	vector<Mat> images1;

	vector<Mat> S;
	vector<Mat> WE;
	vector<Mat> W;
	vector<vector<Mat>> pyr;
	double contrast_parm = 1.0;
	double saturation_parm = 1.0;
	double well_exposedness_parm = 1.0;
	//vector<float> times;
	//loadExposureSeq(parser.get<String>("@input"), images, times
	load_images(parser.get<String>("@input"), images255);
	int r = images255[0].size().height;
	int c = images255[0].size().width;
	int N = images255.size();
	cout << r << endl << c << endl << N << endl;
	for (int i = 0; i < N; i++) {
		Mat result = Mat(r, c, CV_64FC1, Scalar(1, 1, 1));
		W.push_back(result);
	}
	//W[0].at<double>(0, 0) += 3.5425;
	//cout << W[0].at<double>(0, 0);
	//for (int x = 0; x < r;x++) {
	//	for (int y = 0;y < c;y++) {
	//		uchar& ptr = W[0].at<uchar>(x, y);
	//		cout << (double)ptr << " ";
	//	}
	//	cout << endl;
	//}
	if (contrast_parm > 0) {
		vector<Mat> C;
		contrast(images255, C);
		//cout << W[0].type() << endl << C[0].type();
		cout << (double)(C[0].at<uchar>(10, 10) / 255.0) << " ";
		for (int i = 0;i < N;i++) {
			for (int y = 0;y < c;y++) {
				for (int x = 0; x < r;x++) {
					//W[i].at<double>(x, y) = W[i].at<double>(x, y) * pow(C[i].at<double>(x, y), contrast_parm);
					//cout << (double)(C[i].at<uchar>(x, y)/255.0)<<" ";
				}
			}
		}
	}
	//for (int x = 0; x < r;x++) {
	//	cout << W[0].at<double>(x, 0) << endl;
	//}
	//saturation(images255, S);
	//well_exposedness(images255, WE);
	//cout << "C(): " << C.size() << endl;
	//cout << "C().height: " << C[0].size().height << endl; //500
	//cout << "C().width: " << C[0].size().width << endl; //752
	//cout << "S(): " << C.size() << endl;
	//cout << "S().height: " << C[0].size().height << endl; //500
	//cout << "S().width: " << C[0].size().width << endl; //752
	//cout << "WE(): " << C.size() << endl;
	//cout << "WE().height: " << C[0].size().height << endl; //500
	//cout << "WE().width: " << C[0].size().width << endl; //752
	//for (int i = 0;i < N;i++) {
	//	for (int x = 0; x < r;x++) {
	//		for (int y = 0;y < c;y++) {
	//			cout << W[i].at<double>(x, y);
	//		}
	//	}
	//}
	//for (int i = 0;i < images255.size();i++) {
	//	Mat result;
	//	images255[i].convertTo(result, CV_64FC3, 1.0 / 255.0);
	//	images1.push_back(result);
	//}
	//cout << "size(): " << images255[0].size << endl;
	//cout << "size(): " << images255.size() << endl;
	//cout << "size().height: " << images255[0].size().height << endl; //500
	//cout << "size().width: " << images255[0].size().width << endl; //752
	//gaussian_pyramid(images255, pyr);
	//cout << "size(): " << pyr.size() << endl;
	//cout << "[0].size(): " << pyr[0].size() << endl;
	//cout << "[0][0].size: " << pyr[0][0].size << endl;

	//cout << "type(): " << images[0].type() << endl;
	//cout << "channels(): " << images[0].channels() << endl;


	// 
	//cv::Vec3b* ptr = images255[2].ptr<cv::Vec3b>(200);
	//cout << (double)ptr[200][0];
	//cout << -0.5 * (0.05 - 0.5) << endl;
	//cout << -0.5 * pow((0.05 - 0.5), 2) << endl;
	//cout << exp(-0.5 * pow((0.05 - 0.5), 2) / pow(0.2, 2)) << endl;
	//for (int i = 0;i < W.size();i++) {
	//	imwrite(parser.get<String>("@input") + "/works/W" + to_string(i+1) + ".png", W[i]);
	//}
	/*for (int i = 0;i < C.size();i++) {
		imwrite(parser.get<String>("@input") + "/C" + to_string(i) + ".png", C[i]);
	}*/
	//cout << "Greyscale2: " << greyscale2Pixel << endl;
	//imwrite(parser.get<String>("@input") + "/wynik.png", wynik);
	//cout << "Vec3b: " << (int)bgrPixel[0] <<" "<< (int)bgrPixel[1] << " " << (int)bgrPixel[2] << endl;
	//cout << "Vec3b: " << (double)bgrPixel[0] / 255 << " " << (double)bgrPixel[1] / 255 << " " << (double)bgrPixel[2] / 255 << endl;
	//! [Load images and exposure times]

	//! [Perform exposure fusion]
	//Mat fusion;
	//Ptr<MergeMertens> merge_mertens = createMergeMertens(1.0F, 1.0F, 1.0F);
	//merge_mertens->process(images, fusion);
	//! [Perform exposure fusion]

	//! [Write results]
	//imwrite(parser.get<String>("@input") + "/fusion.png", fusion * 255);
	//! [Write results]

	return 0;
}

void loadExposureSeq(String path, vector<Mat>& images, vector<float>& times)
{
	path = path + "/";
	ifstream list_file((path + "list.txt").c_str());
	string name;
	float val;
	while (list_file >> name >> val) {
		Mat img = imread(path + name);
		images.push_back(img);
		times.push_back(1 / val);
	}
	list_file.close();
}
void load_images(String path, vector<Mat>& images)
{
	path = path + "/";
	ifstream list_file((path + "list.txt").c_str());
	string name;
	while (list_file >> name) {
		Mat img = imread(path + name, IMREAD_COLOR);
		images.push_back(img);
	}
	list_file.close();
}
void contrast(vector<Mat> images, vector<Mat>& contrast)
{
	int h[3][3] = { {0,1,0}, {1,-4,1}, {0,1,0} };
	Mat kernel = Mat(3, 3, CV_32SC1, h);
	int N = images.size();
	for (int i = 0;i < N;i++) {
		Mat greyscale;
		cvtColor(images[i], greyscale, COLOR_BGR2GRAY);
		Mat result;
		filter2D(greyscale, result, greyscale.depth(), kernel, Point(-1, -1), (0, 0), BORDER_REPLICATE);
		result = abs(result);
		contrast.push_back(result);
	}
}
void saturation(vector<Mat> images, vector<Mat>& saturation)
{
	int N = images.size();
	for (int i = 0;i < N;i++) {
		Mat grayscale;
		cvtColor(images[i], grayscale, COLOR_BGR2GRAY);
		for (int r = 0; r < images[i].rows; r++) {
			cv::Vec3b* ptr = images[i].ptr<cv::Vec3b>(r);
			uchar* ptr2 = grayscale.ptr<uchar>(r);
			for (int c = 0; c < images[i].cols; c++) {
				double R = (double)((double)ptr[c][0] / 255.0);
				double G = (double)((double)ptr[c][1] / 255.0);
				double B = (double)((double)ptr[c][2] / 255.0);
				//cout << R << " " << G << " " << B << " | ";
				double mu = (R + G + B) / 3.0;
				double result = sqrt(((R - mu) * (R - mu) + (G - mu) * (G - mu) + (B - mu) * (B - mu)) / 3.0);
				result = result * 255.0;
				//cout << result << " | ";
				ptr2[c] = (int)result;
			}
			//cout << " --- " << endl;
		}
		saturation.push_back(grayscale);
	}
}
void well_exposedness(vector<Mat> images, vector<Mat>& well_exposedness)
{
	int N = images.size();
	for (int i = 0;i < N;i++) {
		Mat grayscale;
		cvtColor(images[i], grayscale, COLOR_BGR2GRAY);
		for (int r = 0; r < images[i].rows; r++) {
			cv::Vec3b* ptr = images[i].ptr<cv::Vec3b>(r);
			uchar* ptr2 = grayscale.ptr<uchar>(r);
			for (int c = 0; c < images[i].cols; c++) {
				double R = (double)((double)ptr[c][0] / 255.0);
				double G = (double)((double)ptr[c][1] / 255.0);
				double B = (double)((double)ptr[c][2] / 255.0);
				R = exp(-0.5 * pow((R - 0.5), 2.0) / pow(0.2, 2.0));
				G = exp(-0.5 * pow((G - 0.5), 2.0) / pow(0.2, 2.0));
				B = exp(-0.5 * pow((B - 0.5), 2.0) / pow(0.2, 2.0));
				/*if(R != G && R != B && G != B)
				cout << R << " " << G << " " << B << " | ";*/
				double result = R * G * B;
				result = result * 255.0;
				//cout << result << " | ";
				ptr2[c] = (int)result;
			}
			//cout << " --- " << endl;
		}
		well_exposedness.push_back(grayscale);
	}
}
void gaussian_pyramid(vector<Mat> images, vector<vector<Mat>>& pyr)
{
	int r = images[0].size().height;
	int c = images[0].size().width;
	int nlev = floor(log(min(r, c)) / log(2));
	pyr.push_back(images);
	for (int i = 1; i < nlev; i++) {
		vector<Mat> resultVector;
		for (int j = 0; j < images.size(); j++) {
			pyrDown(images[j], images[j], Size((int)floor((double)images[j].cols / 2 + 0.5), (int)floor((double)images[j].rows / 2 + 0.5)), BORDER_REPLICATE);
			resultVector.push_back(images[j]);
		}
		pyr.push_back(resultVector);
	}
}