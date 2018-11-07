#include <iostream>
#include "opencv2/opencv.hpp"
#include "CurrnetLane.h"
#include "CLine.h"
#include "LaneDetect.h"

#define MIN_AREA_PIXELS 300
#define LINE_SPACING_PIXELS 30

using namespace cv;
using namespace std;

int main()
{
	cout << "***** Lane Detectection Program *** by. JY * JH * CH *****" << endl;
	cout << "File name: ";
	char filename[50];
	cin >> filename;
	VideoCapture capture(filename);
	LaneDetect lanedetect;
	if (!capture.isOpened())
	{
		cout << "Can not open capture !!!" << endl;
		return 0;
	}

	/*원영상의 크기*/
	int width, height;
	width = (int)capture.get(CV_CAP_PROP_FRAME_WIDTH);
	height = (int)capture.get(CV_CAP_PROP_FRAME_HEIGHT);
	Size size = Size((int)width, (int)height);

	cout << "size: " << size << endl;

	/*원영상의 fps(초당 프레임수)*/
	int fps = (int)(capture.get(CV_CAP_PROP_FPS));

	cout << "fps: " << fps << endl;

	int frameNum = -1;
	Mat frame, roiframe, roiframe_L, roiframe_R;
	int roi_x = 0, roi_y = height / 3; //관심영역 시작점의 좌표 x,y
									   //width = 1280;
									   //height = 720/2=340;
									   /*위에서 1/3지점부터 아래 1/2의 영역을 ROI로 설정*/

	/*관심영역 정의*/
	//Parameter: 시작점의(x,y), 관심영역의 크기
	Rect roi(roi_x, roi_y, width, height / 2);
	Rect roileft(roi_x, roi_y, width / 2, height / 2);
	Rect roiright((width - roi_x) / 2, roi_y, width / 2, height / 2);

	CurrentLane *curlane;
	double smallA = 150;
	Point crx;
	Point cry;
	Point crv;
	int check = 0;

	while (char(waitKey(1)) != 'q' && capture.isOpened())
	{
		capture >> frame;
		if (frame.empty())
			break;
		cout << "Detect CurrentLane..." << endl;

		/*관심영역 설정*/
		roiframe = frame(roi);
		roiframe_L = frame(roileft);
		roiframe_R = frame(roiright);

		Mat preImage, edgeImage;
		/*전처리*/
		preImage = lanedetect.preprocessing(frame, roi, roileft, roiright);
		Canny(preImage, edgeImage, 100, 210, 3);

		/*엣지이미지 영역설정*/
		Mat roiEdge_R, roiEdge_L;
		roiEdge_R = edgeImage(roiright);
		roiEdge_L = edgeImage(roileft);

		/*좌우영역을 라벨링*/
		//라벨링에 필요한 변수
		Mat img_labels_R, stats_R, centroids_R;
		Mat img_labels_L, stats_L, centroids_L;

		/*좌우 Labelled 영역의 개수*/
		int numOfLabels_R = connectedComponentsWithStats(roiEdge_R, img_labels_R, stats_R, centroids_R, 8, CV_32S);
		int numOfLabels_L = connectedComponentsWithStats(roiEdge_L, img_labels_L, stats_L, centroids_L, 8, CV_32S);

		// Labelled 영역에서 직선을 추출해 저장하기 위한 메모리 할당 (최대 label 개수만큼)		
		CLine* lines_R = (CLine *)malloc(sizeof(CLine)*numOfLabels_R);
		CLine* lines_L = (CLine *)malloc(sizeof(CLine)*numOfLabels_L);

		/*Labelled 영역에서 직선 추출*/
		int right_lines = lanedetect.extractLine(lines_R, numOfLabels_R, roiEdge_R, img_labels_R, stats_R, centroids_R);
		int left_lines = lanedetect.extractLine(lines_L, numOfLabels_L, roiEdge_L, img_labels_L, stats_L, centroids_L);


		lanedetect.currentLane(roiframe, &smallA, crv, crx, cry, lines_R, lines_L, right_lines, left_lines, &check);

		//동적할당 메모리 해제
		free(lines_R);
		free(lines_L);

		/*현재 차선 표시*/
		if (check >= 0) {
			vector<Point> Lane;
			vector<Point> fillLane;

			Lane.push_back(crv);
			Lane.push_back(crx);
			Lane.push_back(cry);

			Mat temp(roiframe.rows, roiframe.cols, CV_8UC1);
			for (int i = 0; i < temp.cols; i++)
				for (int j = 0; j < temp.rows; j++)
					temp.at<uchar>(Point(i, j)) = 0;

			Mat curlane(roiframe.rows, roiframe.cols, CV_8UC3);
			Mat curarea(roiframe.rows, roiframe.cols, CV_8UC3);
			line(temp, crx, crv, 255, 3);
			line(temp, cry, crv, 255, 3);
			approxPolyDP(Lane, fillLane, 1.0, true);


			fillConvexPoly(temp, &fillLane[0], fillLane.size(), 255, 8, 0);


			fillConvexPoly(curlane, &fillLane[0], fillLane.size(), Scalar(255, 255, 0), 8, 0);

			addWeighted(roiframe, 0.7, curlane, 0.2, 0.0, roiframe);
		}

		/*실행 창*/
		namedWindow("frame", 0);
		imshow("frame", frame);

	}
	return 0;
}