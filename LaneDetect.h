#ifndef _LaneDetect_H_
#define _LaneDetect_H_
#include "CLine.h"

class LaneDetect {
public:
	Mat preprocessing(Mat frame, Rect roi, Rect roileft, Rect roiright);
	int extractLine(CLine * lines, int num_labels, Mat src_img, Mat img_labels, Mat stats, Mat centroids);
	void displayLineinfo(Mat img, CLine * lines, int num_lines, Scalar linecolor, Scalar captioncolor, int width, int height);
	void detectcolor(Mat& image, double minH, double maxH, double minS, double maxS, Mat& mask);
	double getAngle(Point a, Point b, Point c);
	void currentLane(Mat & image, double * angle, Point & curv, Point & curx, Point & cury, CLine * lines_R, CLine * lines_L, int right_lines, int left_lines, int * check);

};
#endif#pragma once
