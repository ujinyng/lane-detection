#pragma once
#ifndef _CurrnetLane_H_
#define _CurrnetLane_H_
#include <iostream>
#include "opencv2/opencv.hpp"
using namespace cv;
using namespace std;

class CurrentLane {
public:
	Point roi_v, roi_x, roi_y;
	double angle;

	CurrentLane(double angle, Point roi_v, Point roi_x, Point roid_y) {
		this->angle = angle;
		this->roi_v = roi_v;
		this->roi_x = roi_x;
		this->roi_y = roi_y;
	}
};
#endif