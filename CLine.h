#pragma once
#ifndef _CLine_H_
#define _CLine_H_

#include <iostream>
#include "opencv2/opencv.hpp"
using namespace cv;
using namespace std;

class CLine {
public:
	Point center;
	double rate;
	double y_inter;
	int left;
	int width;
	Point start;
	Point end;
	CLine(double rate, Point center, int left, int width) {
		this->rate = rate;
		this->left = left;
		this->center = center;
		this->width = width;
		y_inter = -rate * center.x + center.y;
		double x1 = rate;
		double y1 = rate * x1 + y_inter;
		double x2 = left + width;
		double y2 = rate * x2 + y_inter;
		start = Point(x1, y1);
		end = Point(x2, y2);
	}

	double dist_to_point(Point p) {
		double x1 = p.x;
		double y1 = p.y;
		double a = rate;
		double b = -1.0;
		double c = -rate * center.x + center.y;
		double d = fabs(a*x1 + b * y1 + c) / sqrt(a*a + b * b);
		return d;
	}
};
#endif // !_CLine_H_#pragma once
