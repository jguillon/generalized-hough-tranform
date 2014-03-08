/*
 * main.cpp
 *
 *  Created on: 15 f�vr. 2014
 *      Author: J�r�my
 */

#include <iostream>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "GeneralHoughTransform.hpp"

using namespace std;
using namespace cv;

int main(int argc, char** argv) {
	Mat tpl = imread("res/tpl.jpg");
	Mat src = imread("res/image.jpg");
	GeneralHoughTransform ght(tpl);

 	Size s( src.size().width / 4, src.size().height / 4);
 	resize( src, src, s, 0, 0, CV_INTER_AREA );

 	imshow("debug - image", src);

 	ght.accumulate(src);

	return 0;
}


