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
	Mat tpl = imread("res/mire_2.0.png");
	Mat src = imread("res/moi2_1.jpg");
	GeneralHoughTransform ght(tpl);

 	Size s( src.size().width / 1, src.size().height / 1);
 	resize( src, src, s, 0, 0, CV_INTER_AREA );

 	imshow("debug - image", src);

 	ght.accumulate(src);

//	Mat out(src.size(),src.type());
//	src.copyTo(out);
//	ght.drawTemplate(out, max); //TODO
//	imshow("debug - output", out);
//	waitKey(0);

	return 0;
}


