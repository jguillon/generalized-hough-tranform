/*
 * GeneralHoughTransform.cpp
 *
 *  Created on: 14 févr. 2014
 *      Author: jguillon
 */

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream> // Pour le debug

#include "GeneralHoughTransform.hpp"
#include "util.hpp"

using namespace cv;
using namespace std;

GeneralHoughTransform::GeneralHoughTransform(const Mat& templateImage) {
	m_cannyThreshold1 = 50;
	m_cannyThreshold2 = 150;

	m_deltaRotationAngle = PI/8;
	m_minRotationAngle = -PI/4; //TODO Trouver bug
	m_maxRotationAngle =  PI/4;//PI/2;
	m_nRotations = (m_maxRotationAngle - m_minRotationAngle) / m_deltaRotationAngle + 1;
	m_nSlices = (2.0*PI) / m_deltaRotationAngle;

	m_deltaScaleRatio = 0.01;
	m_minScaleRatio = 0.2;
	m_maxScaleRatio = 0.4;
	m_nScales = (m_maxScaleRatio - m_minScaleRatio) / m_deltaScaleRatio + 1;

	m_accumThreshold = 200; // % des points de l'accumulateur concerné (i.e. la rotation et le scale)
	m_minPositivesDistance = 20.0; // Distance (en pixels) minimale entre deux positifs

	setTemplate(templateImage);
}

void GeneralHoughTransform::setTemplate(const Mat& templateImage) {
	templateImage.copyTo(m_templateImage);
// 	resize( m_templateImage, m_templateImage, Size(51,51));
	findOrigin();
	cvtColor(m_templateImage, m_grayTemplateImage, CV_BGR2GRAY);
	m_grayTemplateImage.convertTo(m_grayTemplateImage, CV_8UC1);
	m_template = Mat(m_grayTemplateImage.size(), CV_8UC1);
	Canny(m_grayTemplateImage, m_template, m_cannyThreshold1, m_cannyThreshold2);
	createRTable();
}

void GeneralHoughTransform::findOrigin() {
	m_origin = Vec2f(m_templateImage.cols/2,m_templateImage.rows/2); // Par défaut, l'origine est placée au centre
	for(int j=0 ; j<m_templateImage.rows ; j++) {
		Vec3b* data= (Vec3b*)(m_templateImage.data + m_templateImage.step.p[0]*j);
		for(int i=0 ; i<m_templateImage.cols ; i++)
			if(data[i]==Vec3b(0,0,255)) { // Si le pixel est rouge...
				m_origin = Vec2f(i,j); // ...alors c'est l'origine du template
				cout << "TROUVE" << endl;
			}
	}
}

void GeneralHoughTransform::createRTable() { // OK
	int iSlice;
	double phi;

	// On calcule les gradients
	Mat direction = gradientDirection(m_template);
	imshow("debug - template", m_template);
	imshow("debug - positive directions", direction);
	cout << direction << endl;

	m_RTable.clear();
	m_RTable.resize(m_nSlices);
	m_RTableSize = 0;
	for(int y=0 ; y<m_template.rows ; y++) {
		uchar *templateRow = m_template.ptr<uchar>(y);
		double *directionRow = direction.ptr<double>(y);
		for(int x=0 ; x<m_template.cols ; x++) {
			if(templateRow[x] == 255) {
				phi = directionRow[x]; // direction du gradient en radians dans [-PI;PI]
				iSlice = rad2SliceIndex(phi,m_nSlices);
				m_RTable[iSlice].push_back(Vec2f(m_origin[0]-x, m_origin[1]-y));
				m_RTableSize++;
			}
		}
	}
}

vector< vector<Vec2f> > GeneralHoughTransform::scaleRTable(const vector< vector<Vec2f> >& RTable, double ratio) {
	vector< vector<Vec2f> > RTableScaled(RTable.size());
	for(vector< vector<Vec2f> >::size_type iSlice=0 ; iSlice<RTable.size() ; iSlice++) {
		for(vector<Vec2f>::size_type ir=0 ; ir<RTable[iSlice].size() ; ir++) {
			RTableScaled[iSlice].push_back(Vec2f(ratio*RTable[iSlice][ir][0], ratio*RTable[iSlice][ir][1]));
		}
	}
	return RTableScaled;
}

vector< vector<Vec2f> > GeneralHoughTransform::rotateRTable(const vector< vector<Vec2f> >& RTable, double angle) {
	vector< vector<Vec2f> > RTableRotated(RTable.size());
	double c = cos(angle);
	double s = sin(angle);
	int iSliceRotated;
	for(vector< vector<Vec2f> >::size_type iSlice = 0 ; iSlice<RTable.size() ; iSlice++) {
		iSliceRotated = rad2SliceIndex(iSlice*m_deltaRotationAngle + angle, m_nSlices);
		for(vector<Vec2f>::size_type ir=0 ; ir<RTable[iSlice].size(); ir++) {
			RTableRotated[iSliceRotated].push_back(Vec2f(c*RTable[iSlice][ir][0] - s*RTable[iSlice][ir][1], s*RTable[iSlice][ir][0] + c*RTable[iSlice][ir][1]));
		}
	}
	return RTableRotated;
}

void GeneralHoughTransform::showRTable(vector< vector<Vec2f> > RTable) {
	int N(0);
	cout << "--------" << endl;
	for(vector< vector<Vec2f> >::size_type i=0 ; i<RTable.size() ; i++) {
		for(vector<Vec2f>::size_type j=0 ; j<RTable[i].size() ; j++) {
			cout << RTable[i][j];
			N++;
		}
		cout << endl;
	}
	cout << N << " éléments" << endl;
}

void GeneralHoughTransform::accumulate(const Mat& image) {
	//TODO Fonction edges

	// On prétraite l'image
	Mat grayImage(image.size(), CV_8UC1), edges(image.size(), CV_8UC1);
	cvtColor(image, edges, CV_BGR2GRAY);
//	blur(edges, edges, Size(3,3));
	Canny(edges, edges, m_cannyThreshold1, m_cannyThreshold2);
	Mat direction = gradientDirection(edges);

	imshow("debug - src edges", edges);
	imshow("debug - src edges gradient direction", direction);
	waitKey(0);

	int X = image.cols;
	int Y = image.rows;
	int S = ceil((m_maxScaleRatio - m_minScaleRatio) / m_deltaScaleRatio) + 1; // Scale Slices Number
	int R = ceil((m_maxRotationAngle - m_minRotationAngle) / m_deltaRotationAngle) + 1; // Rotation Slices Number
	int iSlice;
	double phi;

	vector< vector< Mat > > accum(R,vector<Mat>(S, Mat::zeros(Size(X,Y),CV_64F)));
	Mat totalAccum = Mat::zeros(Size(X,Y),CV_32S);
	int iScaleSlice(0), iRotationSlice(0), ix(0), iy(0);
	double max(0.0);
	vector< vector<Vec2f> > RTableRotated(m_RTable.size()), RTableScaled(m_RTable.size());
	Mat showAccum(Size(X,Y),CV_8UC1);
	vector<GHTPoint> points;
	GHTPoint point;

	for(double angle=m_minRotationAngle ; angle<=m_maxRotationAngle+0.0001 ; angle+=m_deltaRotationAngle) { // Pour chaque rotation (0.0001 pour le pb de comparaison de double)
		iRotationSlice = round((angle-m_minRotationAngle)/m_deltaRotationAngle);
		cout << "Rotation Angle\t: " << angle/PI*180 << "�\t:\t" << iRotationSlice << endl;
		RTableRotated = rotateRTable(m_RTable,angle);
		for(double ratio=m_minScaleRatio ; ratio<=m_maxScaleRatio+0.0001 ; ratio+=m_deltaScaleRatio) { // Pour chaque scale (0.0001 pour le pb de comparaison de double)
 			iScaleSlice = round((ratio-m_minScaleRatio)/m_deltaScaleRatio);
			cout << "Scale Ratio\t: " << ratio*100 << "%\t:\t" << iScaleSlice << endl;
			RTableScaled = scaleRTable(RTableRotated,ratio);
			accum[iRotationSlice][iScaleSlice] = Mat::zeros(Size(X,Y),CV_64F);
//			max = 0;
			for(int y=0 ; y<image.rows ; y++) {
				for(int x=0 ; x<image.cols ; x++) {
					phi = direction.at<double>(y,x);
					if(phi != 0.0) {
						iSlice = rad2SliceIndex(phi,m_nSlices);
						for(vector<Vec2f>::size_type ir=0 ; ir<RTableScaled[iSlice].size() ; ir++) { // Pour tous les r associé à cette slice (angle) dans la angle-table
							ix = x + round(RTableScaled[iSlice][ir][0]);	// On calcule x+r, la position supposée de l'origine du template
							iy = y + round(RTableScaled[iSlice][ir][1]);
							if(ix>=0 && ix<image.cols && iy>=0 && iy<image.rows) { // Si celle-ci tombe dans le cadre de l'image
								totalAccum.at<int>(iy,ix)++;
								if(++accum[iRotationSlice][iScaleSlice].at<double>(iy,ix) > max) { // Incrémenter le Vec2f x+r correspondant dans l'accumulateur
									max = accum[iRotationSlice][iScaleSlice].at<double>(iy,ix); // TODO Remplacer par point.hit
									point.phi = angle;
									point.s = ratio;
									point.y.y = iy;
									point.y.x = ix;
									point.hits = accum[iRotationSlice][iScaleSlice].at<double>(iy,ix);
								}
//								normalize(accum[iRotationSlice][iScaleSlice], showAccum, 0, 255, NORM_MINMAX, CV_8UC1);
//								imshow("debug - subaccum", showAccum);	waitKey(1);
								//TODO Afficher le maximum de la matrice PAR TRANSFORMATION
							}
						}
					}
				}
			}
//			points.push_back(point);
//			normalize(accum[iRotationSlice][iScaleSlice], showAccum, 0, 255, NORM_MINMAX, CV_8UC1);
			normalize(totalAccum, showAccum, 0, 255, NORM_MINMAX, CV_8UC1);
			imshow("debug - accum", showAccum);	waitKey(1);
			blur(accum[iRotationSlice][iScaleSlice], accum[iRotationSlice][iScaleSlice], Size(3,3)); // Pour augmenter faire baisser les maxima locaux et augmenter les maxima de zones
		}
	}

	points.push_back(point);

	cout << points.size() << endl;
	// Trouver les meilleurs points
//	vector<GHTPoint> points = findTemplates(accum, m_accumThreshold);

	// Dessiner les meilleurs points
	for(vector<GHTPoint>::size_type i=0 ; i<points.size() ; i++) {
		Mat out(image.size(),image.type());
		image.copyTo(out);
		drawTemplate(out, points[i]);
		imshow("debug - output", out);
		waitKey(0);
	}

	// Affichage de l'image avec les solutions trouvées
}

vector<GHTPoint> GeneralHoughTransform::findTemplates(vector< vector< Mat > >& accum, int threshold) {
//
//	cout << points.size() << endl;
//
//	//=============CE QUI SUIT FONCTIONNE===============// TODO L'Inclure dans la boucle ci-dessus
//
//	double distanceTreshold = m_minPositivesDistance;
//	bool best;
	vector<GHTPoint> newPoints;
//
//	for(vector<GHTPoint>::size_type i=0 ; i<points.size() ; i++) { // On compare chque point...
//		best = true;
//		for(vector<GHTPoint>::size_type ii=0 ; ii<points.size() ; ii++) { // ...avec tous ses compères...
//			int xVec = points[ii].y.x - points[i].y.x;
//			int yVec = points[ii].y.y - points[i].y.y;
//			double norm = fastsqrt(xVec*xVec + yVec*yVec);
//			if(i != ii && norm < distanceTreshold && points[ii].hits > points[i].hits)  // ...si ils sont proches et que son compère est meilleur que lui...
//				best = false;	// ...alors ce n'est pas le meilleur !
//			else if(i != ii && norm < distanceTreshold && points[ii].hits == points[i].hits)
//				points[ii].hits--;
//		}
//		if(best) {
//			cout << points[i].y << " avec un rapport de grandeur de " << points[i].s << " et une rotation de " << points[i].phi << "rad." << endl;
//			newPoints.push_back(points[i]); // Mais si c'est le meilleur, on l'ajoute au tableau des bons points
//		}
//	}
//
//	cout << newPoints.size() << endl;
	return newPoints;
}

void GeneralHoughTransform::drawTemplate(Mat& image, GHTPoint params) {
	cout << params.y << " avec un rapport de grandeur de " << params.s << " et une rotation de " << params.phi/PI*180 << "� et avec " << params.hits << " !" << endl;
	double c = cos(params.phi);
	double s = sin(params.phi);
	int x(0), y(0), relx(0), rely(0);
	for(vector< vector<Vec2f> >::size_type iSlice = 0 ; iSlice<m_RTable.size() ; iSlice++)
		for(vector<Vec2f>::size_type ir=0 ; ir<m_RTable[iSlice].size() ; ir++) {
			relx = params.s * (c*m_RTable[iSlice][ir][0] - s*m_RTable[iSlice][ir][1]); // Coordonnée x du point du template après rotation et mise à l'échelle (relative à l'origine du template)
			rely = params.s * (s*m_RTable[iSlice][ir][0] + c*m_RTable[iSlice][ir][1]); // Coordonnée x du point du template après rotation et mise à l'échelle (relative à l'origine du template)
			x = params.y.x + relx; // Coordonnée x du point du template dans l'image
			y = params.y.y + rely; // Coordonnée x du point du template dans l'image
			if(x>=0 && x<image.cols && y>=0 && y<image.rows)
				image.at<Vec3b>(y,x) = Vec3b(0,255,0);
		}
}
