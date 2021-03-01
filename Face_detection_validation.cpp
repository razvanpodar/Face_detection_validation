#include "stdafx.h"
#include "common.h"
#include "Functions.h"
#include "queue"

// Proiect - detectie robusta de fete

CascadeClassifier face_cascade; // cascade clasifier object for face
CascadeClassifier eyes_cascade; // cascade clasifier object for eyes
CascadeClassifier nose_cascade; // cascade clasifier object for nose

void FaceDetectandDisplay(const string& window_name, Mat frame, int minFaceSize, int minEyeSize)
{
	std::vector<Rect> faces;
	Mat frame_gray;
	Mat regInt;
	Mat faceReg;
	Point theCenter;
	vector<Point> centers;

	cvtColor(frame, frame_gray, CV_BGR2GRAY);
	equalizeHist(frame_gray, frame_gray);

	//-- Detect faces
	face_cascade.detectMultiScale(frame_gray, faces, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE,
		Size(minFaceSize, minFaceSize));
	for (int i = 0; i < faces.size(); i++)
	{
		// get the center of the face
		Point center(faces[i].x + faces[i].width*0.5, faces[i].y + faces[i].height*0.5);
			
		// me
		centers.push_back(center);

		// draw circle around the face
		//ellipse(frame, center, Size(faces[i].width*0.5, faces[i].height*0.5), 0, 0, 360, Scalar(255, 0, 255), 4, 8, 0);

		// face decupation percentage
		float per = 0.18;
		float vPer = 0.20;

		// rectangle(frame, faces[i], Scalar(255, 0, 255));
		Rect decup = Rect(faces[i].x + (per * faces[i].width), faces[i].y + 2.5 * (vPer * faces[i].y), faces[i].width - 2 * (per * faces[i].width), faces[i].height - (vPer * faces[i].height));
		rectangle(frame, decup, Scalar(255, 0, 255));

		regInt = frame_gray(decup);

		equalizeHist(regInt, regInt);

		Point centerDecup = decup.tl() + 0.5 * Point(decup.size());

		std::vector<Rect> nose;
		Rect nose_rect; //the nose is in the 40% ... 75% height of the face
		nose_rect.x = faces[i].x;
		nose_rect.y = faces[i].y + 0.4 * faces[i].height;
		nose_rect.width = faces[i].width;
		nose_rect.height = 0.35 * faces[i].height;
		Mat noseROI = frame_gray(nose_rect);

		Point noseCenter;

		nose_cascade.detectMultiScale(noseROI, nose, 1.1, 1, 0 | CV_HAAR_SCALE_IMAGE, Size(minEyeSize, minEyeSize));
		for (int j = 0; j < nose.size(); j++)
		{
			// relativa la coltul stanga-sus al imaginii:
			Point center(faces[i].x + nose[j].x + nose[j].width*0.5,
				nose_rect.y + nose[j].y + nose[j].height*0.5);
			noseCenter = center;
			int radius = cvRound((nose[j].width + nose[j].height)*0.25);
			// draw circle around the nose
			// circle(frame, center, radius, Scalar(0, 255, 0), 4, 8, 0);
		}

		// Testing all possible vertical symmetry axis
		int histogram[31][512] = { 0 };
		Rect faceRect = Rect(faces[i].x, faces[i].y, faces[i].width, faces[i].height);
		faceReg = frame_gray(faceRect);
		equalizeHist(faceReg, faceReg);

		int sum[31] = { 0 };
		float mean[31] = { 0.0 };
		double variance[31] = { 0.0 };
		double yScore[31] = { 0.0 };
		double maxY = 1000.0;
		double minY = 0.0;
		int posMinY = 0;
		int posMaxY = 1000;

		for (int l = -15; l <= 15; l++)
		{
			Mat rotated;
			Point2f centerFace(faceReg.cols / 2., faceReg.rows / 2.);
			Mat r = getRotationMatrix2D(centerFace, l, 1.0);
			warpAffine(faceReg, rotated, r, faceReg.size());

			Rect faceDRect = Rect((per * faces[i].width), 20, faces[i].width - 2 * (per * faces[i].width), faces[i].height - (vPer * faces[i].height));
			Mat faceRotated = rotated(faceDRect);
			Point centerFaceRotated = faceDRect.tl() + 0.5 * Point(faceDRect.size());

			int d = l + 15;
			
			for (int m = 0; m < faceRotated.rows; m++)
			{
				for (int n = 0; n < centerFaceRotated.x; n++)
				// for (int n = 0; n < noseCenter.x; n++)
				{
					int x = faceRotated.at<uchar>(m, n) - faceRotated.at<uchar>(m, faceRotated.cols - n);
					histogram[d][x + 255]++;
				}
			}

			// Calculate histogram
			for (int m = 0; m < 512; m++)
			{
				sum[d] += histogram[d][m];
			}

			// Calculate mean
			int Sm = 0;
			for (int m = 0; m < 512; m++)
			{
				Sm += m * histogram[d][m];
			}
			mean[d] = Sm / sum[d];
			printf("Mean[%d] = %f\n", d, mean[d]);

			double S = 0.0;
			for (int m = 0; m < 512; m++)
			{
				S += ((histogram[d][m] - mean[d]) * (histogram[d][m] - mean[d])) * histogram[d][m] / sum[d];
			}
			variance[d] = S / sum[d];
			printf("Variance = %f\n", variance[d]);

			yScore[d] = mean[d] / variance[d];
			printf("yscore = %f\n", yScore[d]);
			if (yScore[d] < maxY)
			{
				maxY = yScore[d];
				posMinY = d;
			}

			if (yScore[d] > minY)
			{
				minY = yScore[d];
				posMaxY = d;
			}

			// for analysis - displays all the histograms calculated 
			/*
			string s = "GLDH x = " + std::to_string(d);
			showHistogram(s, histogram[d], 512, 512, true);
			waitKey(0);

			imshow("Rotated", rotated);
			imshow("Rotated face", faceRotated);
			waitKey(0);
			*/
		}

		int km = posMaxY;
		printf("km = %d\n", km);
		string sm = "GLDH x = " + std::to_string(km);
		showHistogram(sm, histogram[km], 512, 512, true);

		int grad = km - 15;

		Mat rotated12 = Mat(frame.rows + 100, frame.cols + 100, CV_8UC1);
		Mat r = getRotationMatrix2D(center, grad, 1.0);
		warpAffine(frame, rotated12, r, frame.size());

		Mat rotated121 = rotated12.clone();

		// draw the symetrical 
		line(rotated12, Point(center.x, 0), Point(center.x, rotated12.rows), Scalar(0, 0, 255), 1, 8);
		line(rotated121, Point(noseCenter.x, 0), Point(noseCenter.x, rotated121.rows), Scalar(255, 0, 0), 1, 8);

		Mat rotated123 = Mat(frame.rows + 100, frame.cols + 100, CV_8UC1);
		Mat rot = getRotationMatrix2D(center, (-1) * grad, 1.0);
		warpAffine(rotated12, rotated123, rot, frame.size());

		Mat rotated1231 = Mat(frame.rows + 100, frame.cols + 100, CV_8UC1);
		Mat rot1 = getRotationMatrix2D(center, (-1) * grad, 1.0);
		warpAffine(rotated121, rotated1231, rot1, frame.size());

		imshow("Finale", rotated123);
		imshow("Finale with line through nose", rotated1231);
	}

	imshow(window_name, frame); //-- Show what you got
}

void faceDetection()
{
	Mat src;
	Mat gray;
	Mat dst;
	char fname[MAX_PATH];
	int M = 0;
	float avg;
	float devStd;
	float y;

	String face_cascade_name = "haarcascade_frontalface_alt.xml";
	String eyes_cascade_name = "haarcascade_eye_tree_eyeglasses.xml";
	String nose_cascade_name = "haarcascade_mcs_nose.xml";
	// Load the cascades
	if (!face_cascade.load(face_cascade_name))
	{
		printf("Error loading face cascades !\n");
		return;
	}
	if (!eyes_cascade.load(eyes_cascade_name))
	{
		printf("Error loading eyes cascades !\n");
		return;
	}
	if (!nose_cascade.load(nose_cascade_name))
	{
		printf("Error loading nose cascades !\n");
		return;
	}

	while (openFileDlg(fname))
	{
		src = imread(fname);
		dst = src.clone();
		cvtColor(src, gray, CV_BGR2GRAY);
		M = src.rows * src.cols;
	
		FaceDetectandDisplay("Face detection", src, 10, 10);

		waitKey(0);
	}
}

int main()
{
	int op;
	do
	{
		system("cls");
		destroyAllWindows();
		printf("Menu:\n");
		printf(" 1 - Face detection\n");
		printf(" 0 - Exit\n\n");
		printf("Option: ");
		scanf("%d",&op);
		switch (op)
		{
			case 1:
				faceDetection();
		}
	}
	while (op!=0);
	return 0;
}