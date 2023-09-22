#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "opencv2/core/core_c.h"

using namespace std;
using namespace cv;

void waitEscPress()
{
    int key = 0;
    while (key != 27) { // Ждем нажатия клавиши Esc (код 27)
        key = waitKey(0);
    }
}

int main()
{
    // любое изображение с диска;

    string filename = "D:\\University\\PMMP\\labs\\lab1\\bunnies.jpg";
    string winname = "mywinname";

    Mat imr = imread(filename);
    if (imr.empty()) {
        cerr << "Не удалось загрузить изображение!" << endl;
    }

    namedWindow(winname);
    imshow(winname, imr);

    waitEscPress();

    // изображение в оттенках серого (одноканальное) (функция cvtColor());

    Mat grayImage;
    cvtColor(imr, grayImage, COLOR_BGR2GRAY);
    namedWindow(winname);
    imshow(winname, grayImage);
    imwrite("grayBunnies.jpg", grayImage);

    waitEscPress();

    // бинарное изображение (черно-белое), полученное методом отсечения порога (функция threshold());

    Mat blackWhiteImage;
    double maxValue = 255;
    int blockSize = 11;
    double C = 2;
    threshold(grayImage, blackWhiteImage, 128, maxValue, THRESH_BINARY);
    namedWindow(winname);
    imshow(winname, blackWhiteImage);
    imwrite("blackwhitebunnies.jpg", blackWhiteImage);

    waitEscPress();

    // бинарное изображение, полученное методом адаптивной бинаризации (функция adaptiveThreshold ()).

    Mat adaptiveBlackWhiteImage;
    adaptiveThreshold(grayImage, adaptiveBlackWhiteImage, maxValue, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, blockSize, C);
    namedWindow(winname);
    imshow(winname, adaptiveBlackWhiteImage);
    imwrite("adaptiveblackwhitebunnies.jpg", adaptiveBlackWhiteImage);

    waitEscPress();



    filename = "peresvet.jpg";
    imr = imread(filename, IMREAD_GRAYSCALE);

    if (imr.empty()) {
        cerr << "Не удалось загрузить изображение!" << endl;
        return -1;
    }

    // Построение гистограммы до выравнивания
    int histSize = 256; // Размер гистограммы
    float range[] = { 0, 256 }; // Диапазон интенсивности пикселей
    const float* histRange = { range };
    Mat histBefore;
    calcHist(&imr, 1, 0, Mat(), histBefore, 1, &histSize, &histRange, true, false);

    // Выравнивание освещенности
    Mat equalizedImage;
    equalizeHist(imr, equalizedImage);

    // Построение гистограммы после выравнивания
    Mat histAfter;
    calcHist(&equalizedImage, 1, 0, Mat(), histAfter, 1, &histSize, &histRange, true, false);

    // Отображение изображений и гистограмм
    namedWindow("Original Image");
    imshow("Original Image", imr);

    namedWindow("Equalized Image");
    imshow("Equalized Image", equalizedImage);

    namedWindow("Histogram Before Equalization");
    namedWindow("Histogram After Equalization");

    int histWidth = 512;
    int histHeight = 400;
    int binWidth = cvRound((double)histWidth / histSize);

    Mat histImageBefore(histHeight, histWidth, CV_8UC3, Scalar(0, 0, 0));
    Mat histImageAfter(histHeight, histWidth, CV_8UC3, Scalar(0, 0, 0));

    normalize(histBefore, histBefore, 0, histImageBefore.rows, NORM_MINMAX, -1, Mat());
    normalize(histAfter, histAfter, 0, histImageAfter.rows, NORM_MINMAX, -1, Mat());

    for (int i = 1; i < histSize; i++) {
        line(histImageBefore, Point(binWidth * (i - 1), histHeight - cvRound(histBefore.at<float>(i - 1))),
            Point(binWidth * i, histHeight - cvRound(histBefore.at<float>(i))),
            Scalar(255, 0, 0), 2, 8, 0);

        line(histImageAfter, Point(binWidth * (i - 1), histHeight - cvRound(histAfter.at<float>(i - 1))),
            Point(binWidth * i, histHeight - cvRound(histAfter.at<float>(i))),
            Scalar(0, 0, 255), 2, 8, 0);
    }

    imshow("Histogram Before Equalization", histImageBefore);
    imshow("Histogram After Equalization", histImageAfter);

    waitEscPress();
}