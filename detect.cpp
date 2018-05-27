#include <opencv2\opencv.hpp>
#include <iostream>
#include <string>
#include <stdio.h>
//#include <math.h>
//#include "rect_detector.h"
//#include "artifacts_detector.h"
//#include "mura_detector.h"
//#include "curve_detector.h"
#include <cv.hpp>
using namespace cv;
using namespace std;


//定义灰度图像变量
IplImage *g_GrayImage = NULL;
//定义二值化图片变量
IplImage *g_BinaryImage = NULL;
//定义二值化窗口标题
const char *WindowBinaryTitle = "二值化图片";
//定义滑块响应函数

//创建源图像窗口标题变量
const char *WindowSrcTitle = "灰度图像";
//创建滑块标题变量
const char *TheSliderTitle = "二值化阀值";




//const char *SrcPath = "C:\\Users\\lenovo\\Desktop\\1.jpg"; ////定义图片路径

IplImage *g_pGrayImage_liantong = NULL;
IplImage *g_pBinralyImage_liantong = NULL;

int contour_num = 0; //数字编号
char  number_buf[10];  ////数字编号存入数组，puttext

#define num_col 11   ////二维数组的列，每一个点缺陷信息的详细信息

long int liantong_all_area = 0; ////连通区域总面积
long int Rect_all_area = 0;  //// 保存最小外接矩形总的面积


////=====================================================================
struct my_struct1{
	double scale;  //// 定义显示图像的比例
	const int threshold_value_binaryzation;  ////定义第一次二值化阀值
	const int threshold_value_second_binaryzation;  ////定义第二次二值化阀值
};
my_struct1 picture = { 0.5, 50, 100 };

////=====================================================================
struct my_struct2{
	int Model1_k1;  ////图像膨胀腐蚀
	int Model1_k2;  ////图像膨胀腐蚀
	int Model2_k1;  ////图像膨胀腐蚀
	int Model2_k2;  ////图像膨胀腐蚀
};
my_struct2 value = { 5, 2, 3, 2 };

////=====================================================================
struct my_struct3{

	double maxarea;  ////最大缺陷面积
	double minarea;  ////最小显示保留的缺陷面积

	double font_scale;  ////字体大小
	int font_thickness; ////字体粗细

	const int Feature_value2_number; ////定义一个二维数组的列，即缺陷的个数

};
my_struct3 value2 = { 5000, 1000, 0.6, 0.8, 100 };

////=====================================================================
struct my_struct4{

	const int hough_Canny_thresh1;
	const int hough_Canny_thresh2;
	const int hough_Canny_kernel;

	const int cvHoughLines2_thresh; ////像素值大于多少才显示，值越大，显示的线段越少
	const int cvHoughLines2_param1; ////显示线段的最小长度
	const int cvHoughLines2_param2; ////线段之间的 最小间隔

};
my_struct4 Hough = { 50, 100, 3, 50, 20, 10 };


void fft2(IplImage *src, IplImage *dst)
{   //实部、虚部  
	IplImage *image_Re = 0, *image_Im = 0, *Fourier = 0;
	//   int i, j;  
	image_Re = cvCreateImage(cvGetSize(src), IPL_DEPTH_64F, 1);  //实部  
	//Imaginary part  
	image_Im = cvCreateImage(cvGetSize(src), IPL_DEPTH_64F, 1);  //虚部  
	//2 channels (image_Re, image_Im)  
	Fourier = cvCreateImage(cvGetSize(src), IPL_DEPTH_64F, 2);
	// Real part conversion from u8 to 64f (double)  
	cvConvertScale(src, image_Re);
	// Imaginary part (zeros)  
	cvZero(image_Im);
	// Join real and imaginary parts and stock them in Fourier image  
	cvMerge(image_Re, image_Im, 0, 0, Fourier);

	// Application of the forward Fourier transform  
	cvDFT(Fourier, dst, CV_DXT_FORWARD);
	cvReleaseImage(&image_Re);
	cvReleaseImage(&image_Im);
	cvReleaseImage(&Fourier);
}

void fft2shift(IplImage *src, IplImage *dst)
{
	IplImage *image_Re = 0, *image_Im = 0;
	int nRow, nCol, i, j, cy, cx;
	double scale, shift, tmp13, tmp24;
	image_Re = cvCreateImage(cvGetSize(src), IPL_DEPTH_64F, 1);
	//Imaginary part  
	image_Im = cvCreateImage(cvGetSize(src), IPL_DEPTH_64F, 1);
	cvSplit(src, image_Re, image_Im, 0, 0);
	//具体原理见冈萨雷斯数字图像处理p123  
	// Compute the magnitude of the spectrum Mag = sqrt(Re^2 + Im^2)  
	//计算傅里叶谱  
	cvPow(image_Re, image_Re, 2.0);
	cvPow(image_Im, image_Im, 2.0);
	cvAdd(image_Re, image_Im, image_Re);
	cvPow(image_Re, image_Re, 0.5);
	//对数变换以增强灰度级细节(这种变换使以窄带低灰度输入图像值映射  
	//一宽带输出值，具体可见冈萨雷斯数字图像处理p62)  
	// Compute log(1 + Mag);  
	cvAddS(image_Re, cvScalar(1.0), image_Re); // 1 + Mag  
	cvLog(image_Re, image_Re); // log(1 + Mag)  

	//Rearrange the quadrants of Fourier image so that the origin is at the image center  
	nRow = src->height;
	nCol = src->width;
	cy = nRow / 2; // image center  
	cx = nCol / 2;
	//CV_IMAGE_ELEM为OpenCV定义的宏，用来读取图像的像素值，这一部分就是进行中心变换  
	for (j = 0; j < cy; j++){
		for (i = 0; i < cx; i++){
			//中心化，将整体份成四块进行对角交换  
			tmp13 = CV_IMAGE_ELEM(image_Re, double, j, i);
			CV_IMAGE_ELEM(image_Re, double, j, i) = CV_IMAGE_ELEM(
				image_Re, double, j + cy, i + cx);
			CV_IMAGE_ELEM(image_Re, double, j + cy, i + cx) = tmp13;

			tmp24 = CV_IMAGE_ELEM(image_Re, double, j, i + cx);
			CV_IMAGE_ELEM(image_Re, double, j, i + cx) =
				CV_IMAGE_ELEM(image_Re, double, j + cy, i);
			CV_IMAGE_ELEM(image_Re, double, j + cy, i) = tmp24;
		}
	}
	//归一化处理将矩阵的元素值归一为[0,255]  
	//[(f(x,y)-minVal)/(maxVal-minVal)]*255  
	double minVal = 0, maxVal = 0;
	// Localize minimum and maximum values  
	cvMinMaxLoc(image_Re, &minVal, &maxVal);
	// Normalize image (0 - 255) to be observed as an u8 image  
	scale = 255 / (maxVal - minVal);
	shift = -minVal * scale;
	cvConvertScale(image_Re, dst, scale, shift);
	cvReleaseImage(&image_Re);
	cvReleaseImage(&image_Im);
}


////=====================================================================
//自适应中值滤波
uchar adaptiveProcess(const Mat &im, int row, int col, int kernelSize, int maxSize)
{
	vector<uchar> pixels;
	for (int a = -kernelSize / 2; a <= kernelSize / 2; a++)
		for (int b = -kernelSize / 2; b <= kernelSize / 2; b++)
		{
			pixels.push_back(im.at<uchar>(row + a, col + b));
		}
	sort(pixels.begin(), pixels.end());
	auto min = pixels[0];
	auto max = pixels[kernelSize * kernelSize - 1];
	auto med = pixels[kernelSize * kernelSize / 2];
	auto zxy = im.at<uchar>(row, col);
	if (med > min && med < max)
	{
		// to B
		if (zxy > min && zxy < max)
			return zxy;
		else
			return med;
	}
	else
	{
		kernelSize += 2;
		if (kernelSize <= maxSize)
			return adaptiveProcess(im, row, col, kernelSize, maxSize); // 增大窗口尺寸，继续A过程。
		else
			return med;
	}
}

int** on_trackbar(const char *SrcPath = "1.BMP"){

	CvSeq* contour = 0;
	CvSeq* _contour = contour;

	//定义存放数组的二维数组，返回指针数组
	int** Feature_value2 = 0;
	Feature_value2 = new int*[value2.Feature_value2_number];

	IplImage *SrcImage_or;
	CvSize src_sz;
	////===============================================================================================预处理
	//载入原图
	printf("预处理\n");
	IplImage *SrcImage_origin = cvLoadImage(SrcPath, CV_LOAD_IMAGE_UNCHANGED);

	//缩放	
	src_sz.width = SrcImage_origin->width* picture.scale;
	src_sz.height = SrcImage_origin->height* picture.scale;
	SrcImage_or = cvCreateImage(src_sz, SrcImage_origin->depth, SrcImage_origin->nChannels);
	cvResize(SrcImage_origin, SrcImage_or, CV_INTER_CUBIC);
	//cvNamedWindow("原图", 0);
	////显示原图到原图窗口
	//cvShowImage("原图", SrcImage_or);

	//单通道灰度化处理
	if (SrcImage_or->nChannels > 1)
	{
		g_GrayImage = cvCreateImage(cvSize(SrcImage_or->width, SrcImage_or->height), IPL_DEPTH_8U, 1);
		cvCvtColor(SrcImage_or, g_GrayImage, CV_BGR2GRAY);
	}
	else
		g_GrayImage = SrcImage_or;
	//抑制曝光过度
	//IplImage *src_threshold = cvCreateImage(cvGetSize(SrcImage_or), IPL_DEPTH_8U, 1);
	//cvThreshold(SrcImage_or, src_threshold, 100, 255, CV_THRESH_BINARY);
	for (int i = 0; i < src_sz.height; i++)
	{
		for (int j = 0; j < src_sz.width; j++)
		{
			if (cvGet2D(g_GrayImage, i, j).val[0]>100)
				cvSet2D(g_GrayImage, i, j, 100);
		}
	}
	//cvNamedWindow("抑制", 0);
	//////显示原图到原图窗口
	//cvShowImage("抑制", SrcImage_or);
	/// 应用直方图均衡化
	IplImage *src_his = cvCreateImage(src_sz, g_GrayImage->depth, g_GrayImage->nChannels);
	cvEqualizeHist(g_GrayImage, src_his);
	cvSaveImage("均衡化.jpg",src_his);
	//fft变换
	//IplImage *Fourier = cvCreateImage(cvGetSize(src_his), IPL_DEPTH_64F, 2);
	//IplImage *dst = cvCreateImage(cvGetSize(src_his), IPL_DEPTH_64F, 2);
	//IplImage *ImageRe = cvCreateImage(cvGetSize(src_his), IPL_DEPTH_64F, 1);
	//IplImage *ImageIm = cvCreateImage(cvGetSize(src_his), IPL_DEPTH_64F, 1);
	//IplImage *Image = cvCreateImage(cvGetSize(src_his), src_his->depth, src_his->nChannels);
	//IplImage *ImageDst = cvCreateImage(cvGetSize(src_his), src_his->depth, src_his->nChannels);
	//double Minval, Maxval;
	//double scale;
	//double shift;
	//fft2(src_his, Fourier);                  //傅里叶变换  
	//fft2shift(Fourier, Image);          //中心化  
	//cvDFT(Fourier, dst, CV_DXT_INV_SCALE);//实现傅里叶逆变换，并对结果进行缩放  
	//cvSplit(dst, ImageRe, ImageIm, 0, 0);

	////对数组每个元素平方并存储在第二个参数中  
	//cvPow(ImageRe, ImageRe, 2);
	//cvPow(ImageIm, ImageIm, 2);
	//cvAdd(ImageRe, ImageIm, ImageRe, NULL);
	//cvPow(ImageRe, ImageRe, 0.5);
	//cvMinMaxLoc(ImageRe, &Minval, &Maxval, NULL, NULL);
	//scale = 255 / (Maxval - Minval);
	//shift = -Minval * scale;
	////将shift加在ImageRe各元素按比例缩放的结果上，存储为ImageDst  
	//cvConvertScale(ImageRe, src_his, scale, shift);
	//cvEqualizeHist(src_his, g_GrayImage);
	//cvAbsDiff(g_GrayImage, ImageRe, g_GrayImage);
	//cvNamedWindow("原图", CV_WINDOW_AUTOSIZE);
	////显示原图到原图窗口
	//cvShowImage("原图", SrcImage);


	//创建二值化原图
	printf("二值化\n");
	g_BinaryImage = cvCreateImage(cvGetSize(g_GrayImage), IPL_DEPTH_8U, 1);

	cvThreshold(g_GrayImage, g_BinaryImage, picture.threshold_value_binaryzation, 255, CV_THRESH_BINARY);
	cvSaveImage("二值化.jpg", g_BinaryImage);
	//动态阈值
	//int blockSize = 7;
	//int constValue = 10;
	//cvAdaptiveThreshold(g_GrayImage, g_BinaryImage, 255, CV_ADAPTIVE_THRESH_GAUSSIAN_C, CV_THRESH_BINARY, blockSize, constValue);
	//显示二值化后的图片
	//cvNamedWindow("二值化", 0);
	//cvShowImage("二值化", g_BinaryImage);
	//g_BinaryImage = cvCloneImage(g_BinaryImage);  //// 膨胀腐蚀

	////===============================================================================================图像膨胀腐蚀
	//////先cvDilate后cvErode，先膨胀后腐蚀，这个为闭合操作，图片中断裂处会缝合。
	//////利用这个操作可以填充细小空洞，连接临近物体，平滑物体边缘，同时不明显改变物体面积
	printf("膨胀腐蚀\n");
	IplImage* temp_cvDilate = cvCreateImage(cvGetSize(g_BinaryImage), IPL_DEPTH_8U, 1);
	IplImage* temp_cvErode = cvCreateImage(cvGetSize(g_BinaryImage), IPL_DEPTH_8U, 1);
	IplImage* temp_cvErode_cvErode = cvCreateImage(cvGetSize(g_BinaryImage), IPL_DEPTH_8U, 1);

	IplConvKernel * myModel1;
	myModel1 = cvCreateStructuringElementEx( //自定义5*5,参考点（3,3）的矩形模板
		value.Model1_k1, value.Model1_k1, value.Model1_k2, value.Model1_k2, CV_SHAPE_ELLIPSE//CV_SHAPE_ELLIPSE, 椭圆元素;
		);
	IplConvKernel * myModel2;
	myModel2 = cvCreateStructuringElementEx( //自定义5*5,参考点（3,3）的矩形模板
		value.Model2_k1, value.Model2_k1, value.Model2_k2, value.Model2_k2, CV_SHAPE_RECT	//CV_SHAPE_RECT, 长方形元素;
		);




	//////先膨胀后腐蚀
	cvDilate(g_BinaryImage, temp_cvDilate, myModel1, 1);//膨胀
	cvErode(temp_cvDilate, temp_cvErode_cvErode, myModel2, 3);//腐蚀

	//namedWindow("temp_cvErode_cvErode", CV_WINDOW_AUTOSIZE);
	//cvShowImage("temp_cvErode_cvErode", temp_cvErode_cvErode);

	g_BinaryImage = cvCloneImage(temp_cvErode_cvErode);  //// 保存膨胀腐蚀结果


	///////================================================================================================检测连通区域
	printf("检测连通区域\n");
	CvMemStorage *liantong_storage = cvCreateMemStorage();
	IplImage* liantogn_dst = cvCreateImage(cvGetSize(g_BinaryImage), 8, 3);
	//提取轮廓   
	cvFindContours(g_BinaryImage, liantong_storage, &contour, sizeof(CvContour), CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);
	cvZero(liantogn_dst);//清空数组   

	IplImage *result = cvCreateImage(cvSize(liantogn_dst->width, liantogn_dst->height), IPL_DEPTH_8U, 3);
	cvCvtColor(src_his, result, CV_GRAY2BGR);//先转成RGB图像，方便上色

	int n = -1, m = 0;//n为面积最大轮廓索引，m为迭代索引   
	////-----------------------------------------------------------对连通区域做处理
	for (; contour != 0; contour = contour->h_next)
	{

		double tmparea = fabs(cvContourArea(contour));
		
		if (tmparea <= value2.minarea)
		{
			cvSeqRemove(contour, 0); //删除面积小于设定值的轮廓   
			continue;
		}
		else
		{
			liantong_all_area = liantong_all_area + tmparea;
		}

		CvRect aRect = cvBoundingRect(contour, 0);
		//if ((aRect.width / aRect.height)<1)
		//{
		//	cvSeqRemove(contour, 0); //删除宽高比例小于设定值的轮廓   
		//	continue;
		//}
		CvBox2D box=cvMinAreaRect2(contour);//外接矩形
		//printf("%d   %d  %f   %f  %f   %f  \n ", g_BinaryImage->width, g_BinaryImage->height, box.center.x, box.center.y, box.size.width, box.size.height);
		//if (box.center.x - box.size.width / 2 <= 0 || box.center.x + box.size.width / 2 >= g_BinaryImage->width ||
		//	box.center.y - box.size.height / 2 <= 0 || box.center.y + box.size.height / 2 >= g_BinaryImage->height)
		//{
		//	
		//	cvSeqRemove(contour, 0); //删除碰到边界的轮廓   
		//	continue;
		//}

		CvScalar color = CV_RGB(rand() & 255, rand() & 255, rand() & 255);//随机颜色
		//CvScalar color = CV_RGB(255,0,0);
		if (tmparea > value2.maxarea)
		{
			//value2.maxarea = tmparea;
			n = m;
			cvDrawContours(liantogn_dst, contour, color, color, -1, -1, 8);//绘制外部和内部的轮廓 
			cvSeqRemove(contour, 0); //删除过大的图像  
			continue;
		}
		m++;

		cvDrawContours(liantogn_dst, contour, color, color, -1, -1, 8);//绘制外部和内部的轮廓   
		cvDrawContours(result, contour, color, color, -1, -1, 8);//绘制外部和内部的轮廓 
		//cvRectangle(src_his, CvPoint(box.center.y-box.size.height,box.center.x),
		//	CvPoint(box.center.y + box.size.height, box.center.x + box.size.width), 3, 4, 1);//好像坐标不太对劲，因为没考虑矩形框的旋转
		//cvSaveImage("fanse.jpg", liantogn_dst);
	}
	//cvNamedWindow("检测结果", 0);
	//cvShowImage("检测结果", liantogn_dst);
	cvSaveImage("连通图.jpg", liantogn_dst);

	cvNamedWindow("检测结果", 0);
	cvShowImage("检测结果", result);
	cvSaveImage("jian.jpg", result);
	printf("结束\n");
	return  Feature_value2; ////返回该数组
}

//划痕检测
void CheckScratch()
{
	Mat image, imagemen, diff, Mask;
	image = imread("C:\\Users\\lenovo\\Desktop\\img\\IMG_1725.BMP");
	//image = imread("F:\\workplace\\matlab_c\\matlab_c\\saveImage.jpg");

	//均值模糊
	printf("高斯滤波\n");
	GaussianBlur(image, imagemen, Size(5, 5),0,0);

	//图像差分
	printf("差分操作\n");
	subtract(imagemen, image, diff);

	//同动态阈值分割dyn_threshold
	printf("阈值分割\n");
	threshold(diff, Mask, 30, 255, THRESH_BINARY_INV);
	//cvNamedWindow("imagemean", 0);
	//imshow("imagemean", imagemen);
	//cvNamedWindow("diff", 0);
	//imshow("diff", diff);
	//cvNamedWindow("Mask", 0);
	//imshow("Mask", Mask);
	Mat imagegray;
	cvtColor(Mask, imagegray, CV_RGB2GRAY);
	vector<vector<Point>> contours;
	vector<Vec4i>hierarchy;
	
	findContours(imagegray, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));
	Mat drawing = Mat::zeros(Mask.size(), CV_8U);
	int j = 0;
	printf("contours\n");
	for (int i = 0; i < contours.size(); i++)
	{
		Moments moms = moments(Mat(contours[i]));
		double area = moms.m00;//零阶矩即为二值图像的面积&nbsp; double area = moms.m00;零阶距.m00表示轮廓的面积，.m10为轮廓重心
		//如果面积超出了设定的范围，则不再考虑该斑点&nbsp;
		if (area > 100 && area < 1000000)
		{
			drawContours(drawing, contours, i, Scalar(255), FILLED, 8, hierarchy, 0, Point());
			j = j + 1;

		}
	}

	Mat element15(3, 3, CV_8U, Scalar::all(1));
	Mat close;
	morphologyEx(drawing, close, MORPH_CLOSE, element15);
	//cvNamedWindow("drawing", 0);
	//imshow("drawing", drawing);
	//waitKey(0);
	vector<vector<Point> > contours1;
	vector<Vec4i> hierarchy1;
	findContours(close, contours1, hierarchy1, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));
	//cvNamedWindow("close", 0);
	//imshow("close", close);
	//waitKey(0);
	j = 0;
	int m = 0;
	printf("contours num:%d\n", contours1.size());
	for (int i = 0; i < contours1.size(); i++)
	{
		Moments moms = moments(Mat(contours1[i]));
		double area = moms.m00;//零阶矩即为二值图像的面积&nbsp; double area = moms.m00;
		//如果面积超出了设定的范围，则不再考虑该斑点&nbsp; 

		double area1 = contourArea(contours1[i]);
		drawContours(image, contours1, i, Scalar(0, 0, 255), FILLED, 80, hierarchy1, 0, Point());
		if (area > 100 && area < 1000000)
		{
			drawContours(image, contours1, i, Scalar(0, 0, 255), FILLED, 8, hierarchy1, 0, Point());
			j = j + 1;

		}
		else if (area >= 0 && area <= 50)
		{
			drawContours(image, contours1, i, Scalar(255, 0, 0), FILLED, 8, hierarchy1, 0, Point());
			m = m + 1;

		}
	}

	char t[256];
	sprintf_s(t, "%01d", j);
	string s = t;
	string txt = "Long NG : " + s;
	putText(image, txt, Point(20, 30), CV_FONT_HERSHEY_COMPLEX, 1,
		Scalar(0, 0, 255), 2, 8);

	sprintf_s(t, "%01d", m);
	s = t;
	txt = "Short NG : " + s;
	putText(image, txt, Point(20, 60), CV_FONT_HERSHEY_COMPLEX, 1,
		Scalar(255, 0, 0), 2, 8);
	imwrite("result.bmp", image);
	printf("finished");
	//cvDestroyWindow("imagemean");
	//cvDestroyWindow("diff");
	//cvDestroyWindow("Mask");
	//cvDestroyWindow("drawing");
	//cvDestroyWindow("close");
}


int main(){
	//rect_detector detector;
	//artifacts_detector artifacts_detector1;
	//mura_detector mura_detector1;
	//curve_detector curve_detector1;
	//cv::namedWindow("original");
	//cv::Mat img = cv::imread("C:\\Users\\lenovo\\Desktop\\img\\11.JPG");
	//cv::imshow("original", img);
	////cv::waitKey(1);
	////    mura_detector1.enable_debug();
	//artifacts_detector1.enable_debug();
	////    detector.enable_debug();
	////    curve_detector1.enable_debug();
	////cv::Mat scr = detector.detect_screen(img);
	//cv::Mat scr = img;
	//artifacts_detector1.detect_artifacts(img);
	//mura_detector1.detect_mura(scr);
	//curve_detector1.detect_curve(scr);
	//cv::imwrite("screen.jpg", scr);
	//cv::waitKey();
	//return 0;
	//CheckScratch();
	int **Tan_return;


	Tan_return = on_trackbar("1.BMP");
	cvWaitKey(0);


	////销毁窗口，释放图片（实际运行退出时一定要销毁窗口）  
	//cvDestroyWindow(WindowBinaryTitle);  
	//cvDestroyWindow(WindowSrcTitle);  
	//cvReleaseImage(&g_BinaryImage);  
	//cvReleaseImage(&g_GrayImage);  
	//cvReleaseImage(&SrcImage);  
	getchar();
	return 0;
}

