// HyperView43.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "opencv2\opencv.hpp"
#include <stdio.h>
#include <math.h>

#include <omp.h>
#include <chrono>
#include <filesystem>

typedef std::chrono::high_resolution_clock Clock;

float* polynom(float *input_array,int length, float xlog, float unlog, float x1, float x2, float x3, float x4, float x5) {
	float* output_array;
	output_array = new float[length];
	for (int i=0; i < length; i++) {
		output_array[i] = log(unlog *input_array[i]+1)*xlog + pow(input_array[i], 5)*x5 + pow(input_array[i], 4)*x4 + pow(input_array[i], 3)*x3 + pow(input_array[i], 2)*x2 + input_array[i] * x1;

	}
	float max = 0;
	for (int i = 0; i < length; i++) {
		if (output_array[i] > max) {
			max = output_array[i];
		}
	}
	for (int i = 0; i < length; i++) {
		output_array[i] = output_array[i] / max;

	}
	return output_array;
}


cv::Vec3b BilinearInterpolation(float x, float y, cv::Mat* img_for_distortion)
{
	{


		int x0 = cv::borderInterpolate(x, img_for_distortion->cols, cv::BORDER_REFLECT_101);
		int x1 = cv::borderInterpolate(x + 1, img_for_distortion->cols, cv::BORDER_REFLECT_101);
		int y0 = cv::borderInterpolate(y, img_for_distortion->rows, cv::BORDER_REFLECT_101);
		int y1 = cv::borderInterpolate(y + 1, img_for_distortion->rows, cv::BORDER_REFLECT_101);

		// x = (int)x;
		//y = (int)y;

		float a = x - (int)x;
		float c = y - (int)y;

		uchar b = (uchar)cvRound((img_for_distortion->at<cv::Vec3b>(y0, x0)[0] * (1.f - a) + img_for_distortion->at<cv::Vec3b>(y0, x1)[0] * a) * (1.f - c)
			+ (img_for_distortion->at<cv::Vec3b>(y1, x0)[0] * (1.f - a) + img_for_distortion->at<cv::Vec3b>(y1, x1)[0] * a) * c);
		uchar g = (uchar)cvRound((img_for_distortion->at<cv::Vec3b>(y0, x0)[1] * (1.f - a) + img_for_distortion->at<cv::Vec3b>(y0, x1)[1] * a) * (1.f - c)
			+ (img_for_distortion->at<cv::Vec3b>(y1, x0)[1] * (1.f - a) + img_for_distortion->at<cv::Vec3b>(y1, x1)[1] * a) * c);
		uchar r = (uchar)cvRound((img_for_distortion->at<cv::Vec3b>(y0, x0)[2] * (1.f - a) + img_for_distortion->at<cv::Vec3b>(y0, x1)[2] * a) * (1.f - c)
			+ (img_for_distortion->at<cv::Vec3b>(y1, x0)[2] * (1.f - a) + img_for_distortion->at<cv::Vec3b>(y1, x1)[2] * a) * c);

		return cv::Vec3b(b, g, r);
	}

}



cv::Mat Hyperview(cv::Mat *widen_input_img, float* distortion_array, int size_of_array) {
	cv::Mat hyperview_img = cv::Mat(cv::Size(widen_input_img->cols, widen_input_img->rows), CV_8UC3);
	omp_set_num_threads(8);
#pragma omp parallel for //shared(hyperview_img)
	for (int x = 0; x < size_of_array; x++) {
		for (int y = 0; y < widen_input_img->rows; y++) {
			//printf("%d,%d, really: %d \n", x, y,widen_input_img->rows);

			hyperview_img.at<cv::Vec3b>(y, x) = BilinearInterpolation(distortion_array[x], y, widen_input_img);

		}
	}
	return  hyperview_img;
}


std::string getPathName(std::string filename) {
	const size_t last_slash_idx = filename.find_last_of("\\/");
	if (std::string::npos != last_slash_idx)
	{
		filename.erase(0, last_slash_idx + 1);
	}

	// Remove extension if present.
	const size_t period_idx = filename.rfind('.');
	if (std::string::npos != period_idx)
	{
		filename.erase(period_idx);
	}
	return filename;
}


int Convert(char* filename) {
	cv::VideoCapture reader;
	//reader.open("D:/superviewtest/vid.mp4");
	//printf("%s", argv[1]);
	reader.open(filename);
	std::string new_filename = getPathName(filename) + "_Hyperview43.mp4";

	std::cout << "Converting this file: " + getPathName(filename) + "\nParameters: \n";
	//std::cout << "The path name is \"" << getPathName(argv[1]) << "\"\n";
	// cap is the object of class video capture that tries to capture Bumpy.mp4
	float FPS = reader.get(cv::CAP_PROP_FPS);
	int width = reader.get(cv::CAP_PROP_FRAME_WIDTH);
	int height = reader.get(cv::CAP_PROP_FRAME_HEIGHT);
	int frame_count = reader.get(cv::CAP_PROP_FRAME_COUNT);

	printf(" FPS: %f, Format: %dx%d \n", FPS, width, height);
	int new_width = (int)(width * (3.0 / 4.0)*(16.0 / 9.0));
	int half_new_width = new_width / 2;

	printf("New Parameters:\n FPS: %f, Format: %dx%d \n", FPS, new_width, height);
	std::cout << "New filename will be: " + new_filename + "\n";

	float* function_array = new float[half_new_width];
	float* distortion_array = new float[new_width];
	// vysledny pixel se poèíta jako bilin interpolace z puvodniho obrazu a uklada se do distorted image. Do funkce se vklada puvodni obraz
	for (int j = 0; j < half_new_width; j++) {
		if (j == 0) {
			function_array[j] = 0.0;
			continue;
		}
		function_array[j] = (float)j / half_new_width;
		//printf("%f \n", function_array[j]);
	}
	function_array = polynom(function_array, half_new_width, 150, 0.6, 1, 1, 0, 0, 0);



	for (int x = 0; x < half_new_width; x++) {
		distortion_array[x + half_new_width] = function_array[x] * half_new_width + half_new_width;
		distortion_array[x] = -function_array[(half_new_width - 1) - x] * half_new_width + half_new_width;
	}
	for (int j = 0; j < new_width; j++) {

		//printf("%f \n", distortion_array[j]);
	}

	cv::waitKey(100);


	if (!reader.isOpened())  // isOpened() returns true if capturing has been initialized.
	{
		std::cout << "Cannot open the video file. \n";
		return -1;
	}

	double fps = reader.get(cv::CAP_PROP_FPS); //get the frames per seconds of the video
											   // The function get is used to derive a property from the element.
											   // Example:
											   // CV_CAP_PROP_POS_MSEC :  Current Video capture timestamp.
											   // CV_CAP_PROP_POS_FRAMES : Index of the next frame.

	cv::namedWindow("Progress", cv::WINDOW_AUTOSIZE); //create a window called "MyVideo"
													  // first argument: name of the window.
													  // second argument: flag- types: 
													  // WINDOW_NORMAL : The user can resize the window.
													  // WINDOW_AUTOSIZE : The window size is automatically adjusted to fitvthe displayed image() ), and you cannot change the window size manually.
													  // WINDOW_OPENGL : The window will be created with OpenGL support.

	cv::VideoWriter outputVideo;                                        // Open the output
	

	auto codec = reader.get(cv::CAP_PROP_FOURCC);
	//codec = cv::VideoWriter::fourcc('M', 'P', '4', '4');
	codec = 0x7634706d;// 'mp4v' best bitrate/size ratio codec I found. I don't know how it works
	//codec = cv::VideoWriter::fourcc('x', '2', '6', '4');
	outputVideo.set(cv::VIDEOWRITER_PROP_QUALITY, 1);

	
	outputVideo.open(new_filename, codec, FPS, cv::Size(new_width, height), true);


	//printf("%d\n",(reader.get(cv::CAP_PROP_FOURCC)));
	//outputVideo.set(cv::VIDEOWRITER_PROP_QUALITY, 0.001);

	auto begin = Clock::now();
	auto one_frame_end = Clock::now();
	auto one_frame_begin = Clock::now();
	float avg_time = 0;
	float remaining_seconds = 0;
	int frame_counter = 0;
	std::string text = "";

	std::vector<float> all_times;
	while (frame_counter<frame_count)
	{
		frame_counter++;
		cv::Mat frame;
		cv::Mat widen;
		cv::Mat hyperview_img = cv::Mat(cv::Size(new_width, height), CV_8UC3);




		// Mat object is a basic image container. frame is an object of Mat.

		if (!reader.read(frame)) // if not success, break loop
								 // read() decodes and captures the next frame.
		{
			std::cout << "\n Cannot read the video file. \n";
			printf("frames: %d, frame_counter: %d\n", frame_count, frame_counter);
			break;
		}

		cv::resize(frame, widen, cv::Size(new_width, height));

		hyperview_img = Hyperview(&widen, distortion_array, new_width);

		//cv::Mat resized;
		float resize_factor = 480.0 / height;
		//cv::resize(frame, resized, cv::Size((int)(width*resize_factor), (int)(height*resize_factor)));
		//imshow("Progress", resized);
		cv::Mat resized_hyperview;
		cv::resize(hyperview_img, resized_hyperview, cv::Size((int)(hyperview_img.cols*resize_factor), (int)(hyperview_img.rows*resize_factor)));
		one_frame_end = Clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(one_frame_end - one_frame_begin).count();
		one_frame_begin = Clock::now();
		all_times.push_back(duration);
		avg_time = 0;
		for (int v = 0; v < all_times.size(); v++) {
			avg_time += all_times[v];
		}
		avg_time /= all_times.size();
		avg_time /= 1000000000;
		remaining_seconds = (frame_count - frame_counter)*avg_time;

		//sprintf_s(text, "One frame time: %0.2f, Remaining time: %0.2f",(avg_time), (remaining_seconds / 60));
		text = "One Frame Time: " + std::to_string(avg_time);
		cv::putText(resized_hyperview, text, cv::Point((int)0, (int)10), 1, 1, cv::Scalar(255, 255, 255, 0));
		text = "Remaining Time:" + std::to_string((int)(remaining_seconds / 60)) + "min" + " " + std::to_string((int)(60 * ((remaining_seconds / 60) - (int)(remaining_seconds / 60)))) + " seconds";
		cv::putText(resized_hyperview, text, cv::Point((int)0, (int)30), 1, 1, cv::Scalar(0, 128, 255, 0));


		imshow("Progress", resized_hyperview);
		// first argument: name of the window.
		// second argument: image to be shown(Mat object).

		outputVideo.write(hyperview_img);

		if (cv::waitKey(1) == 27) // Wait for 'esc' key press to exit
		{
			break;
		}

	}
}


int main(int argc, char** argv)
{
	/*  CAP_PROP_POS_MSEC Current position of the video file in milliseconds or video capture timestamp.
		CAP_PROP_POS_FRAMES 0 - based index of the frame to be decoded / captured next.
		CAP_PROP_POS_AVI_RATIO Relative position of the video file : 0 - start of the film, 1 - end of the film.
		CAP_PROP_FRAME_WIDTH Width of the frames in the video stream.
		CAP_PROP_FRAME_HEIGHT Height of the frames in the video stream.
		CAP_PROP_FPS Frame rate.
		CAP_PROP_FOURCC 4 - character code of codec.
		CAP_PROP_FRAME_COUNT Number of frames in the video file.
		CAP_PROP_FORMAT Format of the Mat objects returned by retrieve() .
		CAP_PROP_MODE Backend - specific value indicating the current capture mode.
		CAP_PROP_BRIGHTNESS Brightness of the image(only for cameras).
		CAP_PROP_CONTRAST Contrast of the image(only for cameras).
		CAP_PROP_SATURATION Saturation of the image(only for cameras).
		CAP_PROP_HUE Hue of the image(only for cameras).
		CAP_PROP_GAIN Gain of the image(only for cameras).
		CAP_PROP_EXPOSURE Exposure(only for cameras).
		CAP_PROP_CONVERT_RGB Boolean flags indicating whether images should be converted to RGB.
		CAP_PROP_WHITE_BALANCE Currently not supported
		CAP_PROP_RECTIFICATION Rectification flag for stereo cameras(note: only supported by DC1394 v 2.x backend currently)*/

	//"D:/superviewtest/vid.mp4"
	if (argc == 1) {
		std::cout << "Drop the video file on this .exe file, or add full path of the file to the args in command line. \nThis window will close in 10 seconds." << std::endl;
		cv::waitKey(10000);
		return 1;
	}

	for (int i = 1; i < argc; i++) {
		std::cout << argv[i] << std::endl;
		Convert(argv[i]);
	}

	
	cv::waitKey(0);

    return 0;
}

