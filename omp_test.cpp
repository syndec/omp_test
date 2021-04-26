#include <sys/time.h>
#include "common/omp_resize.hpp"

#define RESIZE_CALLS_NUM 1000

int main(int argc, char **argv)
{
	cv::Mat image;
	cv::Mat image_resized_cpu;
	cv::Mat image_resized_omp;
	cv::Size newSz(1920/2, 1080/2); //1920x1080
	int32_t *argb = NULL;
	int32_t *argb_res_cpu = new int32_t[newSz.width*newSz.height];
	int32_t *argb_res_omp = new int32_t[newSz.width*newSz.height];

	if (argc < 2)
	{
		printf("Usage:\n\t %s path_to_image\n", argv[0]);
		return -1;
	}

	//const char fname[] = "lena.jpg";
	//image = cv::imread(fname, 1);
	image = cv::imread(argv[1], 1);
	if (image.empty())
	{
		printf("Can't load image %s\n", argv[1]);
		return -1;
	}

	argb = cvtMat2Int32(image);

	struct timeval start, end;
	float delta_time = 0;

#ifdef OMP_PARALLEL
	//OpenMP block start
	gettimeofday(&start, NULL); 
	for (int i = 0; i < RESIZE_CALLS_NUM; i++)
	{
		resizeBilinear_omp(argb, argb_res_omp, image.cols, image.rows, newSz.width, newSz.height);
	}
	gettimeofday(&end, NULL); 
	delta_time = (end.tv_sec - start.tv_sec)*1000000 + (end.tv_usec-start.tv_usec);
	printf("Time CPU (OpenMP): %f us.\n", delta_time);
	//OpenMP block end
#else
	//cpu (no OpenMP) block start
	gettimeofday(&start, NULL); 
	for (int i = 0; i < RESIZE_CALLS_NUM; i++)
	{
		resizeBilinear(argb, argb_res_cpu, image.cols, image.rows, newSz.width, newSz.height);
	}
	gettimeofday(&end, NULL); 
	delta_time = (end.tv_sec - start.tv_sec)*1000000 + (end.tv_usec-start.tv_usec);
	printf("Time CPU: %f us.\n", delta_time);
	//cpu (no OpenMP) block end

#endif
	
#ifdef OMP_PARALLEL
    image_resized_omp = cv::Mat(newSz, CV_8UC3);
    cvtInt322Mat(argb_res_omp, image_resized_omp);
	//cv::imshow("Resized_OMP", image_resized_omp);
	cv::imwrite("omp_resized.jpg", image_resized_omp);
#else
	//show result images of each module
	image_resized_cpu = cv::Mat(newSz, CV_8UC3);
	cvtInt322Mat(argb_res_cpu, image_resized_cpu);

	//cv::imshow("Original", image);
	//cv::imshow("Resized_CPU", image_resized_cpu);
	cv::imwrite("cpu_resized.jpg", image_resized_cpu);
#endif

	//cv::waitKey(0);

	delete[] argb_res_cpu, argb_res_omp;
	delete[] argb;

	return 0;
}
