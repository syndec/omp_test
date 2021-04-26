#pragma once
#ifndef __OMP_RESIZE_H
#define __OMP_RESIZE_H

#include <stdint.h>
#include <stdio.h>
#include <omp.h>
#include <opencv2/opencv.hpp>

#ifdef OMP_PARALLEL
void resizeBilinear_omp(int* pixels, int32_t* temp, int w, int h, int w2, int h2);
#endif

void resizeBilinear(int32_t* pixels, int32_t* temp, int w, int h, int w2, int h2);

int32_t* cvtMat2Int32(const cv::Mat& srcImage);

void cvtInt322Mat(int32_t *pxArray, cv::Mat& outImage);

#endif // !__OMP_RESIZE_H