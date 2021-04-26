#include "omp_resize.hpp"

void resizeBilinear(int32_t* pixels, int32_t* temp, int w, int h, int w2, int h2)
{
	int32_t a, b, c, d, x, y, index;
	float x_ratio = ((float)(w - 1)) / w2;
	float y_ratio = ((float)(h - 1)) / h2;
	float x_diff, y_diff, blue, red, green;
	int offset = 0;
	for (int i = 0; i < h2; i++)
	{
		for (int j = 0; j < w2; j++)
		{
			x = (int)(x_ratio * j);
			y = (int)(y_ratio * i);
			x_diff = (x_ratio * j) - x;
			y_diff = (y_ratio * i) - y;
			index = (y*w + x);
			a = pixels[index];
			b = pixels[index + 1];
			c = pixels[index + w];
			d = pixels[index + w + 1];

			// blue element
			// Yb = Ab(1-w)(1-h) + Bb(w)(1-h) + Cb(h)(1-w) + Db(wh)
			blue = (a & 0xff)*(1 - x_diff)*(1 - y_diff) + (b & 0xff)*(x_diff)*(1 - y_diff) +
				(c & 0xff)*(y_diff)*(1 - x_diff) + (d & 0xff)*(x_diff*y_diff);

			// green element
			// Yg = Ag(1-w)(1-h) + Bg(w)(1-h) + Cg(h)(1-w) + Dg(wh)
			green = ((a >> 8) & 0xff)*(1 - x_diff)*(1 - y_diff) + ((b >> 8) & 0xff)*(x_diff)*(1 - y_diff) +
				((c >> 8) & 0xff)*(y_diff)*(1 - x_diff) + ((d >> 8) & 0xff)*(x_diff*y_diff);

			// red element
			// Yr = Ar(1-w)(1-h) + Br(w)(1-h) + Cr(h)(1-w) + Dr(wh)
			red = ((a >> 16) & 0xff)*(1 - x_diff)*(1 - y_diff) + ((b >> 16) & 0xff)*(x_diff)*(1 - y_diff) +
				((c >> 16) & 0xff)*(y_diff)*(1 - x_diff) + ((d >> 16) & 0xff)*(x_diff*y_diff);

			temp[offset++] =
				0xff000000 |
				((((int32_t)red) << 16) & 0xff0000) |
				((((int32_t)green) << 8) & 0xff00) |
				((int32_t)blue);
		}
	}
}

int32_t* cvtMat2Int32(const cv::Mat& srcImage)
{
	int32_t *result = new int32_t[srcImage.cols*srcImage.rows];
	int offset = 0;

	for (int i = 0; i<srcImage.cols*srcImage.rows * 3; i += 3)
	{
		int32_t blue = srcImage.data[i];
		int32_t green = srcImage.data[i + 1];
		int32_t red = srcImage.data[i + 2];
		result[offset++] =
			0xff000000 |
			((((int32_t)red) << 16) & 0xff0000) |
			((((int32_t)green) << 8) & 0xff00) |
			((int32_t)blue);
	}

	return result;
}

void cvtInt322Mat(int32_t *pxArray, cv::Mat& outImage)
{
	int offset = 0;
	for (int i = 0; i<outImage.cols*outImage.rows * 3; i += 3)
	{
		int32_t a = pxArray[offset++];
		int32_t blue = a & 0xff;
		int32_t green = ((a >> 8) & 0xff);
		int32_t red = ((a >> 16) & 0xff);
		outImage.data[i] = blue;
		outImage.data[i + 1] = green;
		outImage.data[i + 2] = red;
	}
	return;
}

#ifdef OMP_PARALLEL
void resizeBilinear_omp(int* pixels, int32_t* temp, int w, int h, int w2, int h2)
{
	float x_ratio = ((float)(w - 1)) / w2;
	float y_ratio = ((float)(h - 1)) / h2;
	omp_set_num_threads(4);
#pragma omp parallel //num_threads(8)
	{
#pragma omp for //schedule(static, w /24)
		for (int i = 0; i < h2; i++)
		{
			int32_t a, b, c, d, x, y, index;
			float x_diff, y_diff, blue, red, green;
			int offset = i*w2;
			for (int j = 0; j < w2; j++)
			{
				x = (int)(x_ratio * j);
				y = (int)(y_ratio * i);
				x_diff = (x_ratio * j) - x;
				y_diff = (y_ratio * i) - y;
				index = (y*w + x);
				a = pixels[index];
				b = pixels[index + 1];
				c = pixels[index + w];
				d = pixels[index + w + 1];

				// blue element
				// Yb = Ab(1-w)(1-h) + Bb(w)(1-h) + Cb(h)(1-w) + Db(wh)
				blue = (a & 0xff)*(1 - x_diff)*(1 - y_diff) + (b & 0xff)*(x_diff)*(1 - y_diff) +
					(c & 0xff)*(y_diff)*(1 - x_diff) + (d & 0xff)*(x_diff*y_diff);

				// green element
				// Yg = Ag(1-w)(1-h) + Bg(w)(1-h) + Cg(h)(1-w) + Dg(wh)
				green = ((a >> 8) & 0xff)*(1 - x_diff)*(1 - y_diff) + ((b >> 8) & 0xff)*(x_diff)*(1 - y_diff) +
					((c >> 8) & 0xff)*(y_diff)*(1 - x_diff) + ((d >> 8) & 0xff)*(x_diff*y_diff);

				// red element
				// Yr = Ar(1-w)(1-h) + Br(w)(1-h) + Cr(h)(1-w) + Dr(wh)
				red = ((a >> 16) & 0xff)*(1 - x_diff)*(1 - y_diff) + ((b >> 16) & 0xff)*(x_diff)*(1 - y_diff) +
					((c >> 16) & 0xff)*(y_diff)*(1 - x_diff) + ((d >> 16) & 0xff)*(x_diff*y_diff);

				temp[offset++] =
					0xff000000 |
					((((int32_t)red) << 16) & 0xff0000) |
					((((int32_t)green) << 8) & 0xff00) |
					((int32_t)blue);
			}
		}
	}
}
#endif
