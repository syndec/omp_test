#include <iostream>
#include <opencv2/opencv.hpp>
#include <omp.h>

using namespace std;
using namespace cv;

int main(int argc, char *argv[])
{
//    omp_set_num_threads(4);
    #pragma omp parallel for
    for(int i = 0; i < 10; i++)
    {
        cout << i << endl;
    }

    return 0;
}
