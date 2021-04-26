**简介**

测试图像resize的OpenMP加速效果。

代码结构如下：

omp_test
├── build.sh
├── clear.sh
├── CMakeLists.txt
├── common
│   ├── CMakeLists.txt
│   ├── omp_resize.cpp
│   └── omp_resize.hpp
├── data
│   └── lena.jpg
├── omp_test.cpp
└── README.md

执行./build.sh脚本，完成目标文件生成，并安装至./bin文件夹

* build.sh -DOMP=y 打开OpenMP选项

**运行测试**

./bin/omp_test data/lena.jpg
(time ./bin/omp_test data/lena.jpg)

**依赖**

OpenCV3.x

**resize的主体代码参考了以下项目**

https://github.com/Keylost/BilinearImageResize
