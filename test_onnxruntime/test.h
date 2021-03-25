#include <assert.h>
#include <vector>
#include <onnxruntime_cxx_api.h>
#include <opencv2/dnn.hpp>
#include <opencv2/opencv.hpp>
#include <cstdlib>
using namespace std;
using namespace cv;
using namespace cv::dnn;

#define AVS_PREONNX_MAX_IN_C		12000
#define AVS_PREONNX_MAX_DIM			100
#define AVS_PREONNX_OUT_C			64

void dnn_demo();

void dnn_test();

void onnx_demo();

void onnx_test();
