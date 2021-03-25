// Copyright(c) Microsoft Corporation.All rights reserved.
// Licensed under the MIT License.
// https://github.com/microsoft/onnxruntime/blob/master/csharp/test/Microsoft.ML.OnnxRuntime.EndToEndTests.Capi/CXX_Api_Sample.cpp

#include "test.h"

int main(int argc, char* argv[]) {

	// dnn demo
	dnn_demo();

	// dnn test
	dnn_test();
	
	// onnx demo
	onnx_demo();

	// onnx test
	onnx_test();
	
	return 0;
}
