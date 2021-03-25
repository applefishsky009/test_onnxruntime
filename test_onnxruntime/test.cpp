#include "test.h"

/* 读取图像的1000个分类标记文本数据 */
// https://blog.csdn.net/qq_30815237/article/details/87916157
vector<String> readClasslabels(String &labelFile) {
	std::vector<String> classNames;
	std::ifstream fp(labelFile);
	if (!fp.is_open())
	{
		std::cerr << "File with classes labels not found: " << labelFile << std::endl;
		exit(-1);
	}
	std::string name;
	while (!fp.eof())
	{
		std::getline(fp, name);
		if (name.length())
			classNames.push_back(name.substr(name.find(' ') + 1));
	}
	fp.close();
	return classNames;
}

void dnn_demo() {

	String modelTxt = "bvlc_googlenet.prototxt";
	String modelBin = "bvlc_googlenet.caffemodel";
	String labelFile = "synset_words.txt";

	Mat testImage = imread("D:/visual/pointpillars/verify_undistort.png"); // 读取测试图片
	// create googlenet with caffemodel text and bin
	Net net = dnn::readNetFromCaffe(modelTxt, modelBin);
	// 读取分类数据
	vector<String> labels = readClasslabels(labelFile);
	//GoogLeNet accepts only 224x224 RGB-images
	// Mat inputBlob = blobFromImage(testImage, 1, Size(224, 224), Scalar(104, 117, 123));//mean: Scalar(104, 117, 123)

	int sizes[] = { 1, 3, 224, 224 };
	unsigned char *test_blob = (unsigned char *)malloc(3 * 224 * 224 * sizeof(float));
	Mat inputBlob(4, sizes, CV_32F, test_blob);

	// 支持1000个图像分类检测
	Mat prob;
	// 循环10+
	for (int i = 0; i < 10; i++)
	{
		// 输入
		net.setInput(inputBlob, "data");
		// 分类预测
		prob = net.forward("prob");
	}
	// 读取分类索引，最大与最小值
	Mat probMat = prob.reshape(1, 1); //reshape the blob to 1x1000 matrix // 1000个分类
	Point classNumber;
	double classProb;
	minMaxLoc(probMat, NULL, &classProb, NULL, &classNumber); // 可能性最大的一个
	int classIdx = classNumber.x; // 分类索引号
	printf("\n current image classification : %s, possible : %.2f \n", labels.at(classIdx).c_str(), classProb);
}

void dnn_test() {

	dnn::Net net = cv::dnn::readNetFromONNX("pfe_1.7.0.onnx"); //读取网络和参数
	net.setPreferableBackend(DNN_BACKEND_OPENCV);
	net.setPreferableTarget(DNN_TARGET_CPU);

	int sizes[] = { 1, 1, AVS_PREONNX_MAX_IN_C, AVS_PREONNX_MAX_DIM };
	int sizes2[] = { 1, AVS_PREONNX_MAX_IN_C };

	unsigned char *test_pillar_x = (unsigned char *)malloc(AVS_PREONNX_MAX_IN_C * AVS_PREONNX_MAX_DIM * sizeof(float));
	unsigned char *test_pillar_y = (unsigned char *)malloc(AVS_PREONNX_MAX_IN_C * AVS_PREONNX_MAX_DIM * sizeof(float));
	unsigned char *test_pillar_z = (unsigned char *)malloc(AVS_PREONNX_MAX_IN_C * AVS_PREONNX_MAX_DIM * sizeof(float));
	unsigned char *test_pillar_i = (unsigned char *)malloc(AVS_PREONNX_MAX_IN_C * AVS_PREONNX_MAX_DIM * sizeof(float));
	unsigned char *test_num_voxels = (unsigned char *)malloc(AVS_PREONNX_MAX_IN_C * sizeof(float));
	unsigned char *test_x_sub_shaped = (unsigned char *)malloc(AVS_PREONNX_MAX_IN_C * AVS_PREONNX_MAX_DIM * sizeof(float));
	unsigned char *test_y_sub_shaped = (unsigned char *)malloc(AVS_PREONNX_MAX_IN_C * AVS_PREONNX_MAX_DIM * sizeof(float));
	unsigned char *test_mask = (unsigned char *)malloc(AVS_PREONNX_MAX_IN_C * AVS_PREONNX_MAX_DIM * sizeof(float));
	/*float *test = nullptr;
	for (int i = 0; i < AVS_PREONNX_MAX_IN_C * AVS_PREONNX_MAX_DIM; i++) {
		test = (float *)test_pillar_x; test[i] = 1.f;
		test = (float *)test_pillar_y; test[i] = 1.f;
		test = (float *)test_pillar_z; test[i] = 1.f;
		test = (float *)test_pillar_i; test[i] = 1.f;
		test = (float *)test_x_sub_shaped; test[i] = 1.f;
		test = (float *)test_y_sub_shaped; test[i] = 1.f;
		test = (float *)test_mask; test[i] = 1.f;
	}
	for (int i = 0; i < AVS_PREONNX_MAX_IN_C; i++) {
		test = (float *)test_num_voxels; test[i] = 1.f;
	}*/
	Mat pillar_x(4, sizes, CV_32F, test_pillar_x);
	Mat pillar_y(4, sizes, CV_32F, test_pillar_y);
	Mat pillar_z(4, sizes, CV_32F, test_pillar_z);
	Mat pillar_i(4, sizes, CV_32F, test_pillar_i);
	Mat num_voxels(2, sizes2, CV_32F, test_num_voxels);
	Mat x_sub_shaped(4, sizes, CV_32F, test_x_sub_shaped);
	Mat y_sub_shaped(4, sizes, CV_32F, test_y_sub_shaped);
	Mat mask(4, sizes, CV_32F, test_mask);


	net.setInputShape("pillar_x", { 1, 1, AVS_PREONNX_MAX_IN_C, AVS_PREONNX_MAX_DIM });
	net.setInputShape("pillar_y", { 1, 1, AVS_PREONNX_MAX_IN_C, AVS_PREONNX_MAX_DIM });
	net.setInputShape("pillar_z", { 1, 1, AVS_PREONNX_MAX_IN_C, AVS_PREONNX_MAX_DIM });
	net.setInputShape("pillar_i", { 1, 1, AVS_PREONNX_MAX_IN_C, AVS_PREONNX_MAX_DIM });
	net.setInputShape("num_points_per_pillar", { 1, AVS_PREONNX_MAX_IN_C });
	net.setInputShape("x_sub_shaped", { 1, 1, AVS_PREONNX_MAX_IN_C, AVS_PREONNX_MAX_DIM });
	net.setInputShape("y_sub_shaped", { 1, 1, AVS_PREONNX_MAX_IN_C, AVS_PREONNX_MAX_DIM });
	net.setInputShape("mask", { 1, 1, AVS_PREONNX_MAX_IN_C, AVS_PREONNX_MAX_DIM });

	net.setInput(pillar_x, "pillar_x");
	net.setInput(pillar_y, "pillar_y");
	net.setInput(pillar_z, "pillar_z");
	net.setInput(pillar_i, "pillar_i");
	net.setInput(num_voxels, "num_points_per_pillar");	// dnn的输入是否一定是nchw?
	net.setInput(x_sub_shaped, "x_sub_shaped");
	net.setInput(y_sub_shaped, "y_sub_shaped");
	net.setInput(mask, "mask");
	// Mat result = net.forward("163/inv");	// scale 出现了问题
	Mat result = net.forward("163");	// scale 出现了问题
	printf("dnn test done!\n");
}

void onnx_demo() {
	//*************************************************************************
	// initialize  enviroment...one enviroment per process
	// enviroment maintains thread pools and other state info
	Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test");

	// initialize session options if needed
	Ort::SessionOptions session_options;
	session_options.SetIntraOpNumThreads(1);

	// If onnxruntime.dll is built with CUDA enabled, we can uncomment out this line to use CUDA for this
	// session (we also need to include cuda_provider_factory.h above which defines it)
	// #include "cuda_provider_factory.h"
	// OrtSessionOptionsAppendExecutionProvider_CUDA(session_options, 1);

	// Sets graph optimization level
	// Available levels are
	// ORT_DISABLE_ALL -> To disable all optimizations
	// ORT_ENABLE_BASIC -> To enable basic optimizations (Such as redundant node removals)
	// ORT_ENABLE_EXTENDED -> To enable extended optimizations (Includes level 1 + more complex optimizations like node fusions)
	// ORT_ENABLE_ALL -> To Enable All possible opitmizations
	session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

	//*************************************************************************
	// create session and load model into memory
	// using squeezenet version 1.3
	// URL = https://github.com/onnx/models/tree/master/squeezenet
#ifdef _WIN32
	//const wchar_t* model_path = L"squeezenet.onnx";
	const wchar_t* model_path = L"squeezenet1.0-6.onnx";
	//const wchar_t* model_path = L"squeezenet1.1-7.onnx";
#else
	const char* model_path = "squeezenet.onnx";
#endif

	printf("Using Onnxruntime C++ API\n");
	Ort::Session session(env, model_path, session_options);

	//*************************************************************************
	// print model input layer (node names, types, shape etc.)
	Ort::AllocatorWithDefaultOptions allocator;

	// print number of model input nodes
	size_t num_input_nodes = session.GetInputCount();
	std::vector<const char*> input_node_names(num_input_nodes);
	std::vector<int64_t> input_node_dims;  // simplify... this model has only 1 input node {1, 3, 224, 224}.
										   // Otherwise need vector<vector<>>

	printf("Number of inputs = %zu\n", num_input_nodes);

	// iterate over all input nodes
	for (int i = 0; i < num_input_nodes; i++) {
		// print input node names
		char* input_name = session.GetInputName(i, allocator);
		printf("Input %d : name=%s\n", i, input_name);
		input_node_names[i] = input_name;

		// print input node types
		Ort::TypeInfo type_info = session.GetInputTypeInfo(i);
		auto tensor_info = type_info.GetTensorTypeAndShapeInfo();

		ONNXTensorElementDataType type = tensor_info.GetElementType();
		printf("Input %d : type=%d\n", i, type);

		// print input shapes/dims
		input_node_dims = tensor_info.GetShape();
		printf("Input %d : num_dims=%zu\n", i, input_node_dims.size());
		for (int j = 0; j < input_node_dims.size(); j++)
			printf("Input %d : dim %d=%jd\n", i, j, input_node_dims[j]);
	}

	// Results should be...
	// Number of inputs = 1
	// Input 0 : name = data_0
	// Input 0 : type = 1
	// Input 0 : num_dims = 4
	// Input 0 : dim 0 = 1
	// Input 0 : dim 1 = 3
	// Input 0 : dim 2 = 224
	// Input 0 : dim 3 = 224

	//*************************************************************************
	// Similar operations to get output node information.
	// Use OrtSessionGetOutputCount(), OrtSessionGetOutputName()
	// OrtSessionGetOutputTypeInfo() as shown above.

	//*************************************************************************
	// Score the model using sample data, and inspect values

	size_t input_tensor_size = 224 * 224 * 3;  // simplify ... using known dim values to calculate size
											   // use OrtGetTensorShapeElementCount() to get official size!

	std::vector<float> input_tensor_values(input_tensor_size);
	std::vector<const char*> output_node_names = { "softmaxout_1" };
	//std::vector<const char*> output_node_names = { "squeezenet0_flatten0_reshape0" };

	// initialize input data with values in [0.0, 1.0]
	for (unsigned int i = 0; i < input_tensor_size; i++)
		input_tensor_values[i] = (float)i / (input_tensor_size + 1);

	// create input tensor object from data values
	auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
	Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, input_tensor_values.data(), input_tensor_size, input_node_dims.data(), 4);
	assert(input_tensor.IsTensor());

	// score model & input tensor, get back output tensor
	auto output_tensors = session.Run(Ort::RunOptions{ nullptr }, input_node_names.data(), &input_tensor, 1, output_node_names.data(), 1);
	assert(output_tensors.size() == 1 && output_tensors.front().IsTensor());

	// Get pointer to output tensor float values
	float* floatarr = output_tensors.front().GetTensorMutableData<float>();
	assert(abs(floatarr[0] - 0.000045) < 1e-6);

	// score the model, and print scores for first 5 classes
	for (int i = 0; i < 5; i++)
		printf("Score for class [%d] =  %f\n", i, floatarr[i]);

	// Results should be as below...
	// Score for class[0] = 0.000045
	// Score for class[1] = 0.003846
	// Score for class[2] = 0.000125
	// Score for class[3] = 0.001180
	// Score for class[4] = 0.001317
	printf("Done!\n");
}

void onnx_test() {

	Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test");
	Ort::SessionOptions session_options;
	session_options.SetIntraOpNumThreads(1);
	session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
	const wchar_t* model_path = L"pfe_1.7.0.onnx";

	printf("Using Onnxruntime C++ API\n");
	Ort::Session session(env, model_path, session_options);

	std::vector<Ort::Value> ort_inputs;
	std::vector<int64_t> input_pillar_dims = { 1, 1, AVS_PREONNX_MAX_IN_C, AVS_PREONNX_MAX_DIM };
	size_t input_pillar_dims_size = 1 * 1 * AVS_PREONNX_MAX_IN_C * AVS_PREONNX_MAX_DIM;
	std::vector<int64_t> input_voxel_dims = { 1, AVS_PREONNX_MAX_IN_C };
	size_t input_voxel_dims_size = 1 * AVS_PREONNX_MAX_IN_C;

	std::vector<float> input_pillar_x_values(input_pillar_dims_size, 1);
	// TODO fill values
	auto pillar_x_memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
	Ort::Value pillar_x_tensor = Ort::Value::CreateTensor<float>(pillar_x_memory_info,
		input_pillar_x_values.data(), input_pillar_dims_size, input_pillar_dims.data(), 4);
	assert(pillar_x_tensor.IsTensor());

	std::vector<float> input_pillar_y_values(input_pillar_dims_size, 1);
	// TODO fill values
	auto pillar_y_memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
	Ort::Value pillar_y_tensor = Ort::Value::CreateTensor<float>(pillar_y_memory_info,
		input_pillar_y_values.data(), input_pillar_dims_size, input_pillar_dims.data(), 4);
	assert(pillar_y_tensor.IsTensor());

	std::vector<float> input_pillar_z_values(input_pillar_dims_size, 1);
	// TODO fill values
	auto pillar_z_memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
	Ort::Value pillar_z_tensor = Ort::Value::CreateTensor<float>(pillar_z_memory_info,
		input_pillar_z_values.data(), input_pillar_dims_size, input_pillar_dims.data(), 4);
	assert(pillar_z_tensor.IsTensor());

	std::vector<float> input_pillar_i_values(input_pillar_dims_size, 1);
	// TODO fill values
	auto pillar_i_memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
	Ort::Value pillar_i_tensor = Ort::Value::CreateTensor<float>(pillar_i_memory_info,
		input_pillar_i_values.data(), input_pillar_dims_size, input_pillar_dims.data(), 4);
	assert(pillar_i_tensor.IsTensor());
	std::vector<float> input_voxel_values(input_voxel_dims_size, 1);
	// TODO fill values
	auto voxel_memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
	Ort::Value num_voxels_tensor = Ort::Value::CreateTensor<float>(voxel_memory_info,
		input_voxel_values.data(), input_voxel_dims_size, input_voxel_dims.data(), 2);
	assert(num_voxels_tensor.IsTensor());

	std::vector<float> input_x_sub_shaped_values(input_pillar_dims_size, 1);
	// TODO fill values
	auto x_sub_shaped_memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
	Ort::Value x_sub_shaped_tensor = Ort::Value::CreateTensor<float>(x_sub_shaped_memory_info,
		input_x_sub_shaped_values.data(), input_pillar_dims_size, input_pillar_dims.data(), 4);
	assert(x_sub_shaped_tensor.IsTensor());

	std::vector<float> input_y_sub_shaped_values(input_pillar_dims_size, 1);
	// TODO fill values
	auto y_sub_shaped_memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
	Ort::Value y_sub_shaped_tensor = Ort::Value::CreateTensor<float>(y_sub_shaped_memory_info,
		input_y_sub_shaped_values.data(), input_pillar_dims_size, input_pillar_dims.data(), 4);
	assert(y_sub_shaped_tensor.IsTensor());

	std::vector<float> input_mask_values(input_pillar_dims_size, 1);
	// TODO fill values
	auto mask_memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
	Ort::Value mask_tensor = Ort::Value::CreateTensor<float>(mask_memory_info,
		input_mask_values.data(), input_pillar_dims_size, input_pillar_dims.data(), 4);
	assert(mask_tensor.IsTensor());

	std::vector<int64_t> output_pillar_dims = { 1, AVS_PREONNX_OUT_C, AVS_PREONNX_MAX_IN_C, 1 };
	size_t output_pillar_dims_size = 1 * AVS_PREONNX_OUT_C * AVS_PREONNX_MAX_IN_C * 1;
	std::vector<float> output_pillar_values(output_pillar_dims_size, 1);
	auto output_pillar_memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
	Ort::Value output_pillar_tensor = Ort::Value::CreateTensor<float>(output_pillar_memory_info,
		output_pillar_values.data(), output_pillar_dims_size, output_pillar_dims.data(), 4);
	assert(output_pillar_tensor.IsTensor());

	ort_inputs.push_back(std::move(pillar_x_tensor));
	ort_inputs.push_back(std::move(pillar_y_tensor));
	ort_inputs.push_back(std::move(pillar_z_tensor));
	ort_inputs.push_back(std::move(pillar_i_tensor));
	ort_inputs.push_back(std::move(num_voxels_tensor));
	ort_inputs.push_back(std::move(x_sub_shaped_tensor));
	ort_inputs.push_back(std::move(y_sub_shaped_tensor));
	ort_inputs.push_back(std::move(mask_tensor));

	std::vector<const char*> input_names = { "pillar_x", "pillar_y", "pillar_z", "pillar_i",
		"num_points_per_pillar", "x_sub_shaped", "y_sub_shaped", "mask" };
	std::vector<const char*> output_names = { "174" };

	auto ort_outputs = session.Run(Ort::RunOptions{ nullptr },
		input_names.data(), ort_inputs.data(), 8,
		output_names.data(), 1);
	assert(ort_outputs.size() == 1 && ort_outputs.front().IsTensor());

	// Get pointer to output tensor float values
	// float* floatarr = output_tensors.front().GetTensorMutableData<float>();
	// assert(abs(floatarr[0] - 0.000045) < 1e-6);

	printf("Done!\n");
}
