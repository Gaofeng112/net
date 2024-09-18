#include <iostream>
#include <fstream>
#include <sstream>
#include <regex>
#include <map>
#include <vector>
#include <string>
#include <cmath>
#include <ctime>
#include <algorithm>
#include <random>
#include <cstring>
#include <tensorflow/lite/interpreter.h>
#include <tensorflow/lite/model.h>
#include <tensorflow/lite/kernels/register.h>
#include <tensorflow/lite/tools/optimize/quanti   ze.h>
#include <tensorflow/lite/experimental/ruy/ruy.h>
#include "lcm_scheduler.h"
#include "clip_tokenizer.h"
#include "constants.h"

namespace tflite = ::tflite;
using namespace std;

// 设置随机种子，保证结果的可复现性
void set_seed(int seed) {
    std::srand(seed);
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(0.0, 1.0);
}

int main(int argc, char** argv) {
    // 解析命令行参数
    string prompt = "a dog playing with a ball.";
    string model_dir = "./fp32";
    string output_file = "output.png";

    if (argc > 1) prompt = argv[1];
    if (argc > 2) model_dir = argv[2];
    if (argc > 3) output_file = argv[3];

    set_seed(42);

    // 读取文件并解析子图顺序
    ifstream file("./subgraphs_0817.txt");
    stringstream buffer;
    buffer << file.rdbuf();
    string content = buffer.str();

    regex pattern("(\\w+)subgraph(\\d+): order(\\d+)");
    smatch matches;

    map<int, vector<string>> subgraph_order_map;
    auto content_begin = sregex_iterator(content.begin(), content.end(), pattern);
    auto content_end = sregex_iterator();

    for (sregex_iterator i = content_begin; i != content_end; ++i) {
        smatch match = *i;
        string subgraph_type = match[1].str();
        string subgraph_number = match[2].str();
        int order = stoi(match[3].str());
        transform(subgraph_type.begin(), subgraph_type.end(), subgraph_type.begin(), ::tolower);
        string file_path = "./subgraphs_0817/" + subgraph_type + "subgraph" + subgraph_number + ".tflite";
        subgraph_order_map[order].push_back(file_path);
    }

    vector<string> sorted_file_paths;
    for (const auto& pair : subgraph_order_map) {
        const vector<string>& file_paths = pair.second;
        sorted_file_paths.insert(sorted_file_paths.end(), file_paths.begin(), file_paths.end());
    }
    vector<string> model_files = sorted_file_paths;

    // 加载 TFLite 模型
    vector<std::unique_ptr<tflite::Interpreter>> interpreters;
    tflite::ops::builtin::BuiltinOpResolver resolver;
    for (const auto& model_file : model_files) {
        auto model = tflite::FlatBufferModel::BuildFromFile(model_file.c_str());
        if (!model) {
            cerr << "无法加载模型: " << model_file << endl;
            return 1;
        }
        std::unique_ptr<tflite::Interpreter> interpreter;
        tflite::InterpreterBuilder builder(*model, resolver);
        builder(&interpreter);
        if (!interpreter) {
            cerr << "无法创建模型解释器: " << model_file << endl;
            return 1;
        }
        interpreters.push_back(std::move(interpreter));
    }

    // 加载并分配额外模型的张量
    auto text_encoder_model = tflite::FlatBufferModel::BuildFromFile((model_dir + "/converted_text_encoder.tflite").c_str());
    auto diffusion_model = tflite::FlatBufferModel::BuildFromFile((model_dir + "/converted_diffusion_model.tflite").c_str());
    auto decoder_model = tflite::FlatBufferModel::BuildFromFile((model_dir + "/converted_decoder.tflite").c_str());

    if (!text_encoder_model || !diffusion_model || !decoder_model) {
        cerr << "加载一个或多个模型失败。" << endl;
        return 1;
    }

    tflite::InterpreterBuilder text_encoder_builder(*text_encoder_model, resolver);
    tflite::InterpreterBuilder diffusion_builder(*diffusion_model, resolver);
    tflite::InterpreterBuilder decoder_builder(*decoder_model, resolver);

    std::unique_ptr<tflite::Interpreter> text_encoder;
    std::unique_ptr<tflite::Interpreter> diffusion;
    std::unique_ptr<tflite::Interpreter> decoder;

    text_encoder_builder(&text_encoder);
    diffusion_builder(&diffusion);
    decoder_builder(&decoder);

    if (!text_encoder || !diffusion || !decoder) {
        cerr << "创建一个或多个解释器失败。" << endl;
        return 1;
    }

    text_encoder->AllocateTensors();
    diffusion->AllocateTensors();
    decoder->AllocateTensors();

    // 初始化调度器和分词器
    LCMScheduler scheduler(0.00085, 0.0120, "scaled_linear", "epsilon");
    scheduler.set_timesteps(4, 50);
    SimpleTokenizer tokenizer;

    // 准备输入数据
    vector<int> inputs = tokenizer.encode(prompt);
    inputs.resize(77, 49407); // 填充输入到长度 77
    vector<int> pos_ids(77);
    iota(pos_ids.begin(), pos_ids.end(), 0); // 填充 0 到 76

    // 执行推理并处理结果
    clock_t start_time = clock();
    vector<float> latent; // 初始化潜在变量
    vector<float> context; // 初始化上下文
    vector<float> unconditional_context; // 初始化无条件上下文

    // ... (进一步处理，类似于 Python 代码，包括调用推理和处理输出)

    clock_t end_time = clock();
    cout << "执行时间: " << static_cast<double>(end_time - start_time) / CLOCKS_PER_SEC * 1000 << " ms" << endl;

    // 保存输出图像
    // (这需要将张量数据转换为图像格式，类似于 Python 中的 PIL)

    return 0;
}
