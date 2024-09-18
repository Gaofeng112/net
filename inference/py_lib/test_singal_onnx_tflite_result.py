import onnxruntime as ort
import numpy as np
import tensorflow as tf
import pdb
# ONNX 模型推理
def run_onnx_model(model_path, input_shape=(1,4,32,32)):
    # 加载 ONNX 模型
    session = ort.InferenceSession(model_path)

    # 创建输入张量
    input_data = np.random.rand(*input_shape).astype(np.float32)

    # 获取模型的输入名称
    input_name = session.get_inputs()[0].name

    # 运行推理
    outputs = session.run(None, {input_name: input_data})

    # print("ONNX Output:\n",outputs)
    print("Output shape:", outputs[0].shape)
    return outputs

# TFLite 模型推理
def run_tflite_model(model_path, input_shape=(1024, 320)):
    # 加载 TFLite 模型
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    # 获取输入输出的详细信息
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # 创建输入张量
    input_data = np.random.rand(*input_details[0]['shape']).astype(input_details[0]['dtype'])

    # 设置模型输入
    interpreter.set_tensor(input_details[0]['index'], input_data)

    # 运行推理
    interpreter.invoke()

    # 获取输出结果
    output_data = interpreter.get_tensor(output_details[0]['index'])
    pdb.set_trace()
    print("TFLite Output:\n",output_data)
    print("Output shape:", output_data.shape)
    return output_data

# # 定义 ONNX 模型的路径
# model_file_onnx = "./npusubgraph0_v1.onnx"

# # 运行 ONNX 模型
# run_onnx_model(model_file_onnx)

# 定义 TFLite 模型的路径
model_file_tflite = "./diffusion_model_cpu_89/cpu89_0_matmul.tflite"

# 运行 TFLite 模型
run_tflite_model(model_file_tflite)