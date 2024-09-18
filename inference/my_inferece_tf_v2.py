import numpy as np
import math
import multiprocessing
import os
import random
import tensorflow as tf
from clip_tokenizer import SimpleTokenizer
from lcm_scheduler import LCMScheduler
from PIL import Image
from tqdm import tqdm
import re

def parse_model_info(file_path):
    model_info = {}
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    model_name = None
    for line in lines:
        line = line.strip()
        if line.startswith('Model:'):
            model_name = line.split(':')[1].strip()
            model_info[model_name] = {'Inputs': {}, 'Outputs': {}}
        elif line.startswith('Inputs:'):
            current_section = 'Inputs'
        elif line.startswith('Outputs:'):
            current_section = 'Outputs'
        elif line and model_name:
            match = re.match(r'TFLite:\s*(.*) -> ONNX:\s*(.*)', line)
            if match and model_name:
                tflite_name, onnx_name = match.groups()
                model_info[model_name][current_section][tflite_name] = onnx_name

    return model_info

def print_model_info(model_info):
    for model, info in model_info.items():
        print(f"Model: {model}")
        print("Inputs:")
        for tflite, onnx in info['Inputs'].items():
            print(f"  TFLite: {tflite} -> ONNX: {onnx}")
        print("Outputs:")
        for tflite, onnx in info['Outputs'].items():
            print(f"  TFLite: {tflite} -> ONNX: {onnx}")
        print()

# 使用示例
file_path = './modif.txt'  # 替换为你的文本文件路径
model_info = parse_model_info(file_path)
print_model_info(model_info)

def get_onnx_input_name(model_info, model_name, tflite_input_name):
    if model_name not in model_info:
        raise ValueError(f"Model '{model_name}' not found in model_info.")
    
    inputs = model_info[model_name].get('Inputs', {})
    if tflite_input_name not in inputs:
        raise ValueError(f"TFLite input name '{tflite_input_name}' not found for model '{model_name}'.")
    
    onnx_name = inputs[tflite_input_name]
    return onnx_name

def get_onnx_output_name(model_info, model_name, tflite_output_name):
    if model_name not in model_info:
        raise ValueError(f"Model '{model_name}' not found in model_info.")
    
    outpus = model_info[model_name].get('Outputs', {})
    if tflite_output_name not in outpus:
        raise ValueError(f"TFLite outpus name '{tflite_output_name}' not found for model '{model_name}'.")
    
    onnx_name = outpus[tflite_output_name]
    return onnx_name

def convert_input_to_onnx_names(model_info, model_name, tflite_indices, onnx_input_name):
    for tflite_name, index in tflite_indices.items():
        try:
            onnx_name = get_onnx_input_name(model_info, model_name, tflite_name)
            onnx_input_name.append(onnx_name)
        except ValueError as e:
            print(e)  # 或者记录日志/处理异常

def convert_output_to_onnx_names(model_info, model_name, tflite_indices, onnx_input_name):
    for tflite_name, index in tflite_indices.items():
        try:
            onnx_name = get_onnx_output_name(model_info, model_name, tflite_name)
            onnx_input_name.append(onnx_name)
        except ValueError as e:
            print(e)  # 或者记录日志/处理异常

flie = open('subgraphs.txt','r')
content = flie.read()
subgraph_order_map = {}
matches = re.findall(r'(\w+)subgraph(\d+): order(\d+)', content)

for match in matches:
    subgraph_type, subgraph_number, order = match
    lower_subgraph_type = subgraph_type.lower()
    file_path = f"./onnx_lib_tflite/{lower_subgraph_type}subgraph{subgraph_number}.tflite"
    subgraph_order_map[int(order)] = file_path

sorted_file_paths = [subgraph_order_map[order] for order in sorted(subgraph_order_map.keys())]

model_files = []
for model_file in sorted_file_paths:
    model_files.append(model_file)

print(model_files)

interpreters = [tf.lite.Interpreter(model_path=model) for model in model_files]

device_name = 'cpu'
providers = ['CPUExecutionProvider']

prompt="DSLR photograph of an astronaut riding a horse"
# prompt="An island on the sea"
# prompt="Basketball court"
negative_prompt=None
batch_size=1
num_steps=4
unconditional_guidance_scale=7.5
temperature=1
seed=42
input_image=None
input_mask=None
input_image_strength=0.5
tokenizer=SimpleTokenizer()

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

set_seed(42)

path_to_saved_models="./int8"

scheduler = LCMScheduler(beta_start=0.00085, beta_end=0.0120, beta_schedule="scaled_linear", prediction_type="epsilon")
scheduler.set_timesteps(num_steps, 50)
text_encoder = tf.lite.Interpreter(model_path= os.path.join(path_to_saved_models,"converted_text_encoder.tflite"),num_threads=multiprocessing.cpu_count())
text_encoder.allocate_tensors()
# Get input and output tensors.
input_details_text_encoder = text_encoder.get_input_details()
output_details_text_encoder = text_encoder.get_output_details()


# diffusion_model = onnxruntime.InferenceSession(input_onnx_path, providers=providers)

decoder = tf.lite.Interpreter(model_path=os.path.join(path_to_saved_models,"converted_decoder.tflite"),num_threads=multiprocessing.cpu_count())
decoder.allocate_tensors()
input_details_decoder = decoder.get_input_details()
output_details_decoder = decoder.get_output_details()

import time

# record start time
start = time.time()

inputs = tokenizer.encode(prompt)
# assert len(inputs) < 77, "Prompt is too long (should be < 77 tokens)"
phrase = inputs + [49407] * (77 - len(inputs))
phrase = np.array(phrase)[None].astype("int32")
phrase = np.repeat(phrase, batch_size, axis=0)
# Encode prompt tokens (and their positions) into a "context vector"
pos_ids = np.array(list(range(77)))[None].astype("int32")
pos_ids = np.repeat(pos_ids, batch_size, axis=0)
# context = model.text_encoder.predict_on_batch([phrase, pos_ids])
# print(f"context shape {context.shape}")
text_encoder.set_tensor(input_details_text_encoder[0]['index'], phrase)
text_encoder.set_tensor(input_details_text_encoder[1]['index'], pos_ids)
text_encoder.invoke()
context = text_encoder.get_tensor(output_details_text_encoder[0]['index'])

img_height=256
img_width=256   
n_h = img_height // 8
n_w = img_width // 8
latent = tf.random.normal((batch_size, n_h, n_w, 4), seed=seed)

def timestep_embedding(timesteps, dim=320, max_period=10000):
    half = dim // 2
    freqs = np.exp(
        -math.log(max_period) * np.arange(0, half, dtype="float32") / half
    )
    args = np.array(timesteps) * freqs
    embedding = np.concatenate([np.cos(args), np.sin(args)])
    return embedding.reshape(1, -1).astype(np.float32)

def get_guidance_scale_embedding(w, embedding_dim=512):
    w = w * 1000.0

    half_dim = embedding_dim // 2
    emb = np.exp(np.arange(half_dim) * -np.log(10000.0) / (half_dim - 1))
    emb = np.expand_dims(w, axis=-1) * np.expand_dims(emb, axis=0)
    emb = np.concatenate([np.sin(emb), np.cos(emb)], axis=-1)

    return emb

guidance_scale_embedding = get_guidance_scale_embedding(unconditional_guidance_scale, 256).astype(np.float32)
   
timesteps = scheduler.timesteps
progbar = tqdm(enumerate(timesteps))
for index_, timestep in progbar:
    progbar.set_description(f"{index_:3d} {timestep:3d}")

    initial_input_data = {
        "sample": np.array(latent, dtype=np.float32),
        "timestep": np.array([float(timestep)], dtype=np.float32),
        "encoder_hidden_states": np.array(context, dtype=np.float32)
    }
    input_data = initial_input_data

    total_start_time = time.time()
    # 初始化文件以写入模型名称和时间
    with open('model_tflite_times.txt', 'w') as f:
        f.write("tflite\nModel Name, Inference Time (ms)\n")

    # for i, interpreter in enumerate(interpreters):
    for i, (interpreter, model_file) in enumerate(zip(interpreters, model_files)):
        name_without_ext =''
        match = re.search(r'/([^/]+)\.tflite$', model_file)
        if match:
            name_without_ext = match.group(1)
            print(name_without_ext)

        model_start_time = time.time()
        interpreter.allocate_tensors()
        
        # 获取输入和输出张量的详细信息
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        # 构建输入和输出索引字典
        input_indices = {tensor['name']: i for i, tensor in enumerate(input_details)}
        output_indices = {tensor['name']: i for i, tensor in enumerate(output_details)}
        
        for name in input_indices.keys():
            print("input:",name)

        onnx_input_name = []
        onnx_output_name = []

        # 使用 get_onnx_name 函数转换 TFLite 名称为 ONNX 名称
        convert_input_to_onnx_names(model_info, name_without_ext, input_indices, onnx_input_name)
        convert_output_to_onnx_names(model_info, name_without_ext, output_indices, onnx_output_name)
        print(onnx_input_name)
        print(onnx_output_name)

        # 构造输入数据  input带有conv,resize的要先调整shape,npu10子图的input
        model_input_data = {name: input_data[name] for name in onnx_input_name}

        # for name, index in input_indices.items():
        for name, (_, index) in zip(onnx_input_name, input_indices.items()):
            # print(model_input_data[name])
            print("------------------------------")
            print(input_details[index])
            shape1 = input_details[index]['shape']
            shape2 = model_input_data[name].shape
            print(shape1)
            print(shape2)

            def convert_to_nhwc(data):
                """将 NCHW 格式转换为 NHWC 格式"""
                return np.transpose(data, (0, 2, 3, 1))

            def convert_to_nchw(data):
                """将 NHWC 格式转换为 NCHW 格式"""
                return np.transpose(data, (0, 3, 1, 2))

            def is_nchw(shape):
                """判断给定的形状是否为 NCHW 格式"""
                return len(shape) == 4 and (shape[2] == shape[3])

            def is_nhwc(shape):
                """判断给定的形状是否为 NHWC 格式"""
                return len(shape) == 4 and (shape[1] == shape[2])

            def determine_format(shape):
                """根据给定的形状确定格式"""
                if len(shape) == 4:
                    if (shape[2] == shape[3]):
                        return 'NCHW'
                    elif (shape[1] == shape[2]):
                        return 'NHWC'
                return None

            # 根据 input_details 的形状确定目标格式
            target_format = determine_format(shape1)
            print(f"Target format for {name}: {target_format}")
            print("shape2 :is_nchw", is_nchw(shape2))
            print("shape2 :is_nhwc", is_nhwc(shape2))

            # 比较形状是否相等
            if tuple(shape1) == tuple(shape2):
                model_input_data_convert = model_input_data[name]
                print("The shapes are equal.")
            else:
                # 判断当前数据的格式并转换
                if target_format == 'NHWC' and is_nchw(shape2):
                    model_input_data_convert = convert_to_nhwc(model_input_data[name])
                elif target_format == 'NCHW' and is_nhwc(shape2):
                    model_input_data_convert = convert_to_nchw(model_input_data[name])
                print("The shapes are not equal.")
                print(model_input_data_convert.shape)

            interpreter.set_tensor(input_details[index]['index'], model_input_data_convert)

        # 运行推理
        interpreter.invoke()

        model_end_time = time.time()
        model_inference_time = (model_end_time - model_start_time) * 1000  # 转换为毫秒
        # 将模型名称和运行时间写入文件
        with open('model_tflite_times.txt', 'a') as f:
            f.write(f"{os.path.basename(model_file)}, {model_inference_time:.2f}\n")

        # 获取输出数据
        outputs = {name: interpreter.get_tensor(output_details[index]['index']) for name, index in output_indices.items()}
        print(output_details)
        # print(outputs)
        # print(onnx_output_name)

        # 保存输出数据作为下一个模型的输入
        if i < len(interpreters) - 1:
            # for name, output in outputs.items():
            for k, (name, output) in enumerate(outputs.items()):
                print(name)
                print(onnx_output_name[k])
                input_data[onnx_output_name[k]] = output

    total_end_time = time.time()
    total_time = (total_end_time - total_start_time) * 1000
    print(f"Total inference time: {total_time:.2f} ms")
    with open('model_tflite_times.txt', 'a') as f:
        f.write(f"Total, {total_time:.2f}\n")

    for name, output in outputs.items():
        e_t = output
    print(e_t)
    e_t_hf = np.transpose(e_t, (0, 3, 1, 2))
    latent_hf = np.transpose(latent, (0, 3, 1, 2))
    output_latent = scheduler.step(e_t_hf, index_, timestep, latent_hf)
    latent = np.transpose(output_latent[0], (0, 2, 3, 1))

denoised = output_latent[1]
denoised = np.transpose(denoised, (0, 2, 3, 1))
decoder.set_tensor(input_details_decoder[0]['index'], denoised)
decoder.invoke()
decoded = decoder.get_tensor(output_details_decoder[0]['index'])
# decoded = model.decoder.predict_on_batch(latent)
decoded = ((decoded + 1) / 2) * 255
img=np.clip(decoded, 0, 255).astype("uint8")
image = Image.fromarray(img[0])
image.save("test.png")
end = time.time()

print("The time of execution of above program is :",
      (end-start) * 10**3, "ms")
