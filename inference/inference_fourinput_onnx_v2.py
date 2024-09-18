import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType
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

# 读取文件
with open('subgraphs_ios.txt', 'r') as file:  # 使用正确的变量名 "file"
    content = file.read()

# 创建一个字典来存储子图的order和对应的文件路径
subgraph_order_map = {}

# 正则表达式匹配所有子图的order和名称--保存为元组，内容分别为3个捕获项
# 改进正则表达式以便更准确地匹配
matches = re.findall(r'(\w+)subgraph(\d+): order(\d+)', content)

# 构建子图的文件路径并存储到字典中
for match in matches:
    subgraph_type, subgraph_number, order = match
    # 将NPU和CPU转换为小写
    lower_subgraph_type = subgraph_type.lower()
    file_path = f"./diffusion_model_fp32_subgraphs/{lower_subgraph_type}subgraph{subgraph_number}.onnx"
    # 如果order已经存在，则将路径添加到一个列表中，否则创建一个新的列表
    if int(order) in subgraph_order_map:
        subgraph_order_map[int(order)].append(file_path)
    else:
        subgraph_order_map[int(order)] = [file_path]

# 按照order排序子图文件路径
sorted_file_paths = []
for order in sorted(subgraph_order_map.keys()):
    sorted_file_paths.extend(subgraph_order_map[order])

# 打印排序后的文件路径列表
print(sorted_file_paths)

#动态量化
# quantized_model_files = []
# for model_file in sorted_file_paths:
#     quantized_model_file = model_file.replace('.onnx', '_quant.onnx')
#     quantize_dynamic(model_file, quantized_model_file, weight_type=QuantType.QUInt8)
#     quantized_model_files.append(quantized_model_file)

# print(quantized_model_files)

sessions = [ort.InferenceSession(model) for model in quantized_model_files]

device_name = 'cpu'
providers = ['CPUExecutionProvider']

# prompt="DSLR photograph of an astronaut riding a horse"
prompt="An island on the sea"
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
for index, timestep in progbar:
    progbar.set_description(f"{index:3d} {timestep:3d}")

    initial_input_data = {
        "input_1": np.array(timestep_embedding([timestep]), dtype=np.float32),
        "input_2": np.array(get_guidance_scale_embedding(unconditional_guidance_scale, 256).astype(np.float32), dtype=np.float32),
        "input_3": np.array(latent, dtype=np.float32),
        "input_4": np.array(context, dtype=np.float32),
    }

    input_data = initial_input_data

    for i, session in enumerate(sessions):
        input_names = [inp.name for inp in session.get_inputs()]
        model_input_data = {name: input_data[name] for name in input_names}
        outputs = session.run(None, model_input_data)
        output_names = [out.name for out in session.get_outputs()]
        
        if i < len(sessions) - 1:
            for output, output_name in zip(outputs, output_names):
                input_data[output_name] = output

    e_t = outputs[0]
    e_t_hf = np.transpose(e_t, (0, 3, 1, 2))
    latent_hf = np.transpose(latent, (0, 3, 1, 2))
    # pdb.set_trace()
    output_latent = scheduler.step(e_t_hf, index, timestep, latent_hf)

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
 
# print the difference between start
# and end time in milli. secs
print("The time of execution of above program is :",
      (end-start) * 10**3, "ms")