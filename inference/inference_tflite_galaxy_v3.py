# from numba import jit, cuda
# import tflite_runtime.interpreter as tflite
import argparse
import sys
# sys.path.append("/home/kris/pengzhiyu/bk-sdm-tf/aigc_model/model")
import math
import multiprocessing
import os
import random
from PIL import Image
# import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
# import torch
from lcm_scheduler import LCMScheduler
from clip_tokenizer import SimpleTokenizer
from constants import (
    _ALPHAS_CUMPROD,
    _UNCONDITIONAL_TOKENS,
    PYTORCH_CKPT_MAPPING,
)
from tensorflow.keras.utils import img_to_array, load_img, normalize
from tqdm import tqdm
import re
import pdb

# 读取文件
with open('./subgraphs_tflite_0816.txt', 'r') as file:
    content = file.read()
subgraph_order_map = {}
matches = re.findall(r'(\w+)subgraph(\d+): order(\d+)', content)


for match in matches:
    subgraph_type, subgraph_number, order = match
    lower_subgraph_type = subgraph_type.lower()
    file_path = f"./subgraphs_0816/{lower_subgraph_type}subgraph{subgraph_number}.tflite"
    if int(order) in subgraph_order_map:
        subgraph_order_map[int(order)].append(file_path)
    else:
        subgraph_order_map[int(order)] = [file_path]

# sorted_file_paths = [subgraph_order_map[order] for order in sorted(subgraph_order_map.keys())]

# model_files=[]
# for order_model_file in sorted_file_paths:
#     for model_file in order_model_file:
#         model_files.append(model_file)

# 按照order排序子图文件路径
sorted_file_paths = []
for order in sorted(subgraph_order_map.keys()):
    sorted_file_paths.extend(subgraph_order_map[order])
    print(f"order{order}:{subgraph_order_map[order]}")
# 打印排序后的文件路径
for i, path in enumerate(sorted_file_paths):
    print(f"order{i}:{path}")
model_files = sorted_file_paths
sessions = [tf.lite.Interpreter(model_path=model,num_threads=multiprocessing.cpu_count()) for model in model_files]
# 分配张量
# for session in sessions:
#     session.allocate_tensors()

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

# Call the function at the beginning of your script
set_seed(42)

parser = argparse.ArgumentParser(description='Process some integers.')
# parser.add_argument('-p', '--prompt', type=str, default="DSLR photograph of an astronaut riding a horse", help="Prompt for the model")
parser.add_argument('-p', '--prompt', type=str, default="A plate of donuts with a person in the background.", help="Prompt for the model")
parser.add_argument('-m', '--model-dir', type=str, default="./int8", help="Path to the model directory")
parser.add_argument('-o', '--output', type=str, default="output.png", help="Output file name")
args = parser.parse_args()

# Load TFLite model and allocate tensors.
path_to_saved_models=args.model_dir

# path_to_saved_models="./int8"

scheduler = LCMScheduler(beta_start=0.00085, beta_end=0.0120, beta_schedule="scaled_linear", prediction_type="epsilon")
scheduler.set_timesteps(4, 50)
text_encoder = tf.lite.Interpreter(model_path= os.path.join(path_to_saved_models,"converted_text_encoder.tflite"),num_threads=multiprocessing.cpu_count())
text_encoder.allocate_tensors()
# Get input and output tensors.
input_details_text_encoder = text_encoder.get_input_details()
output_details_text_encoder = text_encoder.get_output_details()

decoder = tf.lite.Interpreter(model_path=os.path.join(path_to_saved_models,"converted_decoder.tflite"),num_threads=multiprocessing.cpu_count())
decoder.allocate_tensors()
input_details_decoder = decoder.get_input_details()
output_details_decoder = decoder.get_output_details()

import time

# record start time
start = time.time()
# prompt = 'DSLR photograph of an astronaut riding a horse' 
prompt = args.prompt
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

inputs = tokenizer.encode(prompt)
# assert len(inputs) < 77, "Prompt is too long (should be < 77 tokens)"
phrase = inputs + [49407] * (77 - len(inputs))
phrase = np.array(phrase)[None].astype("int32")
phrase = np.repeat(phrase, batch_size, axis=0)
# Encode prompt tokens (and their positions) into a "context vector"
pos_ids = np.array(list(range(77)))[None].astype("int32")
pos_ids = np.repeat(pos_ids, batch_size, axis=0)
# print(f"pos_ids,phrase shape {pos_ids.shape},{phrase.shape}")
# context = model.text_encoder.predict_on_batch([phrase, pos_ids])
# print(f"context shape {context.shape}")
text_encoder.set_tensor(input_details_text_encoder[0]['index'], phrase)
text_encoder.set_tensor(input_details_text_encoder[1]['index'], pos_ids)
text_encoder.invoke()
context = text_encoder.get_tensor(output_details_text_encoder[0]['index'])
print(context.shape)
        
input_image_tensor = None

unconditional_tokens=_UNCONDITIONAL_TOKENS
unconditional_tokens = np.array(unconditional_tokens)[None].astype("int32")
unconditional_tokens = np.repeat(unconditional_tokens, batch_size, axis=0)
# print(f"unconditional tokens,pos_ids shape {unconditional_tokens.shape},{pos_ids.shape}")
# unconditional_context = model.text_encoder.predict_on_batch(
#     [unconditional_tokens, pos_ids]
# )
# print(f"unconditional context shape {unconditional_context.shape}")
text_encoder.set_tensor(input_details_text_encoder[0]['index'], unconditional_tokens)
text_encoder.set_tensor(input_details_text_encoder[1]['index'], pos_ids)
text_encoder.invoke()
unconditional_context = text_encoder.get_tensor(output_details_text_encoder[0]['index'])
# print(f"unconditional context shape {unconditional_context.shape}")
timesteps = np.arange(1, 1000, 1000 // num_steps)
input_img_noise_t = timesteps[ int(len(timesteps)*input_image_strength) ]
img_height=256
img_width=256   
def get_starting_parameters(timesteps, batch_size, seed,  input_image=None, input_img_noise_t=None):
    n_h = img_height // 8
    n_w = img_width // 8
    alphas = [_ALPHAS_CUMPROD[t] for t in timesteps]
    alphas_prev = [1.0] + alphas[:-1]
    if input_image is None:
        latent = tf.random.normal((batch_size, n_h, n_w, 4), seed=seed)
    else:
        latent = encoder(input_image)
        latent = tf.repeat(latent , batch_size , axis=0)
        latent = add_noise(latent, input_img_noise_t)
    return latent, alphas, alphas_prev
latent, alphas, alphas_prev = get_starting_parameters(
            timesteps, batch_size, seed , input_image=input_image_tensor, input_img_noise_t=input_img_noise_t
        )

# print(input_details_diffusion)
# print(output_details_diffusion)
tf.keras.mixed_precision.global_policy().name== 'mixed_float16'
dtype = tf.float32
if tf.keras.mixed_precision.global_policy().name == 'mixed_float16':
    dtype = tf.float16
def timestep_embedding(timesteps, dim=320, max_period=10000):
    half = dim // 2
    freqs = np.exp(
        -math.log(max_period) * np.arange(0, half, dtype="float32") / half
    )
    args = np.array(timesteps) * freqs
    embedding = np.concatenate([np.cos(args), np.sin(args)])
    return tf.convert_to_tensor(embedding.reshape(1, -1),dtype=dtype)

def get_guidance_scale_embedding(w, embedding_dim=512):
    w = w * 1000.0

    half_dim = embedding_dim // 2
    emb = np.log(10000.0) / (half_dim - 1)
    emb = np.exp(np.arange(half_dim) * -emb)
    emb = np.expand_dims(w, axis=-1) * np.expand_dims(emb, axis=0)
    emb = np.concatenate([np.sin(emb), np.cos(emb)], axis=-1)
    
    if embedding_dim % 2 == 1:  # zero pad
        emb = np.pad(emb, ((0, 0), (0, 1)), mode='constant')
    
    return emb
   
def get_x_prev_and_pred_x0(x, e_t, index, a_t, a_prev, temperature, seed):
    sigma_t = 0
    sqrt_one_minus_at = math.sqrt(1 - a_t)
    pred_x0 = (x - sqrt_one_minus_at * e_t) / math.sqrt(a_t)
    # Direction pointing to x_t
    dir_xt = math.sqrt(1.0 - a_prev - sigma_t**2) * e_t
    noise = sigma_t * tf.random.normal(x.shape, seed=seed) * temperature
    x_prev = math.sqrt(a_prev) * pred_x0 + dir_xt
    return x_prev, pred_x0



i_inference_time = 0
timesteps = scheduler.timesteps
progbar = tqdm(enumerate(timesteps))
print(time.time())

for index, timestep in progbar:
    progbar.set_description(f"{index:3d} {timestep:3d}")
    initial_input_data = {
        "serving_default_input_1:0": np.array(latent, dtype=np.float32),
        "serving_default_input_2:0": np.array(timestep_embedding([timestep]), dtype=np.float32),
        "serving_default_input_3:0": np.array(context, dtype=np.float32), 
    }
    input_data = initial_input_data
    i_start_time = time.time()
    
    for i, (session, model_file) in enumerate(zip(sessions, model_files)):
        session.allocate_tensors()
        input_details = session.get_input_details()
        output_details = session.get_output_details()
        # pdb.set_trace()
        # 构建输入和输出索引字典
        input_indices = {tensor['name']: i for i, tensor in enumerate(input_details)}
        output_indices = {tensor['name']: i for i, tensor in enumerate(output_details)}
        # for detail in input_details:
        #     print(f"Name: {detail['name']}, Index: {detail['index']}, Shape: {detail['shape']}")
        model_input_data = {name: input_data[name] for name in input_indices}
        for name, idx in input_indices.items():
            # pdb.set_trace()
            session.set_tensor(input_details[idx]['index'], model_input_data[name])

        i_start_time = time.time()
        session.invoke()
        i_end_time = time.time()

        # 获取输出
        outputs = {name: session.get_tensor(output_details[index]['index']) for name, index in output_indices.items()}
        # print(output_details)

        if i < len(sessions) - 1:
            for name, output in outputs.items():
                input_data[name] = output
                # pdb.set_trace()
    
    for name, output in outputs.items():
        e_t = output
    e_t_hf = np.transpose(e_t, (0, 3, 1, 2))
    latent_hf = np.transpose(latent, (0, 3, 1, 2))
    output_latent = scheduler.step(e_t_hf, index, timestep, latent_hf)
    latent = np.transpose(output_latent[0], (0, 2, 3, 1))
# pdb.set_trace()
denoised = output_latent[1]
denoised_image = np.transpose(denoised, (0, 2, 3, 1))
decoder.set_tensor(input_details_decoder[0]['index'], denoised_image)
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
