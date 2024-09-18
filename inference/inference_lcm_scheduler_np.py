# from numba import jit, cuda
# import tflite_runtime.interpreter as tflite
import argparse
import math
import multiprocessing
import os
import random
import sys
import onnx
import onnxruntime
import pdb
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from clip_tokenizer import SimpleTokenizer
from constants import _ALPHAS_CUMPROD, _UNCONDITIONAL_TOKENS, PYTORCH_CKPT_MAPPING
from lcm_scheduler import LCMScheduler
from PIL import Image
from tensorflow.keras.utils import img_to_array, load_img, normalize
from tqdm import tqdm
import onnxruntime as ort


device_name = 'cpu' # or 'cpu'
input_onnx_path = "./diffusion_model_cpu_89/cpu89_0_matmul.onnx"
providers = ['CPUExecutionProvider']
# if device_name == 'cpu':
#     providers = ['CPUExecutionProvider']
# elif device_name == 'cuda:0':
#     providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']



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

def connect_onnx():
    # 指定包含所有CPU和NPU子图的文件夹路径
    cpu_folder_path = './onnx_lib/cpu_subgraphs'
    npus_folder_path = './onnx_lib/npu_subgraphs'

    # 创建字典来存储所有CPU和NPU子模型的InferenceSession实例
    cpu_subgraph_models = {}
    npus_subgraph_models = {}
    cpu_subgraph_info = {}  # 存储CPU子模型的输入输出信息
    npus_subgraph_info = {} # 存储NPU子模型的输入输出信息

    # 加载CPU子图
    for filename in os.listdir(cpu_folder_path):
        if filename.endswith('.onnx'):
            model_path = os.path.join(cpu_folder_path, filename)
            session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
            cpu_subgraph_models[filename] = session
            # 获取输入输出信息
            input_names = [i.name for i in session.get_inputs()]
            output_names = [o.name for o in session.get_outputs()]
            cpu_subgraph_info[filename] = {'input_names': input_names, 'output_names': output_names}

    # 加载NPU子图，假设NPU Execution Provider已经配置好
    for filename in os.listdir(npus_folder_path):
        if filename.endswith('.onnx'):
            model_path = os.path.join(npus_folder_path, filename)
            session = ort.InferenceSession(model_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
            npus_subgraph_models[filename] = session
            # 获取输入输出信息
            input_names = [i.name for i in session.get_inputs()]
            output_names = [o.name for o in session.get_outputs()]
            npus_subgraph_info[filename] = {'input_names': input_names, 'output_names': output_names}

    
def set_seed(seed=42):
    # Set Python random seed
    random.seed(seed)
    
    # Set NumPy random seed
    np.random.seed(seed)
    
    # Set PyTorch random seed
    # torch.manual_seed(seed)
    # if torch.cuda.is_available():
    #     torch.cuda.manual_seed(seed)
    #     torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    
    # Set TensorFlow random seed
    tf.random.set_seed(seed)
    
    # Ensure reproducibility in certain environments
    os.environ['PYTHONHASHSEED'] = str(seed)

# Call the function at the beginning of your script
set_seed(42)

# Load TFLite model and allocate tensors.
path_to_saved_models="./fp32"

scheduler = LCMScheduler(beta_start=0.00085, beta_end=0.0120, beta_schedule="scaled_linear", prediction_type="epsilon")
scheduler.set_timesteps(num_steps, 50)
text_encoder = tf.lite.Interpreter(model_path= os.path.join(path_to_saved_models,"converted_text_encoder.tflite"),num_threads=multiprocessing.cpu_count())
text_encoder.allocate_tensors()
# Get input and output tensors.
input_details_text_encoder = text_encoder.get_input_details()
output_details_text_encoder = text_encoder.get_output_details()


diffusion_model = onnxruntime.InferenceSession(input_onnx_path, providers=providers)

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


context.tofile('context.bin')
import pdb
# pdb.set_trace()
img_height=256
img_width=256   
n_h = img_height // 8
n_w = img_width // 8
latent = tf.random.normal((batch_size, n_h, n_w, 4), seed=seed)
# latent = np.random.normal(size=(batch_size, n_h, n_w, 4)).astype(np.float32)

def timestep_embedding(timesteps, dim=320, max_period=10000):
    half = dim // 2
    freqs = np.exp(
        -math.log(max_period) * np.arange(0, half, dtype="float32") / half
    )
    args = np.array(timesteps) * freqs
    embedding = np.concatenate([np.cos(args), np.sin(args)])
    return embedding.reshape(1, -1).astype(np.float32)

import numpy as np


def get_guidance_scale_embedding(w, embedding_dim=512):
    w = w * 1000.0

    half_dim = embedding_dim // 2
    # emb = np.log(10000.0) / (half_dim - 1)
    emb = np.exp(np.arange(half_dim) * -np.log(10000.0) / (half_dim - 1))
    emb = np.expand_dims(w, axis=-1) * np.expand_dims(emb, axis=0)
    emb = np.concatenate([np.sin(emb), np.cos(emb)], axis=-1)
    
    # if embedding_dim % 2 == 1:  # zero pad
    #     emb = np.pad(emb, ((0, 0), (0, 1)), mode='constant')
    
    return emb

guidance_scale_embedding = get_guidance_scale_embedding(unconditional_guidance_scale, 256).astype(np.float32)
   
timesteps = scheduler.timesteps
progbar = tqdm(enumerate(timesteps))
for index, timestep in progbar:
    progbar.set_description(f"{index:3d} {timestep:3d}")

    # pdb.set_trace()

    onnx_input = {diffusion_model.get_inputs()[0].name:np.array(latent, dtype=np.float32).transpose(0, 3, 1, 2),
                  diffusion_model.get_inputs()[1].name:np.array([float(timestep)], dtype=np.float32),
                  diffusion_model.get_inputs()[2].name:np.array(context, dtype=np.float32)}

    e_t = diffusion_model.run(None, onnx_input)
    e_t_hf = e_t[0]
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