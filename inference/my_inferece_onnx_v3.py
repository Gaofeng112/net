import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType
import numpy as np
import math
import multiprocessing
import os
import random
import numpy as np
import tensorflow as tf
from clip_tokenizer import SimpleTokenizer
from lcm_scheduler import LCMScheduler
from PIL import Image
from tqdm import tqdm
import re
import time
import pdb
# 读取文件
# with open('subgraphs.txt', 'r') as file:
with open('./65subgraph/instr_modified.txt', 'r') as file:
    content = file.read()
subgraph_order_map = {}
matches = re.findall(r'(\w+)subgraph(\d+): order(\d+)', content)

for match in matches:
    subgraph_type, subgraph_number, order = match
    lower_subgraph_type = subgraph_type.lower()
    file_path = f"./65subgraph/{lower_subgraph_type}subgraph{subgraph_number}.onnx"
    if int(order) in subgraph_order_map:
        subgraph_order_map[int(order)].append(file_path)
    else:
        subgraph_order_map[int(order)] = [file_path]
sorted_file_paths = []
for order in sorted(subgraph_order_map.keys()):
    sorted_file_paths.extend(subgraph_order_map[order])
model_files = sorted_file_paths

# pdb.set_trace()
# 动态量化带有 'npu' 的 ONNX 模型
# quantized_model_files = []
# for model_file in model_files:
#     if 'npu' in model_file:
#         quantized_model_file = model_file.replace('.onnx', '_quant.onnx')
#         quantize_dynamic(model_file, quantized_model_file, weight_type=QuantType.QUInt8)
#         quantized_model_files.append(quantized_model_file)
#     else:
#         quantized_model_files.append(model_file)

sessions = [ort.InferenceSession(model) for model in model_files]

device_name = 'cpu'
providers = ['CPUExecutionProvider']

# prompt="an astronaut riding a horse"
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

path_to_saved_models="./fp32"

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
    # 预先生成输入数据
    initial_input_data = {
        "sample": np.array(latent, dtype=np.float32).transpose(0, 3, 1, 2),
        "timestep": np.array([float(timestep)], dtype=np.float32),
        "encoder_hidden_states": np.array(context, dtype=np.float32),
        # "/down_blocks.0/attentions.0/Add_output_0": None
    }
    # for key, value in initial_input_data.items():
    #      print(f"{key}: {value.shape}")
    # pdb.set_trace()
    # 设置初始输入数据
    input_data = initial_input_data
    
    total_start_time = time.time()
    # 初始化文件以写入模型名称和时间
    with open('model_onnx_times.txt', 'w') as f:
        f.write("onnx\nModel Name, Inference Time (ms)\n")

    for i, (session,model_flie) in enumerate(zip(sessions,model_files)):
        model_start_time = time.time()
        input_names = [inp.name for inp in session.get_inputs()]
        model_input_data = {name: input_data[name] for name in input_names}
        outputs = session.run(None, model_input_data)

        model_end_time = time.time()
        model_inference_time = (model_end_time - model_start_time) * 1000  # 转换为毫秒
        with open('model_onnx_times.txt', 'a') as f:
            f.write(f"{os.path.basename(model_flie)}, {model_inference_time:.2f}\n")

        output_names = [out.name for out in session.get_outputs()]

        if i < len(sessions) - 1:
            for output, output_name in zip(outputs, output_names):
                input_data[output_name] = output

    total_end_time = time.time()
    total_time = (total_end_time - total_start_time) * 1000
    print(f"Total inference time: {total_time:.2f} ms")
    with open('model_onnx_times.txt', 'a') as f:
            f.write(f"Total, {total_time:.2f}\n")

    # final_output = {name: output for name, output in zip(output_names, outputs)}
    e_t_hf = outputs[0]
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
image.save("test_V5.png")
