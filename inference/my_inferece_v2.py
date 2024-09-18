import onnxruntime as ort
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

model_files = ["./onnx_lib/npusubgraph0.onnx", "./onnx_lib/cpusubgraph1.onnx", "./onnx_lib/npusubgraph1.onnx", "./onnx_lib/cpusubgraph0.onnx",
               "./onnx_lib/npusubgraph2.onnx", "./onnx_lib/cpusubgraph2.onnx", "./onnx_lib/npusubgraph3.onnx", "./onnx_lib/cpusubgraph3.onnx",
               "./onnx_lib/npusubgraph4.onnx", "./onnx_lib/cpusubgraph4.onnx", "./onnx_lib/npusubgraph5.onnx", "./onnx_lib/cpusubgraph5.onnx",
               "./onnx_lib/npusubgraph6.onnx", "./onnx_lib/cpusubgraph6.onnx", "./onnx_lib/npusubgraph7.onnx", "./onnx_lib/cpusubgraph7.onnx",
               "./onnx_lib/npusubgraph8.onnx", "./onnx_lib/cpusubgraph8.onnx", "./onnx_lib/npusubgraph9.onnx", "./onnx_lib/cpusubgraph9.onnx",
               "./onnx_lib/npusubgraph10.onnx", "./onnx_lib/cpusubgraph10.onnx", "./onnx_lib/npusubgraph11.onnx", "./onnx_lib/cpusubgraph11.onnx",
               "./onnx_lib/npusubgraph12.onnx", "./onnx_lib/cpusubgraph12.onnx", "./onnx_lib/npusubgraph13.onnx", "./onnx_lib/cpusubgraph13.onnx",
               "./onnx_lib/npusubgraph14.onnx", "./onnx_lib/cpusubgraph14.onnx", "./onnx_lib/npusubgraph15.onnx", "./onnx_lib/cpusubgraph15.onnx",
               "./onnx_lib/npusubgraph16.onnx", "./onnx_lib/cpusubgraph16.onnx", "./onnx_lib/npusubgraph17.onnx", "./onnx_lib/cpusubgraph17.onnx",
               "./onnx_lib/npusubgraph18.onnx", "./onnx_lib/cpusubgraph18.onnx", "./onnx_lib/npusubgraph19.onnx", "./onnx_lib/cpusubgraph19.onnx", 
               "./onnx_lib/npusubgraph20.onnx", "./onnx_lib/cpusubgraph20.onnx", "./onnx_lib/npusubgraph21.onnx", "./onnx_lib/cpusubgraph21.onnx", 
               "./onnx_lib/npusubgraph22.onnx", "./onnx_lib/cpusubgraph22.onnx", "./onnx_lib/npusubgraph23.onnx", "./onnx_lib/cpusubgraph23.onnx", 
               "./onnx_lib/npusubgraph24.onnx", "./onnx_lib/cpusubgraph24.onnx", "./onnx_lib/npusubgraph25.onnx", "./onnx_lib/cpusubgraph25.onnx", 
               "./onnx_lib/npusubgraph26.onnx", "./onnx_lib/cpusubgraph26.onnx", "./onnx_lib/npusubgraph27.onnx", "./onnx_lib/cpusubgraph27.onnx", 
               "./onnx_lib/npusubgraph28.onnx", "./onnx_lib/cpusubgraph28.onnx", "./onnx_lib/npusubgraph29.onnx", "./onnx_lib/cpusubgraph29.onnx", 
               "./onnx_lib/npusubgraph30.onnx", "./onnx_lib/cpusubgraph30.onnx", "./onnx_lib/npusubgraph31.onnx", "./onnx_lib/cpusubgraph31.onnx", 
               "./onnx_lib/npusubgraph32.onnx", "./onnx_lib/cpusubgraph32.onnx", "./onnx_lib/npusubgraph33.onnx", "./onnx_lib/cpusubgraph33.onnx", 
               "./onnx_lib/npusubgraph34.onnx", "./onnx_lib/cpusubgraph34.onnx", "./onnx_lib/npusubgraph35.onnx", "./onnx_lib/cpusubgraph35.onnx", 
               "./onnx_lib/npusubgraph36.onnx", "./onnx_lib/cpusubgraph36.onnx", "./onnx_lib/npusubgraph37.onnx", "./onnx_lib/cpusubgraph37.onnx", 
               "./onnx_lib/npusubgraph38.onnx", "./onnx_lib/cpusubgraph38.onnx", "./onnx_lib/npusubgraph39.onnx", "./onnx_lib/cpusubgraph39.onnx", 
               "./onnx_lib/npusubgraph40.onnx", "./onnx_lib/cpusubgraph40.onnx"]

# 加载模型并创建会话
sessions = [ort.InferenceSession(model) for model in model_files]

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
        "encoder_hidden_states": np.array(context, dtype=np.float32)
    }
    # 设置初始输入数据
    input_data = initial_input_data

    for i, session in enumerate(sessions):
        # 获取当前模型的所有输入名称
        input_names = [inp.name for inp in session.get_inputs()]
        # 构造输入字典
        model_input_data = {name: input_data[name] for name in input_names}
        # 运行推理
        outputs = session.run(None, model_input_data)
        # 获取当前模型的所有输出名称
        output_names = [out.name for out in session.get_outputs()]
        # 保存输出数据作为下一个模型的输入
        if i < len(sessions) - 1:
            for output, output_name in zip(outputs, output_names):
                input_data[output_name] = output

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
image.save("test_V2.png")

