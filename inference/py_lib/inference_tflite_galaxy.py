# from numba import jit, cuda
# import tflite_runtime.interpreter as tflite
import argparse
import sys
import pdb
# sys.path.append("/home/kris/pengzhiyu/bk-sdm-tf/aigc_model/model")
import math
import multiprocessing
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import torch
from lcm_scheduler import LCMScheduler
from clip_tokenizer import SimpleTokenizer
from constants import (
    _ALPHAS_CUMPROD,
    _UNCONDITIONAL_TOKENS,
    PYTORCH_CKPT_MAPPING,
)
from tensorflow.keras.utils import img_to_array, load_img, normalize
from tqdm import tqdm


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

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('-p', '--prompt', type=str, default="An island on the sea.", help="Prompt for the model")
parser.add_argument('-m', '--model-dir', type=str, default="./fp32", help="Path to the model directory")
parser.add_argument('-o', '--output', type=str, default="output.png", help="Output file name")
args = parser.parse_args()

# Load TFLite model and allocate tensors.
path_to_saved_models=args.model_dir

scheduler = LCMScheduler(beta_start=0.00085, beta_end=0.0120, beta_schedule="scaled_linear", prediction_type="epsilon")
scheduler.set_timesteps(4, 50)
text_encoder = tf.lite.Interpreter(model_path= os.path.join(path_to_saved_models,"converted_text_encoder.tflite"),num_threads=multiprocessing.cpu_count())
text_encoder.allocate_tensors()
# Get input and output tensors.
input_details_text_encoder = text_encoder.get_input_details()
output_details_text_encoder = text_encoder.get_output_details()
diffusion_model = tf.lite.Interpreter(model_path=os.path.join(path_to_saved_models,"converted_diffusion_model.tflite"),num_threads=multiprocessing.cpu_count())
diffusion_model.allocate_tensors()
input_details_diffusion = diffusion_model.get_input_details()
output_details_diffusion =diffusion_model.get_output_details()
decoder = tf.lite.Interpreter(model_path=os.path.join(path_to_saved_models,"converted_decoder.tflite"),num_threads=multiprocessing.cpu_count())
decoder.allocate_tensors()
input_details_decoder = decoder.get_input_details()
output_details_decoder = decoder.get_output_details()

import time

# record start time
start = time.time()
prompt=args.prompt
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
print(f"pos_ids,phrase shape {pos_ids.shape},{phrase.shape}")
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
print(f"unconditional tokens,pos_ids shape {unconditional_tokens.shape},{pos_ids.shape}")
# unconditional_context = model.text_encoder.predict_on_batch(
#     [unconditional_tokens, pos_ids]
# )
# print(f"unconditional context shape {unconditional_context.shape}")
text_encoder.set_tensor(input_details_text_encoder[0]['index'], unconditional_tokens)
text_encoder.set_tensor(input_details_text_encoder[1]['index'], pos_ids)
text_encoder.invoke()
unconditional_context = text_encoder.get_tensor(output_details_text_encoder[0]['index'])
print(f"unconditional context shape {unconditional_context.shape}")
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

print(input_details_diffusion)
print(output_details_diffusion)
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

import numpy as np


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
   
print(f"Latent shape {latent.shape}")
def get_model_output(
    latent,
    t,
    context,
    unconditional_context,
    unconditional_guidance_scale,
    batch_size):
    timesteps = np.array([t])
    t_emb = timestep_embedding(timesteps)
    t_emb = np.repeat(t_emb, batch_size, axis=0)
    t_cond = get_guidance_scale_embedding(unconditional_guidance_scale-1, 256)
    t_cond = t_cond.astype(np.float32)
    # unconditional_latent = model.diffusion_model.predict_on_batch(
    #     [latent, t_emb, unconditional_context]
    # )
    # print(unconditional_latent.shape)
    # diffusion_model.set_tensor(input_details_diffusion[0]['index'], latent)
    # diffusion_model.set_tensor(input_details_diffusion[1]['index'], t_emb)
    # diffusion_model.set_tensor(input_details_diffusion[2]['index'], unconditional_context)
    # diffusion_model.set_tensor(input_details_diffusion[3]['index'], t_cond)
    # diffusion_model.invoke()
    unconditional_latent = diffusion_model.get_tensor(output_details_diffusion[0]['index'])
    # print(unconditional_latent.shape)
    # latent = model.diffusion_model.predict_on_batch([latent, t_emb, context])
    # print(latent.shape)
    diffusion_model.set_tensor(input_details_diffusion[0]['index'], latent)
    diffusion_model.set_tensor(input_details_diffusion[1]['index'], t_emb)
    diffusion_model.set_tensor(input_details_diffusion[2]['index'], context)
    diffusion_model.set_tensor(input_details_diffusion[3]['index'], t_cond)
    diffusion_model.invoke()
    latent = diffusion_model.get_tensor(output_details_diffusion[0]['index'])
    # print(latent.shape)
    return unconditional_latent + unconditional_guidance_scale * (
        latent - unconditional_latent
    )

def get_x_prev_and_pred_x0(x, e_t, index, a_t, a_prev, temperature, seed):
    sigma_t = 0
    sqrt_one_minus_at = math.sqrt(1 - a_t)
    pred_x0 = (x - sqrt_one_minus_at * e_t) / math.sqrt(a_t)
    # Direction pointing to x_t
    dir_xt = math.sqrt(1.0 - a_prev - sigma_t**2) * e_t
    noise = sigma_t * tf.random.normal(x.shape, seed=seed) * temperature
    x_prev = math.sqrt(a_prev) * pred_x0 + dir_xt
    return x_prev, pred_x0

timesteps = scheduler.timesteps
progbar = tqdm(enumerate(timesteps))
for index, timestep in progbar:
    progbar.set_description(f"{index:3d} {timestep:3d}")

    initial_input_data = {
        "serving_default_input_1:0": np.array(latent, dtype=np.float32),
        "serving_default_input_2:0": np.array(timestep_embedding([timestep]), dtype=np.float32),
        "serving_default_input_3:0": np.array(context, dtype=np.float32),
    }

    input_data = initial_input_data

    diffusion_model.set_tensor(input_details_diffusion[0]['index'], latent)
    diffusion_model.set_tensor(input_details_diffusion[1]['index'], timestep_embedding([timestep]))
    diffusion_model.set_tensor(input_details_diffusion[2]['index'], context)
    diffusion_model.set_tensor(input_details_diffusion[3]['index'], get_guidance_scale_embedding(unconditional_guidance_scale, 256).astype(np.float32))

    diffusion_model.invoke()
    e_t = diffusion_model.get_tensor(output_details_diffusion[0]['index'])
    # e_t_hf = torch.tensor(np.transpose(e_t, (0, 3, 1, 2)))
    # index_hf = torch.tensor(index)
    # timestep_hf = torch.tensor(timestep)
    # latent_hf = torch.tensor(np.transpose(latent, (0, 3, 1, 2)))
    e_t_hf = np.transpose(e_t, (0, 3, 1, 2))
    index_hf = index
    timestep_hf = timestep
    latent_hf = np.transpose(latent, (0, 3, 1, 2))
    output_latent = scheduler.step(e_t_hf, index_hf, timestep_hf, latent_hf)
    # latent = np.transpose(output_latent[0].detach().cpu().numpy(), (0, 2, 3, 1))
    latent = np.transpose(output_latent[0], (0, 2, 3, 1))


# pdb.set_trace()
denoised = output_latent[1]
# denoised_image = denoised.detach().cpu().numpy()
denoised = np.transpose(denoised, (0, 2, 3, 1))
# denoised_image = np.transpose(denoised_image, (0, 2, 3, 1))
# denoised = np.transpose(denoised, (0, 2, 3, 1))
decoder.set_tensor(input_details_decoder[0]['index'], denoised)
decoder.invoke()
decoded = decoder.get_tensor(output_details_decoder[0]['index'])
# decoded = model.decoder.predict_on_batch(latent)
decoded = ((decoded + 1) / 2) * 255
img=np.clip(decoded, 0, 255).astype("uint8")

plt.imshow(img[0])
img=img[0]
plt.savefig(args.output)
end = time.time()
 
# print the difference between start
# and end time in milli. secs
print("The time of execution of above program is :",
      (end-start) * 10**3, "ms")