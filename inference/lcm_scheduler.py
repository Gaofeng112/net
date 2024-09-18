from typing import List, Optional, Tuple, Union

import numpy as np


class LCMScheduler:
    order = 1

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        beta_schedule: str = "linear",
        trained_betas: Optional[Union[np.ndarray, List[float]]] = None,
        clip_sample: bool = True,
        set_alpha_to_one: bool = True,
        steps_offset: int = 0,
        prediction_type: str = "epsilon",
        thresholding: bool = False,
        dynamic_thresholding_ratio: float = 0.995,
        clip_sample_range: float = 1.0,
        sample_max_value: float = 1.0,
        timestep_spacing: str = "leading",
        rescale_betas_zero_snr: bool = False,
    ):
        self.prediction_type = prediction_type
        self.num_train_timesteps = num_train_timesteps
        self.betas = (
            np.linspace(beta_start**0.5, beta_end**0.5, num_train_timesteps, dtype=np.float32) ** 2
        )

        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = np.cumprod(self.alphas)

        self.final_alpha_cumprod = np.array(1.0) if set_alpha_to_one else self.alphas_cumprod[0]

        # standard deviation of the initial noise distribution
        self.init_noise_sigma = 1.0

        # setable values
        self.num_inference_steps = None
        self.timesteps = np.arange(0, num_train_timesteps)[::-1].copy().astype(np.int64)

    def set_timesteps(self, num_inference_steps: int, lcm_origin_steps: int):
        self.num_inference_steps = num_inference_steps
        
        # LCM Timesteps Setting:  # Linear Spacing
        c = self.num_train_timesteps // lcm_origin_steps
        lcm_origin_timesteps = np.asarray(list(range(1, lcm_origin_steps + 1))) * c  - 1   # LCM Training  Steps Schedule
        skipping_step = len(lcm_origin_timesteps) // num_inference_steps
        timesteps = lcm_origin_timesteps[::-skipping_step][:num_inference_steps]           # LCM Inference Steps Schedule
        
        self.timesteps = timesteps
    def get_scalings_for_boundary_condition_discrete(self, t):
        self.sigma_data = 0.5       # Default: 0.5
        
        # By dividing 0.1: This is almost a delta function at t=0.     
        c_skip = self.sigma_data**2 / (
                (t / 0.1) ** 2 + self.sigma_data**2
            )
        c_out = (( t / 0.1)  / ((t / 0.1) **2 + self.sigma_data**2) ** 0.5)
        return c_skip, c_out
        
    
    def step(
        self,
        model_output: np.ndarray,
        timeindex: int,
        timestep: int,
        sample: np.ndarray,
        eta: float = 0.0,
    ):
        # 1. get previous step value
        prev_timeindex = timeindex + 1
        if prev_timeindex < len(self.timesteps):
            prev_timestep = self.timesteps[prev_timeindex]
        else:
            prev_timestep = timestep
        
        # 2. compute alphas, betas
        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.final_alpha_cumprod
        
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev
        
        # 3. Get scalings for boundary conditions
        c_skip, c_out = self.get_scalings_for_boundary_condition_discrete(timestep)
        
        # 4. Different Parameterization:
        parameterization = self.prediction_type
        
        if parameterization == "epsilon":           # noise-prediction
            pred_x0 = (sample - np.sqrt(beta_prod_t) * model_output) / np.sqrt(alpha_prod_t)
            
        # 4. Denoise model output using boundary conditions
        denoised = c_out * pred_x0 + c_skip * sample
        
        # 5. Sample z ~ N(0, I), For MultiStep Inference
        # Noise is not used for one-step sampling.
        noise = np.random.normal(size=model_output.shape)
        prev_sample = np.sqrt(alpha_prod_t_prev) * denoised + np.sqrt(beta_prod_t_prev) * noise
        
        return prev_sample.astype(np.float32), denoised.astype(np.float32)
