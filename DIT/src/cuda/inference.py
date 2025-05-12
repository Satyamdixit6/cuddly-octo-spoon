import sys
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load
import numpy as np
from PIL import Image
import os
import urllib.request
from diffusers.models import AutoencoderKL
from diffusers import DDPMScheduler
import logging
from tqdm import tqdm
import subprocess

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Add the DiT directory to the Python path
sys.path.insert(0, "/home/satyam/Music/DIT/src/cuda/DiT")

# Pre-trained model URLs
MODEL_URLS = {
    '256': 'https://dl.fbaipublicfiles.com/DiT/models/DiT-XL-2-256x256.pt',
    '512': 'https://dl.fbaipublicfiles.com/DiT/models/DiT-XL-2-512x512.pt'
}

# Text-to-ImageNet class mapping
TEXT_TO_CLASS = {
    "cat": 281,  # Tabby cat
    "dog": 207,  # Golden retriever
    "car": 817,  # Sports car
    "tree": 974,  # Pine tree
}

# Fallback diffusion scheduler
def create_diffusion(timesteps):
    return DDPMScheduler(num_train_timesteps=int(timesteps))

# Load CUDA extensions
use_cuda = False
dit_block_mod = None
try:
    logger.info("Loading CUDA extensions...")
    dit_block_mod = load(
        name="dit_block_mod",
        sources=[
            "/home/satyam/Music/DIT/src/cuda/dit_block.cu",
            "/home/satyam/Music/DIT/src/cuda/cuda_adaln_modulation.cu",
            "/home/satyam/Music/DIT/src/cuda/cuda_attention.cu",
            "/home/satyam/Music/DIT/src/cuda/cuda_mlp.cu",
            "/home/satyam/Music/DIT/src/cuda/cuda_timestep_embed.cu",
            "/home/satyam/Music/DIT/src/cuda/cuda_util_kernels.cu",
            "/home/satyam/Music/DIT/src/cuda/cuda_layernorm.cu",
            "/home/satyam/Music/DIT/src/cuda/cuda_label_embed.cu",
            "/home/satyam/Music/DIT/src/cuda/final_layer.cu"
        ],
        extra_cuda_cflags=["-O3", "-Xcompiler", "-Wall"],
        extra_ldflags=["-lcublas"],  # Link cuBLAS
        verbose=True
    )
    logger.info("CUDA extensions loaded successfully.")
    use_cuda = True
except subprocess.CalledProcessError as e:
    logger.error(f"CUDA compilation failed: {e.output.decode()}")
    logger.info("Falling back to PyTorch implementation.")
except Exception as e:
    logger.error(f"Failed to load CUDA extensions: {e}")
    logger.info("Falling back to PyTorch implementation.")

class DiTBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, hidden_size)
        )
        self.adaLN_modulation = nn.Linear(hidden_size, 6 * hidden_size)

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=-1)
        x = self.norm1(x)
        x = x * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
        attn_out, _ = self.attn(x, x, x)
        x = x + gate_msa.unsqueeze(1) * attn_out
        x = self.norm2(x)
        x = x * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        mlp_out = self.mlp(x)
        x = x + gate_mlp.unsqueeze(1) * mlp_out
        return x

class DiTInference(nn.Module):
    def __init__(
        self,
        input_size=32,  # 256/8 = 32 for 256x256 images
        hidden_size=1152,  # DiT-XL/2 default
        depth=28,  # DiT-XL/2 default
        num_heads=16,  # DiT-XL/2 default
        mlp_ratio=4.0,
        num_classes=1000,
        learn_sigma=True
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.depth = depth
        
        # Input embedding
        self.x_embedder = nn.Conv2d(4, hidden_size, kernel_size=1)
        self.pos_embed = nn.Parameter(torch.zeros(1, input_size ** 2, hidden_size))
        
        # Time and class embeddings
        if not use_cuda:
            self.t_embedder = nn.Sequential(
                nn.Linear(1, hidden_size),
                nn.SiLU(),
                nn.Linear(hidden_size, hidden_size)
            )
            self.y_embedder = nn.Embedding(num_classes + 1, hidden_size)
        else:
            # CUDA weights for timestep and label embeddings
            self.t_freqs = torch.exp(-torch.linspace(0, 1, hidden_size // 2) * 4.0)
            self.y_embed_table = nn.Parameter(torch.randn(num_classes + 1, hidden_size))
        
        # DiT blocks (PyTorch fallback or CUDA weights)
        if not use_cuda:
            self.blocks = nn.ModuleList([
                DiTBlock(hidden_size, num_heads, mlp_ratio) for _ in range(depth)
            ])
        else:
            self.block_weights = nn.ModuleList([
                nn.ModuleDict({
                    'w_mod': nn.Parameter(torch.randn(hidden_size, 6 * hidden_size)),
                    'b_mod': nn.Parameter(torch.randn(6 * hidden_size)),
                    'wQ': nn.Parameter(torch.randn(hidden_size, hidden_size)),
                    'bQ': nn.Parameter(torch.randn(hidden_size)),
                    'wK': nn.Parameter(torch.randn(hidden_size, hidden_size)),
                    'bK': nn.Parameter(torch.randn(hidden_size)),
                    'wV': nn.Parameter(torch.randn(hidden_size, hidden_size)),
                    'bV': nn.Parameter(torch.randn(hidden_size)),
                    'wO': nn.Parameter(torch.randn(hidden_size, hidden_size)),
                    'bO': nn.Parameter(torch.randn(hidden_size)),
                    'w1': nn.Parameter(torch.randn(hidden_size, int(hidden_size * mlp_ratio))),
                    'b1': nn.Parameter(torch.randn(int(hidden_size * mlp_ratio))),
                    'w2': nn.Parameter(torch.randn(int(hidden_size * mlp_ratio), hidden_size)),
                    'b2': nn.Parameter(torch.randn(hidden_size))
                }) for _ in range(depth)
            ])
        
        # Final layer
        self.final_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.final_adaLN_modulation = nn.Linear(hidden_size, 2 * hidden_size)
        self.final_linear = nn.Linear(hidden_size, 8 if learn_sigma else 4)
        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)
        nn.init.normal_(self.pos_embed, std=0.02)

    def unpatchify(self, x):
        h = w = self.input_size
        x = x.reshape(shape=(x.shape[0], h, w, self.hidden_size))
        return x.permute(0, 3, 1, 2)  # (N, C, H, W)

    def timestep_embedding(self, t):
        dim = self.hidden_size
        half = dim // 2
        freqs = torch.exp(-torch.linspace(0, 1, half, device=t.device) * 4.0)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        return self.t_embedder(embedding)

    def forward(self, x, timesteps, y):
        x = self.x_embedder(x)
        x = x.flatten(2).transpose(1, 2)  # (N, T, C)
        x = x + self.pos_embed

        if use_cuda and dit_block_mod is not None:
            t_emb = dit_block_mod.timestep_embedding_forward(timesteps, self.t_freqs.to(x.device), self.hidden_size)
            y_emb = dit_block_mod.label_embed_forward(y, self.y_embed_table, torch.zeros_like(y, dtype=torch.int32))
            c = t_emb + y_emb

            for block_weights in self.block_weights:
                x = dit_block_mod.dit_block_forward(
                    x, c,
                    block_weights['w_mod'], block_weights['b_mod'],
                    block_weights['wQ'], block_weights['bQ'],
                    block_weights['wK'], block_weights['bK'],
                    block_weights['wV'], block_weights['bV'],
                    block_weights['wO'], block_weights['bO'],
                    block_weights['w1'], block_weights['b1'],
                    block_weights['w2'], block_weights['b2'],
                    eps=1e-6
                )
            
            shift, scale = self.final_adaLN_modulation(c).chunk(2, dim=-1)
            x = dit_block_mod.adaln_forward(x, shift, scale, eps=1e-6)
            x = self.unpatchify(x)
            x = self.final_linear(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        else:
            t_emb = self.timestep_embedding(timesteps)
            y_emb = self.y_embedder(y)
            c = t_emb + y_emb
            for block in self.blocks:
                x = block(x, c)
            shift, scale = self.final_adaLN_modulation(c).chunk(2, dim=-1)
            x = self.final_norm(x)
            x = x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
            x = self.unpatchify(x)
            x = self.final_linear(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        if self.learn_sigma:
            x, log_var = x.chunk(2, dim=1)
            return x, log_var
        return x, None

def download_model(image_size):
    os.makedirs('pretrained', exist_ok=True)
    model_path = f'pretrained/DiT-XL-2-{image_size}x{image_size}.pt'
    
    if not os.path.exists(model_path):
        logger.info(f"Downloading DiT-XL/2 {image_size}x{image_size} model...")
        with tqdm(unit='B', unit_scale=True, desc="Downloading") as pbar:
            def report_hook(count, block_size, total_size):
                pbar.total = total_size
                pbar.update(block_size)
            urllib.request.urlretrieve(MODEL_URLS[str(image_size)], model_path, reporthook=report_hook)
        logger.info(f"Model downloaded to {model_path}")
    
    return model_path

def generate_images(
    prompt=None,
    num_samples=1,
    image_size=256,
    guidance_scale=7.5,
    num_inference_steps=50,
    seed=None,
    device="cuda"
):
    assert image_size in [256, 512], "Image size must be either 256 or 512"
    
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        device = "cpu"
    
    logger.info("Loading model...")
    model_path = download_model(image_size)
    model = DiTInference(input_size=image_size//8)
    try:
        state_dict = torch.load(model_path, map_location=device)
        if "model" in state_dict:
            state_dict = state_dict["model"]
        model.load_state_dict(state_dict)
    except Exception as e:
        logger.error(f"Failed to load model checkpoint: {e}")
        raise
    model.to(device)
    model.eval()
    logger.info("Model loaded successfully")

    if isinstance(prompt, str):
        for key, class_id in TEXT_TO_CLASS.items():
            if key in prompt.lower():
                class_labels = torch.full((num_samples,), class_id, dtype=torch.long, device=device)
                break
        else:
            logger.warning(f"No class mapping for prompt '{prompt}', using class 0")
            class_labels = torch.zeros(num_samples, dtype=torch.long, device=device)
    elif isinstance(prompt, int):
        class_labels = torch.full((num_samples,), prompt, dtype=torch.long, device=device)
    else:
        class_labels = torch.zeros(num_samples, dtype=torch.long, device=device)

    logger.info("Loading VAE...")
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-ema").to(device)
    
    diffusion = create_diffusion(num_inference_steps)
    
    logger.info("Starting image generation...")
    with torch.no_grad():
        latents = torch.randn(num_samples, 4, image_size//8, image_size//8, device=device)
        
        def model_fn(x_t, t, y):
            if guidance_scale == 1:
                return model(x_t, t, y)[0]
            
            x_in = torch.cat([x_t] * 2)
            t_in = torch.cat([t] * 2)
            y_in = torch.cat([y, torch.full_like(y, model.num_classes)])
            noise_pred, _ = model(x_in, t_in, y_in)
            noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
            return noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
        
        for i in tqdm(range(num_inference_steps), desc="Diffusion steps"):
            t = torch.full((num_samples,), diffusion.timesteps[i], device=device, dtype=torch.float32)
            noise_pred = model_fn(latents, t, class_labels)
            latents = diffusion.denoising_step(noise_pred, t, latents)
        
        logger.info("Decoding latents...")
        images = vae.decode(latents / 0.18215).sample
        images = torch.clamp((images + 1) / 2, 0, 1)
        images = (images * 255).round().to(torch.uint8)
        images = images.cpu().permute(0, 2, 3, 1).numpy()
        
        return [Image.fromarray(img) for img in images]

if __name__ == "__main__":
    logger.info("Starting script...")
    images = generate_images(
        prompt="A photo of a cat",
        num_samples=4,
        image_size=256,
        guidance_scale=7.5,
        num_inference_steps=50,
        seed=42
    )
    
    os.makedirs('outputs', exist_ok=True)
    for i, image in enumerate(images):
        image.save(f'outputs/sample_{i}.png')
    logger.info("Images saved to outputs/")