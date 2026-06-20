import torch
from diffusers import FluxPipeline, FluxTransformer2DModel
from transformers import BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",  
)


transformer = FluxTransformer2DModel.from_pretrained(
    "diffusers/FLUX.1-dev-bnb-4bit",
    subfolder="transformer",
    quantization_config=quantization_config,
    torch_dtype=torch.bfloat16
)

pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    transformer=transformer,
    torch_dtype=torch.bfloat16
)

pipe.enable_model_cpu_offload()


prompt = "A high-tech cyberpunk laboratory with glowing neon blue lights, a robotic arm assembling a glowing crystal, hyper-realistic, 8k"

image = pipe(
    prompt=prompt,
    height=1024,
    width=1024,
    guidance_scale=3.5,
    num_inference_steps=28, 
    generator=torch.manual_seed(42)
).images[0]

image.save("flux_local_4bit.png")