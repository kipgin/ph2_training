import torch
from diffusers import Flux2Pipeline

pipe = Flux2Pipeline.from_pretrained(
    "diffusers/FLUX.2-dev-bnb-4bit", 
    torch_dtype=torch.bfloat16
)


pipe.enable_model_cpu_offload()

prompt = "A cinematic shot of a neon-lit cyberpunk street, 4MP resolution, high detail"

image = pipe(
    prompt=prompt,
    num_inference_steps=28, 
    guidance_scale=4.0,
    height=1024,
    width=1024,
).images[0]

image.save("flux2_output.png")