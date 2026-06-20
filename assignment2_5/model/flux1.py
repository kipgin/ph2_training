from diffusers import FluxPipeline
import torch

pipe = FluxPipeline.from_pretrained(
    "diffusers/FLUX.1-dev-bnb-8bit",
    torch_dtype=torch.bfloat16,
    device_map="balanced"  
)

prompt = "Baroque style, a lavish palace interior with ornate gilded ceilings, intricate tapestries, and dramatic lighting over a grand staircase."

pipe_kwargs = {
    "prompt": prompt,
    "height": 1024,
    "width": 1024,
    "guidance_scale": 3.5,
    "num_inference_steps": 50,
    "max_sequence_length": 512,
}

image = pipe(
    **pipe_kwargs, 
    generator=torch.manual_seed(0),
).images[0]

image.save("flux.png")

#pipe.enable_model_cpu_offload()
#pipe.enable_sequential_cpu_offload()