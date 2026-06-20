import torch
from transformers import BitsAndBytesConfig
from diffusers import QwenImagePipeline, FluxTransformer2DModel 

model_id = "Qwen/Qwen-Image"

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)


print("Loading quantized transformer component...")
transformer = FluxTransformer2DModel.from_pretrained(
    model_id,
    subfolder="transformer",
    quantization_config=quantization_config,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

print("Initializing Qwen-Image Pipeline...")
pipe = QwenImagePipeline.from_pretrained(
    model_id,
    transformer=transformer,
    torch_dtype=torch.bfloat16,
)

pipe.enable_model_cpu_offload()

prompt = "A majestic dragon flying over a Chinese pagoda, traditional ink painting style, high resolution."

image = pipe(
    prompt=prompt,
    num_inference_steps=28,
    guidance_scale=3.5,
    height=1024,
    width=1024,
).images[0]

image.save("qwen_gen.png")
print("Image saved as qwen_gen.png")