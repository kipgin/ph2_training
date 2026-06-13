import torch
from diffusers import StableDiffusionPipeline

def export_pipeline_blueprints(pipe, architecture_file_path: str, flow_algorithm_path: str):
    """
    Exports the static PyTorch architecture and the dynamic tensor flow 
    (Text Encoder, UNet, and VAE) of a Stable Diffusion pipeline to external text files.
    """
    
    # --- 1. Write Static Architecture Blueprints ---
    with open(architecture_file_path, "w", encoding="utf-8") as f_arch:
        f_arch.write("==================================================\n")
        f_arch.write("TEXT ENCODER ARCHITECTURE & MODULE DIMENSIONS\n")
        f_arch.write("==================================================\n")
        f_arch.write(str(pipe.text_encoder))
        f_arch.write("\n\n")
        
        f_arch.write("==================================================\n")
        f_arch.write("UNET ARCHITECTURE & MODULE DIMENSIONS\n")
        f_arch.write("==================================================\n")
        f_arch.write(str(pipe.unet))
        f_arch.write("\n\n")
        
        f_arch.write("==================================================\n")
        f_arch.write("VAE ARCHITECTURE & MODULE DIMENSIONS\n")
        f_arch.write("==================================================\n")
        f_arch.write(str(pipe.vae))
        f_arch.write("\n")

    # --- 2. Setup Dynamic Flow File & Hooks ---
    f_flow = open(flow_algorithm_path, "w", encoding="utf-8")
    f_flow.write("==================================================\n")
    f_flow.write("DYNAMIC TENSOR FLOW & FEATURE MAP SHAPES\n")
    f_flow.write("==================================================\n\n")

    def create_hook_fn(prefix: str):
        """Generates a hook function tagged with the component prefix."""
        def hook_fn(module, input, output):
            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            
            # Extract underlying tensor if output is wrapped in a Diffusers/Transformers output object
            if hasattr(output, 'sample') and isinstance(output.sample, torch.Tensor):
                tensor_shape = list(output.sample.shape)
            elif hasattr(output, 'last_hidden_state') and isinstance(output.last_hidden_state, torch.Tensor):
                tensor_shape = list(output.last_hidden_state.shape)
            elif isinstance(output, torch.Tensor):
                tensor_shape = list(output.shape)
            elif isinstance(output, tuple) and len(output) > 0 and isinstance(output[0], torch.Tensor):
                tensor_shape = list(output[0].shape)
            else:
                # Skip non-tensor components (like attention weights lists)
                return

            f_flow.write(f"[{prefix}][{class_name}]: Feature Map Shape -> {tensor_shape}\n")
        return hook_fn

    hooks = []
    
    # Register Text Encoder Hooks
    for name, layer in pipe.text_encoder.named_children():
        hook = layer.register_forward_hook(create_hook_fn("TextEncoder"))
        hooks.append(hook)

    # Register UNet Hooks
    for name, layer in pipe.unet.named_children():
        hook = layer.register_forward_hook(create_hook_fn("UNet"))
        hooks.append(hook)

    # Register VAE Hooks
    for name, layer in pipe.vae.named_children():
        hook = layer.register_forward_hook(create_hook_fn("VAE"))
        hooks.append(hook)

    # --- 3. Execute Dummy Pass to Populate Flow Data ---
    try:
        prompt = "a structural blueprint mapping out execution routing"
        with torch.inference_mode():
            # Run a standard generation pass to trigger TextEncoder and UNet components
            latents = pipe(prompt, num_inference_steps=1, height=512, width=512).images
            
            # The standard pipeline decodes to PIL images, bypasses VAE hooks for internal submodules sometimes.
            # To explicitly force a full forward pass through all VAE components:
            dummy_latents = torch.randn(1, 4, 64, 64, dtype=pipe.vae.dtype, device=pipe.device)
            pipe.vae.decode(dummy_latents)
            
    finally:
        # Clean up hooks and close the flow file safely
        for hook in hooks:
            hook.remove()
        f_flow.close()