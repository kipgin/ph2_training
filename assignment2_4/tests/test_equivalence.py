import torch
from models import Transformer2DModel
from diffusers.models.transformers.transformer_2d import Transformer2DModel as HFTransformer2DModel

def test_transformer_internals():
    print("--- Detailed Transformer2DModel Diagnostics ---")
    custom = Transformer2DModel(in_channels=320, n_heads=8, d_head=40, context_dim=768)
    hf = HFTransformer2DModel(in_channels=320, num_attention_heads=8, attention_head_dim=40, cross_attention_dim=768)
    
    custom.load_state_dict(hf.state_dict())
    custom.eval()
    hf.eval()
    
    x = torch.randn(2, 320, 16, 16)
    context = torch.randn(2, 77, 768)
    
    with torch.no_grad():
        # Step 1: Input GroupNorm
        custom_norm = custom.norm(x)
        hf_norm = hf.norm(x)
        print("1. GroupNorm Max Diff:", (custom_norm - hf_norm).abs().max().item())
        
        # Step 2: proj_in
        custom_proj_in = custom.proj_in(custom_norm)
        hf_proj_in = hf.proj_in(hf_norm)
        print("2. proj_in Max Diff:", (custom_proj_in - hf_proj_in).abs().max().item())
        
        # Step 3: Reshape/permute for block input
        custom_block_in = custom_proj_in.permute(0, 2, 3, 1).reshape(2, 16*16, -1)
        custom_block_out = custom_block_in
        for block in custom.transformer_blocks:
            custom_block_out = block(custom_block_out, context=context)
            
        hf_block_in = hf_proj_in.permute(0, 2, 3, 1).reshape(2, 16*16, -1)
        hf_block_out = hf_block_in
        for block in hf.transformer_blocks:
            # check argument names
            # In diffusers, BasicTransformerBlock forward takes encoder_hidden_states
            hf_block_out = block(hf_block_out, encoder_hidden_states=context)
            
        print("3. Transformer Blocks output Max Diff:", (custom_block_out - hf_block_out).abs().max().item())
        
        # Step 4: Output reshape/permute
        custom_out_reshape = custom_block_out.reshape(2, 16, 16, -1).permute(0, 3, 1, 2)
        hf_out_reshape = hf_block_out.reshape(2, 16, 16, -1).permute(0, 3, 1, 2)
        print("4. Reshape/permute back Max Diff:", (custom_out_reshape - hf_out_reshape).abs().max().item())
        
        # Step 5: proj_out
        custom_proj_out = custom.proj_out(custom_out_reshape)
        hf_proj_out = hf.proj_out(hf_out_reshape)
        print("5. proj_out Max Diff:", (custom_proj_out - hf_proj_out).abs().max().item())
        
        # Step 6: Total output (residual + proj_out)
        custom_total = custom_proj_out + x
        hf_total = hf(x, encoder_hidden_states=context).sample
        print("6. Total Output Max Diff:", (custom_total - hf_total).abs().max().item())

if __name__ == "__main__":
    test_transformer_internals()
