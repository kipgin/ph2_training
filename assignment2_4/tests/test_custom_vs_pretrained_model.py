import torch
from models import UNet2DConditionModel, get_timestep_embedding
from diffusers import UNet2DConditionModel as HFUNet2DConditionModel

def test_unet_internals():
    print("--- Detailed UNet2DConditionModel Diagnostics ---")
    model_id = "stable-diffusion-v1-5/stable-diffusion-v1-5"
    custom = UNet2DConditionModel.from_pretrained(model_id, torch_dtype=torch.float32)
    hf = HFUNet2DConditionModel.from_pretrained(model_id, subfolder="unet", torch_dtype=torch.float32)
    
    custom.eval()
    hf.eval()
    
    sample = torch.randn(2, 4, 64, 64)
    timestep = torch.tensor([15, 30], dtype=torch.long)
    encoder_hidden_states = torch.randn(2, 77, 768)
    
    with torch.no_grad():
        t_custom = timestep
        if t_custom.shape[0] == 1:
            t_custom = t_custom.expand(sample.shape[0])
            
        temb_sin_custom = get_timestep_embedding(t_custom, custom.conv_in.out_channels)
        temb_custom = custom.time_embedding(temb_sin_custom)
        
        t_hf = timestep.expand(sample.shape[0])
        t_emb_hf = hf.time_proj(t_hf)
        temb_hf = hf.time_embedding(t_emb_hf)
        
        print("--- Sinusoidal Embedding Stats ---")
        print("Sinusoidal Diff:", (temb_sin_custom - t_emb_hf).abs().max().item())
        print("1. Time embedding Diff:", (temb_custom - temb_hf).abs().max().item())
        
        # 2. conv_in comparison
        x_custom = custom.conv_in(sample)
        x_hf = hf.conv_in(sample)
        print("2. conv_in Diff:", (x_custom - x_hf).abs().max().item())
        
        # 3. Down blocks comparison
        xs_custom = [x_custom]
        xs_hf = [x_hf]
        
        # down block 0
        x_custom, states_custom = custom.down_blocks[0](x_custom, temb_custom, encoder_hidden_states)
        xs_custom.extend(states_custom)
        x_hf, states_hf = hf.down_blocks[0](x_hf, temb_hf, encoder_hidden_states)
        xs_hf.extend(states_hf)
        print("3. Down Block 0 output Diff:", (x_custom - x_hf).abs().max().item())
        
        # down block 1
        x_custom, states_custom = custom.down_blocks[1](x_custom, temb_custom, encoder_hidden_states)
        xs_custom.extend(states_custom)
        x_hf, states_hf = hf.down_blocks[1](x_hf, temb_hf, encoder_hidden_states)
        xs_hf.extend(states_hf)
        print("4. Down Block 1 output Diff:", (x_custom - x_hf).abs().max().item())
        
        # down block 2
        x_custom, states_custom = custom.down_blocks[2](x_custom, temb_custom, encoder_hidden_states)
        xs_custom.extend(states_custom)
        x_hf, states_hf = hf.down_blocks[2](x_hf, temb_hf, encoder_hidden_states)
        xs_hf.extend(states_hf)
        print("5. Down Block 2 output Diff:", (x_custom - x_hf).abs().max().item())
        
        # down block 3
        x_custom, states_custom = custom.down_blocks[3](x_custom, temb_custom, encoder_hidden_states)
        xs_custom.extend(states_custom)
        x_hf, states_hf = hf.down_blocks[3](x_hf, temb_hf)
        xs_hf.extend(states_hf)
        print("6. Down Block 3 output Diff:", (x_custom - x_hf).abs().max().item())
        
        # 4. Mid block comparison
        x_custom = custom.mid_block(x_custom, temb_custom, encoder_hidden_states)
        x_hf = hf.mid_block(x_hf, temb_hf, encoder_hidden_states)
        print("7. Mid Block output Diff:", (x_custom - x_hf).abs().max().item())
        
        # 5. Up blocks comparison
        skips_custom = list(xs_custom)
        skips_hf = list(xs_hf)
        
        # up block 0
        x_custom = custom.up_blocks[0](x_custom, skips_custom, temb_custom, encoder_hidden_states)
        hf_skips_0 = skips_hf[-3:]
        skips_hf = skips_hf[:-3]
        x_hf = hf.up_blocks[0](x_hf, hf_skips_0, temb_hf)
        print("8. Up Block 0 output Diff:", (x_custom - x_hf).abs().max().item())
        
        # up block 1
        x_custom = custom.up_blocks[1](x_custom, skips_custom, temb_custom, encoder_hidden_states)
        hf_skips_1 = skips_hf[-3:]
        skips_hf = skips_hf[:-3]
        x_hf = hf.up_blocks[1](x_hf, hf_skips_1, temb_hf, encoder_hidden_states)
        print("9. Up Block 1 output Diff:", (x_custom - x_hf).abs().max().item())
        
        # up block 2
        x_custom = custom.up_blocks[2](x_custom, skips_custom, temb_custom, encoder_hidden_states)
        hf_skips_2 = skips_hf[-3:]
        skips_hf = skips_hf[:-3]
        x_hf = hf.up_blocks[2](x_hf, hf_skips_2, temb_hf, encoder_hidden_states)
        print("10. Up Block 2 output Diff:", (x_custom - x_hf).abs().max().item())
        
        # up block 3
        x_custom = custom.up_blocks[3](x_custom, skips_custom, temb_custom, encoder_hidden_states)
        hf_skips_3 = skips_hf[-3:]
        skips_hf = skips_hf[:-3]
        x_hf = hf.up_blocks[3](x_hf, hf_skips_3, temb_hf, encoder_hidden_states)
        print("11. Up Block 3 output Diff:", (x_custom - x_hf).abs().max().item())
        
        # 6. Final output layers comparison
        custom_final = custom.conv_out(custom.conv_act(custom.conv_norm_out(x_custom)))
        hf_final = hf.conv_out(hf.conv_act(hf.conv_norm_out(x_hf)))
        print("12. Final output Diff:", (custom_final - hf_final).abs().max().item())

test_unet_internals()