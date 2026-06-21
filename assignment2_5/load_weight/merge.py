import os
import torch
from safetensors.torch import load_file, save_file

def merge_safetensors(input_dir, output_file_name="combined_model.safetensors"):
    # Lọc và sắp xếp các file phân mảnh theo thứ tự tăng dần
    files_to_load = sorted([f for f in os.listdir(input_dir) if f.endswith('.safetensors') and '-of-' in f])
    
    if not files_to_load:
        print("Không tìm thấy file .safetensors phân mảnh nào trong thư mục hiện tại.")
        return

    combined_dict = {}
    print(f"Tìm thấy {len(files_to_load)} file phân mảnh. Bắt đầu đọc dữ liệu...")

    for file_name in files_to_load:
        file_path = os.path.join(input_dir, file_name)
        print(f"-> Đang nạp: {file_name}")
        
        # Load dữ liệu tensor từ file phân mảnh
        state_dict = load_file(file_path)
        combined_dict.update(state_dict)

    print(f"Đang tiến hành gộp và lưu thành một file duy nhất: {output_file_name}...")
    save_file(combined_dict, output_file_name)
    print("Quá trình gộp file đã hoàn thành!")

# Chạy trực tiếp script tại thư mục hiện tại
if __name__ == "__main__":
    merge_safetensors(input_dir="/workspace/comfyui/models/text_encoders", output_file_name="qwen_merge_text_encoder.safetensors")