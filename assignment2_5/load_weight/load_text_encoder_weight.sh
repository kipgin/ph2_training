cd /workspace/comfyui/models/text_encoders
mkdir -p qwen_image_text_encoders
# Chạy vòng lặp tải 3 file (Thay đổi URL và tên file tương ứng với mô hình bạn cần tải)
for i in {1..2}; do
  # Định dạng số thứ tự thành dạng 5 chữ số (00001, 00002, 00003)
  num=$(printf "%05d" $i)
  
  wget -O "model-${num}-of-00003.safetensors" "https://huggingface.co/ovedrive/qwen-image-4bit/resolve/main/text_encoder/model-${num}-of-00002.safetensors"
done