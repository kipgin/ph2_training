from huggingface_hub import HfApi
api = HfApi()
api.upload_folder(
folder_path='/workspace/comfyui/models/checkpoints',
repo_id='your-username/your-private-repo-name',
repo_type='model'
)