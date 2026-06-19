from huggingface_hub import HfApi
import os

hf_token = os.environ.get("MY_HF_TOKEN")
api = HfApi(token=hf_token)

dataset_path = '/home/muhammadali/datasets/starMeter'

api.upload_large_folder(
    folder_path=dataset_path,
    repo_id='kalandarX/starMeter',
    repo_type='dataset',
)
