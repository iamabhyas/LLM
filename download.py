import os

from huggingface_hub import hf_hub_download

HUGGING_FACE_API_KEY = ''
model_path = 'models'

# Check if the folder exists
if not os.path.exists(model_path):
    # If it does not exist, create it
    os.makedirs(model_path)
    print(f"The directory {model_path} was created.")
else:
    print(f"The directory {model_path} already exists.")

repo_id = "bigcode/starcoder"
filenames = [
    "pytorch_model-00006-of-00007.bin",
    "config.json",
    "generation_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json"
]

for filename in filenames:
    downloaded_model_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=model_path,
        local_dir_use_symlinks=False,
	token=HUGGING_FACE_API_KEY
    )
    print(downloaded_model_path)
