from huggingface_hub import HfApi
import os

# 1. Configuration
# Change this to your actual Tourism Space ID
repo_id = "reebamarium/Customer-Purchase-Prediction-App"
token = os.getenv("HF_TOKEN")

# Initialize the API
api = HfApi(token=token)

# 2. Upload the entire project to the Space

try:
    print(f"🚀 Starting upload to Hugging Face Space: {repo_id}...")
    api.upload_folder(
        folder_path=".",              # Uploads everything in the current directory
        repo_id=repo_id,
        repo_type="space",
        # We ignore folders we don't need on the web server to keep it fast
        ignore_patterns=["sample_data/*", "*.csv", "tourism_project/model_building/*"]
    )
    print(f"✅ Success! Your app is being built at: https://huggingface.co/spaces/{repo_id}")
except Exception as e:
    print(f"❌ Upload failed: {e}")
