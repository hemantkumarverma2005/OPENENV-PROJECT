"""Upload all project files to HuggingFace Space using the API."""
import os
from huggingface_hub import HfApi, login

# Login using the provided token
token = os.getenv("HF_TOKEN")
if token:
    login(token=token)
else:
    print("HF_TOKEN not found in environment, assuming already logged in or not required.")

api = HfApi()
repo_id = "Tyr-123/TEST"
repo_type = "space"
folder = os.path.dirname(os.path.abspath(__file__))

print(f"Uploading project to: https://huggingface.co/spaces/{repo_id}")
print(f"From folder: {folder}")

api.upload_folder(
    folder_path=folder,
    repo_id=repo_id,
    repo_type=repo_type,
    ignore_patterns=["*.git*", "__pycache__/**", "uv.lock",
                     "generate_pdf.py", "generate_qa_pdf.py",
                     "RL_Crash_Course.pdf", "Judge_QA_Guide.pdf",
                     "test_integration.py", ".gemini/**", "upload_hf.py"],
    commit_message="SocialContract-v0: Complete submission with multi-agent, curriculum, GRPO training",
)

print(f"\nDone! View at: https://huggingface.co/spaces/{repo_id}")
