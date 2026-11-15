# download_hf.py
from huggingface_hub import list_repo_files, hf_hub_download

REPO = "McAuley-Lab/Amazon-Reviews-2023"
files = [f for f in list_repo_files(REPO, repo_type="dataset")
         if "Video_Games" in f and f.endswith(".jsonl")
         and ("review_categories" in f or "meta_categories" in f)]
print(files)  # проверьте, что тут два файла: отзывы и meta
for f in files:
    print(hf_hub_download(REPO, f, repo_type="dataset", local_dir="F:/New folder/data/raw"))