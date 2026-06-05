import os
import hashlib
import git 

def get_system_metadata() -> dict:
    try:
        repo = git.Repo(search_parent_directories=True)
        git_sha = repo.head.object.hexsha[:8]
        is_dirty = repo.is_dirty()
    except:
        git_sha = "dev-unknown"
        is_dirty = True

    with open("app/core/system2_agent.py", "r") as f:
        prompt_content = f.read()
        prompt_hash = hashlib.md5(prompt_content.encode()).hexdigest()[:8]

    return {
        "service_version": "0.3.0-pilot",
        "git_sha": git_sha,
        "is_dirty": is_dirty,
        "vision_model": "yolov8s-v0.3",
        "llm_model": "llama3.1:8b",
        "prompt_hash": prompt_hash,
        "cost_table_version": "Q3-2024-v1"
    }