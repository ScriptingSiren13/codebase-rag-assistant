import os
from git import Repo


def clone_repository(repo_url, save_path="data/repositories"):

    repo_name = repo_url.split("/")[-1].replace(".git", "")
    repo_path = os.path.join(save_path, repo_name)

    os.makedirs(save_path, exist_ok=True)

    if not os.path.exists(repo_path):
        print("Cloning repository...")
        Repo.clone_from(repo_url, repo_path)
    else:
        print("Repository already exists.")

    return repo_path