import os
from git import Repo


def clone_repository(repo_url, save_path="data/repositories"):

    #Extracting a structured folder name
    repo_name = repo_url.split("/")[-1].replace(".git", "")
    repo_path = os.path.join(save_path, repo_name)

    #Ensure storage directory exists
    os.makedirs(save_path, exist_ok=True)

    #Checking if repo is already present
    if not os.path.exists(repo_path):
        print("Cloning repository...")
        Repo.clone_from(repo_url, repo_path)  #	Cloning only if needed
    else:
        print("Repository already exists.")

    return repo_path  #Return the local path