import os


def load_repository(repo_path):

    documents = []

    for root, dirs, files in os.walk(repo_path):  #Traverse all directories recursively

        # 	Filter out irrelevant folders
        if ".git" in root or "__pycache__" in root:
            continue

        for file in files:

            file_path = os.path.join(root, file)

            try:
                with open(file_path, "r", encoding="utf-8") as f:   #Read each valid file
                    content = f.read() 

                #Convert each file into a structured document
                documents.append({
                    "file_path": file_path,
                    "content": content
                })

            except:
                # Skip binary files (images, compiled files, etc.)
                continue

    return documents


# Example run (for testing only)
if __name__ == "__main__":

    repo_path = "data/repositories/sample_repo"

    files = load_repository(repo_path)

    print("Total files loaded:", len(files))

    print("\nExample file path:")
    print(files[0]["file_path"])

    print("\nFirst 200 characters of code:")
    print(files[0]["content"][:200])