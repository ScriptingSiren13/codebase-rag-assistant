import os


def load_repository(repo_path):

    documents = []

    for root, dirs, files in os.walk(repo_path):

        for file in files:

            file_path = os.path.join(root, file)

            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()

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

