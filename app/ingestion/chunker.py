def chunk_code(files, chunk_size=500, overlap=50):

    chunks = []

    for file in files:

        content = file["content"]
        file_path = file["file_path"]

        start = 0

        while start < len(content):

            end = start + chunk_size

            chunk_text = content[start:end]

            chunks.append({
                "file_path": file_path,
                "chunk": chunk_text
            })

            start = end - overlap

    return chunks
    

#For testing:
if __name__ == "__main__":

    from repo_loader import load_repository

    files = load_repository("data/repositories/sample_repo")

    chunks = chunk_code(files)

    print("Total chunks:", len(chunks))
    print(chunks[0])


