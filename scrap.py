from huggingface_hub import HfApi

# Define the repository details
REPO_ID = "alexanderdann/CTSpine1K" 
REPO_TYPE = "dataset"
PATH_PREFIX = "raw_data/volumes/COLONOG"

OUTPUT_FILE = "colonog_files.txt"

try:
    api = HfApi()
    files = api.list_repo_files(repo_id=REPO_ID, repo_type=REPO_TYPE)

    matching_files = [f for f in files if f.startswith(PATH_PREFIX)]
    
    with open(OUTPUT_FILE, "w") as f:
        for file_path in matching_files:
            # Remove the prefix, keep only filename
            filename = file_path.replace(PATH_PREFIX + "/", "")
            f.write(filename + "\n")
    
    print(f"Saved {len(matching_files)} files to '{OUTPUT_FILE}'")

except Exception as e:
    print(f"An error occurred: {e}")

