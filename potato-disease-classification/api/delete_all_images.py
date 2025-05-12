import cloudinary
import cloudinary.api
import os
from dotenv import load_dotenv
load_dotenv()

# Step 1: Configure your Cloudinary credentials
cloudinary.config(
    cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME"),
    api_key=os.getenv("CLOUDINARY_API_KEY"),
    api_secret=os.getenv("CLOUDINARY_API_SECRET"),
    secure=True
)

# Step 2: Define your target folder
folder_name = "potato_disease_uploads"

def list_all_images(folder):
    public_ids = []
    next_cursor = None

    while True:
        response = cloudinary.api.resources(
            type="upload",
            prefix=f"{folder}/",
            max_results=500,
            next_cursor=next_cursor
        )
        public_ids.extend([resource["public_id"] for resource in response.get("resources", [])])
        
        next_cursor = response.get("next_cursor")
        if not next_cursor:
            break

    return public_ids


# Step 4: Delete all resources
def delete_all_images(public_ids):
    if public_ids:
        response = cloudinary.api.delete_resources(public_ids)
        print(f"[✅] Deleted {len(public_ids)} image(s):")
        print(public_ids)
    else:
        print("[ℹ️] No images found to delete.")


if __name__ == "__main__":
    public_ids = list_all_images(folder_name)
    delete_all_images(public_ids)

  