import os
import numpy as np
from supabase import create_client
from dotenv import load_dotenv

from src.general_utils import preprocess_image

load_dotenv()

url = os.environ.get("SUPABASE_URL")
key = os.environ.get("SUPABASE_KEY")
main_bucket = os.environ.get("MAIN_BUCKET")

def extract_imgs_from_db(bucket_name):

    supabase = create_client(url, key)
    images = {}

    folders = supabase.storage.from_(bucket_name).list()

    for folder in folders:

        folder_name = folder["name"]
        limit = 100
        offset = 0
        files = []

        while True:
            batch = supabase.storage.from_(bucket_name).list(
                folder_name,
                {"limit": limit, "offset": offset}
            )

            if not batch:
                break

            files.extend(batch)
            offset += limit

        for file in files:
            file_path = f"{folder_name}/{file['name']}"

            try:
                data = supabase.storage.from_(bucket_name).download(file_path)
            except Exception as e:
                print(f"Failed: {file_path} → {e}")
                continue

            tensor = preprocess_image(data)

            if folder_name not in images:
                images[folder_name] = []

            images[folder_name].append(tensor)

    return images