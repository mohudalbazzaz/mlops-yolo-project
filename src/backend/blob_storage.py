from typing import Dict, List
import numpy as np
from azure.storage.blob import BlobServiceClient
from azure.identity import DefaultAzureCredential

from src.backend.general_utils import preprocess_image


def extract_imgs_from_db(container_name: str) -> Dict[str, List[np.ndarray]]:
    """
    Extract and preprocess images from an Azure Blob Storage container.

    Args:
        container_name (str): Name of the Azure Blob Storage container
            containing the images.

    Returns:
        Dict[str, List[np.ndarray]]:
            A dictionary mapping class labels (folder names) to lists of
            preprocessed image tensors.
    """
    credential = DefaultAzureCredential()

    blob_service_client = BlobServiceClient(
        account_url=f"https://{container_name}.blob.core.windows.net",
        credential=credential,
    )

    container_client = blob_service_client.get_container_client(container_name)

    images = {}

    blobs = container_client.list_blobs()

    for blob in blobs:
        blob_client = container_client.get_blob_client(blob.name)

        try:
            data = blob_client.download_blob().readall()
        except Exception as e:
            print(f"Failed: {blob.name} --> {e}")
            continue

        tensor = preprocess_image(data)

        folder_name = blob.name.split("/")[0]

        if folder_name not in images:
            images[folder_name] = []

        images[folder_name].append(tensor)

    return images
