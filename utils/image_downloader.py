import json
import requests
import os
import shutil
from pathlib import Path

class ImageDownloader:
    def __init__(self, img_out_folder: str, export_json_path: str):
        """
        Initialize the ImageDownloader.

        Args:
            img_out_folder (str): The folder where downloaded images will be saved.
            export_json_path (str): The path to the JSON file containing image information.
        """
        self.img_out_folder = Path(img_out_folder)
        self.img_out_folder.mkdir(exist_ok=True)
        self.export_json_path = export_json_path

    def download_images(self) -> None:
        """
        Download images from URLs specified in the JSON dataset.

        This method downloads images from URLs provided in the JSON dataset. It saves
        the downloaded images to the specified output folder and handles any errors
        or unreachable images.

        Returns:
            None
        """
        # Load the JSON dataset
        with open(self.export_json_path, 'r', encoding='utf-8') as file:
            dataset = json.load(file)
        
        # Create a file to store unreachable image names
        with open("unreachable_imgs.txt", "w", encoding='utf-8') as unreachable_imgs_file:
            for item in dataset:
                asset_url = item['asset']
                asset_name = item['externalId']
                img_path = self.img_out_folder / asset_name
                
                # Skip if the image is already downloaded
                if os.path.isfile(img_path):
                    continue
                
                try:
                    # Try to fetch the image with a timeout of 30 seconds
                    asset_response = requests.get(asset_url, stream=True, timeout=30)
                    asset_response.raise_for_status()
                    
                    if asset_response.status_code == 200:
                        with open(img_path, 'wb') as f:
                            asset_response.raw.decode_content = True
                            shutil.copyfileobj(asset_response.raw, f)
                        
                except requests.exceptions.HTTPError as e:
                    # Handle 404 errors
                    unreachable_imgs_file.write(asset_name + "\n")
                    print('\n'.join( [f"Error downloading image {asset_name}: {e}"] ))
                
        print("Image downloading completed.")
