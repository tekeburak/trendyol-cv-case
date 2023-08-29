import json
from PIL import Image
import os
from pathlib import Path
from collections import defaultdict


class ImageCropper:
    def __init__(
        self,
        images_folder: str,
        export_json_path: str,
        crop_out_folder: str,
        dataset_csv_out: str
    ):
        """
        Initialize the ImageCropper.

        Args:
            images_folder (str): Path to the folder containing images.
            export_json_path (str): Path to the JSON file containing dataset information.
            crop_out_folder (str): Path to the folder where cropped images will be saved.
            dataset_csv_out (str): Path to the CSV file where dataset information will be saved.
        """
        self.images_folder = Path(images_folder)
        self.export_json_path = export_json_path
        self.crop_out_folder = Path(crop_out_folder)
        self.crop_out_folder.mkdir(exist_ok=True)
        self.dataset_csv_out = dataset_csv_out
        self.crop_info_dict = defaultdict(lambda: 0)
        
        self.default_img_width = 1200
        self.default_img_height = 1800
        self.not_accessed_image_list = []

    def crop_images(self) -> None:
        """
        Crop images based on bounding box information and save them.

        Returns:
            None
        """
        # Open the export JSON file for reading
        with open(self.export_json_path, 'r', encoding='utf-8') as file:
            # Load the JSON data into the 'dataset' variable
            dataset = json.load(file)
            
        # Open a CSV file for writing dataset information
        dataset_info_csv_file = open(self.dataset_csv_out, 'w', encoding='utf-8')
        # Write the header line to the CSV file
        dataset_info_csv_file.write("asset_name,cropped_image_name," +
                                    "main_category,crop_label," +
                                    "crop_label_index,xmin,ymin," +
                                    "xmax,ymax" + "\n")
        
        for item in dataset:
            asset_url = item['asset']
            asset_name = item['externalId']
            img_path = self.images_folder / asset_name
            
            # Check if the image exists in the images folder
            if not os.path.isfile(img_path):
                # If the image is missing, log it and skip to the next item
                self.not_accessed_image_list.append(asset_name)
                continue
            
            # Open the asset image
            asset_image = Image.open(img_path)
            
            # Loop through each crop object in the item's tasks
            for crop_obj in item['tasks'][0]['objects']:
                main_category = crop_obj['title']
                bounding_box = crop_obj['bounding-box']
                x = int(bounding_box['x'])
                y = int(bounding_box['y'])
                
                width = int(bounding_box['width'])
                height = int(bounding_box['height'])
                
                # Check if width or height is negative
                if width < 0:
                    x = x + width  # Adjust x-coordinate
                    width = abs(width)  # Make width positive
                
                if height < 0:
                    y = y + height  # Adjust y-coordinate
                    height = abs(height)  # Make height positive
                
                # Calculate coordinates for cropping
                xmin, ymin, xmax, ymax = (x, y, x + width, y + height)
                
                # Extract crop label information
                crop_label = crop_obj['classifications'][0]['answer']
                self.crop_info_dict[crop_label] += 1
                crop_label_index = list(self.crop_info_dict.keys()).index(crop_label) + 1
                crop_label_index = str(crop_label_index).zfill(2)
                
                # Create a unique name for the cropped image
                cropped_image_name = f"id_{str(self.crop_info_dict[crop_label]).zfill(5)}_{crop_label}.jpg"
                
                # Create a line for the dataset info file
                dataset_info_line = f"{asset_name},{cropped_image_name}, \
                                      {main_category},{crop_label}, \
                                      {crop_label_index},{xmin},{ymin}, \
                                      {xmax},{ymax}"
                
                # Write the dataset info line to the dataset file
                dataset_info_csv_file.write(dataset_info_line + '\n')
                
                # Skip if the image is already cropped
                if os.path.isfile(self.crop_out_folder / cropped_image_name):
                    continue
                
                # Crop the image using calculated coordinates
                cropped_image = asset_image.crop((xmin, ymin, xmax, ymax))
                
                # Convert the image to RGB mode if not already
                if cropped_image.mode != 'RGB':
                    cropped_image = cropped_image.convert('RGB')
                
                # Save the cropped image to the designated output folder
                cropped_image.save(self.crop_out_folder / cropped_image_name)
                
        # Close the dataset CSV file after writing is complete
        dataset_info_csv_file.close()
