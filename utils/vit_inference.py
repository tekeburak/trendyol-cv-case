from pathlib import Path
import pandas as pd
import numpy as np
import os

import torch
from oml.inference.flat import inference_on_images
from oml.models import ViTExtractor
from oml.registry.transforms import get_transforms_for_pretrained
from sklearn.metrics.pairwise import cosine_similarity

from utils.visualizer import ImageSimilarityVisualizer

class ImageSimilarityViT:
    def __init__(self,
                 crop_folder: str,
                 dataset_csv_path: str,
                 pretrained_dataset: str = "DeepFashion",
                 num_queries: int = 10,
                 top_k: int = 12):
        """
        Initialize the ImageSimilarityViT.

        Args:
            crop_folder (str): Path to the folder containing cropped images.
            dataset_csv_path (str): Path to the CSV file containing dataset information.
            pretrained_dataset (str, optional): Name of the pretrained dataset (either "DeepFashion" or "StanfordOnline").
                                                Defaults to "DeepFashion".
            num_queries (int, optional): Number of query images to use. Defaults to 10.
            top_k (int, optional): Number of top similar images to retrieve. Defaults to 12.
        """
        self.crop_folder = Path(crop_folder)
        self.dataset_csv_path = Path(dataset_csv_path)
        if pretrained_dataset == "DeepFashion":
            self.pretrained_dataset = "vits16_inshop"
        elif pretrained_dataset == "StanfordOnline":
            self.pretrained_dataset = "vits16_sop"
            
        self.num_queries = num_queries
        self.top_k = top_k
        
        self.data_folder = self.dataset_csv_path.parent / "ViT"
        self.data_folder.mkdir(exist_ok=True)

    def analyze_similarity(self) -> None:
        """
        Analyze image similarity and calculate top similar indices and scores.

        Returns:
            None
        """
        # Read the CSV dataset info using the provided CSV path
        dataset = pd.read_csv(self.dataset_csv_path)
        
        # Create a list of image paths by combining the crop folder path with each cropped image name
        list_of_image_paths = dataset["cropped_image_name"].apply(
            lambda x: (self.crop_folder / x).as_posix()).to_list()
        
        # Create a random number generator with a specified seed (for reproducibility)
        rng = np.random.default_rng(seed=42)
        # Generate random indices to select query images
        query_img_index = rng.integers(low=0, high=len(list_of_image_paths), size=self.num_queries)
        # Retrieve the corresponding query image features using the generated indices   
        query_images = [list_of_image_paths[index] for index in query_img_index.tolist()]

        # Initialize ViT feature extractor and transformations
        extractor = ViTExtractor.from_pretrained(self.pretrained_dataset)
        extractor = extractor.to("mps")  # Apple M1 GPU
        transform, im_reader = get_transforms_for_pretrained(self.pretrained_dataset)

        args = {"num_workers": 0, "batch_size": 128}
        # Extract features from query images
        query_features = inference_on_images(extractor, paths=query_images, transform=transform, **args)
        
        dataset_features_file_name = f"dataset_features_{self.pretrained_dataset}_ViT.pt"
        if not os.path.isfile((self.data_folder / dataset_features_file_name).as_posix()):
            dataset_features = inference_on_images(extractor, paths=list_of_image_paths, transform=transform, **args)
            # Save the dataset features as a .pt file
            torch.save(dataset_features, self.data_folder / dataset_features_file_name)
        else:
            dataset_features = torch.load(self.data_folder / dataset_features_file_name)
        
        # Calculate cosine similarity between all query images and dataset features
        pairwise_similarities = cosine_similarity(query_features.numpy(), dataset_features.numpy())

        # Find indices of top k similar images for each query
        top_indices = np.argsort(pairwise_similarities, axis=1)[:, -self.top_k:][:, ::-1]
        top_similarities = np.take_along_axis(pairwise_similarities, top_indices, axis=1)
        
        # Create an instance of the ImageSimilarityVisualizer class for visualizing results
        visualizer = ImageSimilarityVisualizer(top_indices,
                                               top_similarities,
                                               list_of_image_paths,
                                               query_img_index,
                                               dataset)
        visualizer.visualize_results()
