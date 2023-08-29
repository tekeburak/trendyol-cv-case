import numpy as np
import pandas as pd
import os
import torch
import tensorflow as tf
from pathlib import Path
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.metrics.pairwise import cosine_similarity
from utils.visualizer import ImageSimilarityVisualizer

import faiss
from faiss import write_index, read_index


class ImageSimilarityResNet:
    def __init__(self,
                 crop_folder: str,
                 dataset_csv_path: str,
                 method: str,
                 num_queries: int = 10,
                 top_k: int = 12):
        """
        Initialize the ImageSimilarityResNet.

        Args:
            crop_folder (str): Path to the folder containing images.
            dataset_csv_path (str): Path to the CSV file containing crop information.
            method (str): Method to use for similarity analysis. Can be "faiss" or "cosine".
            num_queries (int, optional): Number of query images to use. Defaults to 10.
            top_k (int, optional): Number of top similar images to retrieve. Defaults to 12.
        """
        self.base_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
        self.crop_folder = Path(crop_folder)
        self.dataset_csv_path = Path(dataset_csv_path)
        self.method = method
        self.num_queries = num_queries
        self.top_k = top_k
        
        self.data_folder = self.dataset_csv_path.parent / "ResNet"
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
        
        if not os.path.isfile((self.data_folder / 'dataset_features_resnet50.npy').as_posix()):
            # Initialize an empty list to store dataset crop image features
            dataset_features = []

            # Define the batch size for inference
            batch_size = 128
            
            # Iterate through the image paths in batches of specified size
            for i in range(0, len(list_of_image_paths), batch_size):
                # Get a batch of image paths
                batch_image_paths = list_of_image_paths[i : i + batch_size if 
                                                        i + batch_size <= len(list_of_image_paths) 
                                                        else len(list_of_image_paths)]
                
                # Initialize an empty list to store preprocessed images for this batch
                preprocessed_images = []
        
                # Loop through each image path in the batch and preprocess the images
                for image_path in batch_image_paths:
                    # Load the image and resize it to the target size (224x224)
                    image = load_img(image_path, target_size=(224, 224))
                    
                    # Convert the image to an array and expand dimensions to create a batch of size 1
                    image = img_to_array(image)
                    image = np.expand_dims(image, axis=0)
                    
                    # Preprocess the image according to ResNet50's preprocessing requirements
                    image = tf.keras.applications.resnet50.preprocess_input(image)
                    
                    # Append the preprocessed image to the list
                    preprocessed_images.append(image)
                
                # Stack the preprocessed images to create a batch for inference
                batch_images = np.vstack(preprocessed_images)
                
                # Perform inference on the batch of images using the ResNet50 model
                batch_features = self.base_model.predict(batch_images)
                
                # Append the batch features to the dataset features list
                dataset_features.extend(batch_features)

            # Convert the list of dataset features to a NumPy array
            dataset_features = np.array(dataset_features)
            
            # Save the dataset features as a .npy file
            np.save(self.data_folder / 'dataset_features_resnet50.npy',
                    dataset_features)
        else:
            dataset_features = np.load(self.data_folder / 'dataset_features_resnet50.npy')
        
        # Create a random number generator with a specified seed (for reproducibility)
        rng = np.random.default_rng(seed=42)
        # Generate random indices to select query images
        query_img_index = rng.integers(low=0, high=len(list_of_image_paths), size=self.num_queries)
        # Retrieve the corresponding query image features using the generated indices
        query_features = dataset_features[query_img_index]
        
        if self.method == "faiss":
            if not os.path.isfile((self.data_folder / "faiss_resnet_large.index").as_posix()):
                dataset_features_faiss = np.copy(dataset_features)
                query_features_faiss = np.copy(query_features)
                dimensionality = dataset_features_faiss.shape[1]
                
                # Initialize an index using Faiss with the specified dimensionality and metric
                index = faiss.index_factory(dimensionality, "Flat", faiss.METRIC_INNER_PRODUCT)
                # Normalize dataset features for use in Faiss
                faiss.normalize_L2(dataset_features_faiss)
                # Add dataset features to the index
                index.add(dataset_features_faiss)
                # Write the index to a file
                write_index(index,
                            (self.data_folder / "faiss_resnet_large.index").as_posix())
                # Normalize query features for use in Faiss
                faiss.normalize_L2(query_features_faiss)

            else:
                # Read the Faiss index from a file
                index = read_index((self.data_folder / "faiss_resnet_large.index").as_posix())
            
            # Search the Faiss index to retrieve top similar images
            top_similarities, top_indices = index.search(query_features_faiss, self.top_k)
            
        elif self.method == "cosine":
            # Calculate cosine similarity between all query images and dataset features
            pairwise_similarities = cosine_similarity(query_features, dataset_features)

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