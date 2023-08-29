from typing import List
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

class ImageSimilarityVisualizer:
    def __init__(self,
                 index_closest: np.ndarray,
                 scores: float,
                 crop_img_list: List[str],
                 query_img_index: List[int],
                 dataset_csv: pd.DataFrame):
        """
        Initialize the ImageSimilarityVisualizer.

        Args:
            index_closest (torch.Tensor or np.ndarray): Indices of closest images for each query.
            scores (np.ndarray): Similarity scores for top similar images.
            crop_img_list (List[str]): Paths to gallery images.
            query_img_index (List[int]): Indices of query images.
            dataset_csv (pd.DataFrame): DataFrame containing dataset information.
        """
        self.index_closest = index_closest
        self.scores = scores
        self.crop_img_list = crop_img_list
        self.query_img_index = query_img_index
        self.dataset_csv = dataset_csv
        self.out_folder = Path("results")

    def visualize_results(self) -> None:
        """
        Visualize query and gallery images for image similarity task.

        Returns:
            None
        """
        # Read the crop label indices from the CSV dataset
        crop_label_ids = self.dataset_csv["crop_label_index"]
        # Create a figure to plot the images
        plt.figure(figsize=(15, 8))
        
        for index, result_crop_IDs in enumerate(self.index_closest):
            query_img_no = self.query_img_index[index]
            query_img_path = self.crop_img_list[query_img_no]
            query_img = Image.open(query_img_path)
            query_img_label_id = crop_label_ids[query_img_no]
            
            # Clear the previous plot and create a subplot for the query image
            plt.clf()
            plt.subplot(5, 6, 1)
            plt.imshow(query_img)
            plt.title(f"Input | id={query_img_label_id}", fontweight='bold')
            plt.tight_layout(pad=2.25)
            
            # Create a subplot for displaying the output
            subplot = plt.subplot(5, 6, 7)
            subplot.axis('off')
            subplot.text(0.5, 0.5, 'Output: ', ha='center', va='center',
                         fontsize=12, fontweight='bold', transform=subplot.transAxes)
            
            # Iterate over the indices and label IDs of top similar crop images for the current query image
            for idx, result_crop_id in enumerate(result_crop_IDs):
                result_crop_img_path = self.crop_img_list[result_crop_id]
                pred_crop_label_id = crop_label_ids[result_crop_id]
                result_crop_img = Image.open(result_crop_img_path)
                plt.subplot(5, 6, idx + 13)
                plt.subplots_adjust(hspace=0.5)
                plt.title(f"{self.scores[index, idx]:.4f} | id={pred_crop_label_id}")
                plt.imshow(result_crop_img)
                
            # Save the visualization to a PDF file
            result_path = self.out_folder / f"result_query_{index+1}.pdf"
            plt.savefig(result_path.as_posix(), format="pdf")
            
        print(f"Results are saved in {self.out_folder} folder...")