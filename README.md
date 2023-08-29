# Trendyol Image Similarity

## Project Description:

This project focuses on measuring the similarity between two products.

The provided JSON file includes 20,000 fashion product images along with their bounding boxes. The goal is to identify the most similar fashion products within this dataset for a given input image.

The task involves randomly selecting 10 cropped images from the JSON file and identifying the 12 images that are most similar to each of them.

## Installation Instructions:

To install the required dependencies, run the following command:

```bash
pip install -r requirements.txt
```

## Usage Instructions:

1. Specify the necessary paths in `main.py`:
   - `images_folder`: Path to the folder where images will be saved.
   - `export_json_path`: Path to the JSON file containing image and bounding box information.
   - `crop_out_folder`: Path to the folder where cropped images will be saved.
   - `dataset_csv_out`: Path to the CSV file for extracting dataset information.

2. Run the `main.py` script:
   
```bash
python main.py
```

## Features and Functionalities:

This project utilizes the following models for feature extraction:

- **ResNet50** with [ImageNet](https://www.image-net.org) pretrained weights.
- **ViT** with [DeepFashion In-shop](https://mmlab.ie.cuhk.edu.hk/projects/DeepFashion/InShopRetrieval.html) pretrained weights using [Open Metric Learning (OML)](https://github.com/OML-Team/open-metric-learning) library.
- **ViT** with [Stanford Online Products](https://cvgl.stanford.edu/projects/lifted_struct/) pretrained weights using [OML](https://github.com/OML-Team/open-metric-learning).
- For efficient similarity search, [Faiss](https://github.com/facebookresearch/faiss) library is used on ResNet feature vectors, saving feature vectors as indexes and allowing fast search query vectors.

Based on the results, **ViT with DeepFashion In-shop pretrained weights** outperforms other models.

## Example Code Snippets:
```python
from utils.image_cropper import ImageCropper
from utils.image_downloader import ImageDownloader
from utils.resnet_inference import ImageSimilarityResNet
from utils.vit_inference import ImageSimilarityViT


def main():
    
    images_folder = "data/images"
    export_json_path = "data/export.json"
    crop_out_folder = "data/cropped_images"
    dataset_csv_out = "data/dataset_info.csv"
    
    downloader = ImageDownloader(images_folder,
                                 export_json_path)
    downloader.download_images()

    cropper = ImageCropper(images_folder,
                           export_json_path,
                           crop_out_folder,
                           dataset_csv_out)
    cropper.crop_images()
    
    img_sim_resnet = ImageSimilarityResNet(crop_out_folder,
                                           dataset_csv_out,
                                           "faiss")
    img_sim_resnet.analyze_similarity()
    
    img_sim_vit = ImageSimilarityViT(crop_out_folder,
                                     dataset_csv_out,
                                     pretrained_dataset="StanfordOnline")
    img_sim_vit.analyze_similarity()

if __name__ == "__main__":
    main()
```

## Results:

The outcomes of all models have been documented in PDF files, accessible in the [results folder](results/) of this repository. Each PDF contains the query image and its top 12 similar images, providing a visual representation of the image retrieval process. You can click on the links below to view the PDF results:

- [Result for query 1 image using ResNet(ImageNet)](results/ResNet/imagenet/cosine_similarity/result_query_1.pdf) | [Result for query 2 image using ResNet(ImageNet) ](results/ResNet/imagenet/cosine_similarity/result_query_2.pdf)

- [Result for query 1 image using ResNet(ImageNet) along with Faiss](results/ResNet/imagenet/faiss/result_query_1.pdf) | [Result for query 2 image using ResNet(ImageNet) along with Faiss](results/ResNet/imagenet/faiss/result_query_2.pdf)

- [Result for query 1 image using ViT(DeepFashion In-shop)](results/ViT/DeepFashion/cosine_similarity/result_query_1.pdf) | [Result for query 2 image using ViT(DeepFashion In-shop)](results/ViT/DeepFashion/cosine_similarity/result_query_2.pdf)

- [Result for query 1 image using ViT(Stanford Online Products)](results/ViT/StanfordOnline/cosine_similarity/result_query_1.pdf) | [Result for query 2 image using ViT(Stanford Online Products)](results/ViT/StanfordOnline/cosine_similarity/result_query_2.pdf)

Feel free to explore the similarity assessments achieved by each model.

## License Information:

This project is licensed under the [Creative Commons Attribution-NonCommercial 4.0 International License](https://creativecommons.org/licenses/by-nc/4.0/).

[![CC BY-NC 4.0 License](https://i.creativecommons.org/l/by-nc/4.0/88x31.png)](https://creativecommons.org/licenses/by-nc/4.0/)

For more details, please refer to the [LICENSE](LICENSE) file.