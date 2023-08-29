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