import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import argparse
import glob
import json
dict = {0:'FGSM',1:'PGD',2:'CW',3:'Square',4:'AutoAttack',5:'ZO-SignAttack'}
class ImagePredictor:
    def __init__(self, model_path, device="cpu"):
        self.device = device
        self.model = self._load_model(model_path)
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])
        
    def _load_model(self, model_path):
        model = models.resnet18(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, 6)  # 6 attack classes
        model.load_state_dict(torch.load(model_path))
        return model.to(self.device)
    
    def predict_image(self, image_path):
        image = Image.open(image_path).convert('RGB')
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            outputs = self.model(tensor)
            _, pred = torch.max(outputs, 1)
        return pred.item()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-folder", required=True, help="Path to testing folder")
    parser.add_argument("--model-path", required=True, help="Path to model weights")
    parser.add_argument("--output", default="predictions.json", help="Output file name")
    args = parser.parse_args()

    # Initialize predictor
    predictor = ImagePredictor(args.model_path)

    # Get all images in folder
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(glob.glob(os.path.join(args.input_folder, ext)))

    if not image_paths:
        print(f"No images found in {args.input_folder}")
        return

    # Process images
    results = []
    for img_path in image_paths:
        try:
            pred = predictor.predict_image(img_path)
            results.append({
                "image": os.path.basename(img_path),
                "prediction": pred
            })
            print(f"Processed {img_path} - Prediction: {dict[pred]}")
        except Exception as e:
            print(f"Error processing {img_path}: {str(e)}")

    # Save results
    output_path = os.path.join(args.input_folder, args.output)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Predictions saved to {output_path}")

if __name__ == "__main__":
    main()