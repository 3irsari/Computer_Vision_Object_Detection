# Import necessary libraries
import os
import json
import shutil
import random
from PIL import Image,ImageDraw
import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F
import torchvision.transforms as T
from torch.utils.data import DataLoader,Dataset
from tqdm import tqdm

# Load Faster R-CNN model
model = fasterrcnn_resnet50_fpn(pretrained=True)
model.train()  # Set to training mode

# Custom dataset class for loading images and annotations
class SkyFusionDataset(Dataset):
    def __init__(self,annotation_file,image_dir,transform=None):
        self.image_dir = image_dir
        self.transform = transform

        # Load COCO annotations
        with open(annotation_file) as f:
            coco_data = json.load(f)

        self.images = {img['id']: img for img in coco_data['images']}
        self.annotations = coco_data['annotations']
        self.categories = {cat['id']: cat['name'] for cat in coco_data['categories']}

        # Filter for desired categories
        self.airplane_category_id = [cat['id'] for cat in coco_data['categories'] if cat['name'].lower() in categories]
        self.filtered_annotations = [ann for ann in self.annotations if ann['category_id'] in self.airplane_category_id]


      # Remove annotations with missing images
        self.filtered_annotations = [
            ann for ann in self.filtered_annotations if
            os.path.exists(os.path.join(self.image_dir,self.images[ann['image_id']]['file_name']))
        ]

    def __len__(self):
        return len(self.filtered_annotations)

    def __getitem__(self,idx):
        annotation = self.filtered_annotations[idx]
        img_id = annotation['image_id']
        img_info = self.images[img_id]
        img_path = os.path.join(self.image_dir,img_info['file_name'])

        # Load the image if it exists
        try:
            image = Image.open(img_path).convert("RGB")
        except FileNotFoundError:
            print(f"Warning: Image file {img_info['file_name']} not found, skipping.")
            return None,None  # or raise an exception if desired

        # Apply transformations
        if self.transform:
            image = self.transform(image)

        # Format bounding box data
        box = annotation['bbox']
        target = {
            "boxes": torch.tensor([box],dtype=torch.float32),
            "labels": torch.tensor([1],dtype=torch.int64),  # 1 for airplane
        }

        return F.to_tensor(image),target

    # Data loader modified to skip None entries
    def collate_fn(batch):
        batch = [item for item in batch if item[0] is not None]  # Skip None entries
        return tuple(zip(*batch))

# Define augmentations (only for training data)
augmentations = T.Compose([
    T.RandomHorizontalFlip(p=0.5),
    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    T.RandomRotation(degrees=10)
])

# Initialize datasets and data loaders
train_dataset = SkyFusionDataset(
    'SkyFusion/train/_annotations.coco.json',
    'SkyFusion/train/',
    transform=augmentations,
    augment=True  # Enable augmentations for training data
)
# Initialize datasets and data loaders

train_loader = DataLoader(train_dataset,batch_size=4,shuffle=True,collate_fn=collate_fn)

# Define optimizer and hyperparameters
optimizer = torch.optim.SGD(model.parameters(),lr=0.005,momentum=0.9,weight_decay=0.0005)
num_epochs = 10

# Initialize datasets and data loaders

# Training loop
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    for images,targets in tqdm(train_loader,desc=f"Epoch {epoch + 1}/{num_epochs}"):
        images = list(image for image in images)
        targets = [{k: v for k,v in t.items()} for t in targets]

        optimizer.zero_grad()
        loss_dict = model(images,targets)
        losses = sum(loss for loss in loss_dict.values())
        losses.backward()

        optimizer.step()
        epoch_loss += losses.item()

    print(f"Epoch {epoch + 1}, Loss: {epoch_loss / len(train_loader):.4f}")


# Evaluation function
def evaluate_model(data_loader):
    model.eval()
    total_true_positives = 0
    total_false_positives = 0
    total_false_negatives = 0

    with torch.no_grad():
        for images,targets in tqdm(data_loader,desc="Evaluating"):
            images = list(image for image in images)
            targets = [{k: v for k,v in t.items()} for t in targets]
            predictions = model(images)

            # Evaluate predictions (assumes one object per image)
            for i in range(len(images)):
                gt_boxes = targets[i]["boxes"]
                pred_boxes = predictions[i]["boxes"]

                if pred_boxes.shape[0] > 0:
                    # Assume True Positive if IoU >= 0.5
                    iou = calculate_iou(pred_boxes[0].tolist(),gt_boxes[0].tolist())
                    if iou >= 0.5:
                        total_true_positives += 1
                    else:
                        total_false_positives += 1
                else:
                    total_false_negatives += 1

    precision = total_true_positives / (total_true_positives + total_false_positives) \
        if ( total_true_positives + total_false_positives) > 0 else 0
    recall = total_true_positives / (total_true_positives + total_false_negatives) \
        if ( total_true_positives + total_false_negatives) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    print(f"Precision: {precision:.2f}, Recall: {recall:.2f}, F1 Score: {f1_score:.2f}")
    return precision,recall,f1_score


# Initialize validation dataset and data loader
valid_dataset = SkyFusionDataset('SkyFusion/valid/_annotations.coco.json','SkyFusion/valid/')
valid_loader = DataLoader(valid_dataset,batch_size=1,shuffle=False,collate_fn=lambda x: tuple(zip(*x)))

# Run evaluation on validation set
evaluate_model(valid_loader)

# Compress the 'faster_R-CNN' directory
shutil.make_archive('faster_R-CNN_final','zip','SkyFusion/augmented/')
print("faster_R-CNN dataset compressed and saved as faster_R-CNN_final.zip")

