import os
import json
import shutil
from PIL import Image,ImageDraw
import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F

# Load Faster R-CNN model
model = fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()

# Define desired categories for filtering
categories = ['airplane','aircraft']

# List of paths to the annotation files and corresponding image directories
dataset_splits = [
    {'name': 'train','annotation_file': 'SkyFusion/train/_annotations.coco.json','input_image_dir': 'SkyFusion/train/'},
    {'name': 'valid','annotation_file': 'SkyFusion/valid/_annotations.coco.json','input_image_dir': 'SkyFusion/valid/'},
    {'name': 'test','annotation_file': 'SkyFusion/test/_annotations.coco.json','input_image_dir': 'SkyFusion/test/'}
]

# Define base paths for output
base_output_dir = 'SkyFusion/faster_R-CNN'
output_with_detections_dir = os.path.join(base_output_dir,'detections')
output_without_detections_dir = os.path.join(base_output_dir,'no_detections')


# Function to calculate IoU
def calculate_iou(box1,box2):
    x1,y1,w1,h1 = box1
    x2,y2,w2,h2 = box2

    # Calculate overlap coordinates
    x_left = max(x1,x2)
    y_top = max(y1,y2)
    x_right = min(x1 + w1,x2 + w2)
    y_bottom = min(y1 + h1,y2 + h2)

    if x_right < x_left or y_bottom < y_top:
        return 0.0  # No overlap

    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - intersection_area

    return intersection_area / union_area


# Loop through dataset splits (train, valid, test)
for split in dataset_splits:
    split_name = split['name']
    annotation_file = split['annotation_file']
    input_image_dir = split['input_image_dir']

    # Define paths for split-specific images and labels in both folders
    output_image_dir_with_detections = os.path.join(output_with_detections_dir,split_name,'images')
    output_label_dir_with_detections = os.path.join(output_with_detections_dir,split_name,'labels')
    output_image_dir_without_detections = os.path.join(output_without_detections_dir,split_name,'images')
    output_label_dir_without_detections = os.path.join(output_without_detections_dir,split_name,'labels')

    # Ensure output directories exist
    os.makedirs(output_image_dir_with_detections,exist_ok=True)
    os.makedirs(output_label_dir_with_detections,exist_ok=True)
    os.makedirs(output_image_dir_without_detections,exist_ok=True)
    os.makedirs(output_label_dir_without_detections,exist_ok=True)

    # Load the COCO annotations file
    with open(annotation_file) as f:
        coco_data = json.load(f)

    # Find category_id for "airplane" or "aircraft"
    airplane_category_id = None
    for category in coco_data['categories']:
        if category['name'].lower() in categories:
            airplane_category_id = category['id']
            break

    if airplane_category_id is None:
        print(f"Category not found in {annotation_file}. Skipping...")
        continue

    # Dictionary to store annotations for each image containing airplanes
    annotations_dict = {}

    # Filter annotations for the desired category
    for annotation in coco_data['annotations']:
        if annotation['category_id'] == airplane_category_id:
            image_id = annotation['image_id']
            if image_id not in annotations_dict:
                annotations_dict[image_id] = []
            annotations_dict[image_id].append(annotation)

    # Initialize evaluation metrics
    total_true_positives = 0
    total_false_positives = 0
    total_false_negatives = 0

    # Process images
    for image in coco_data['images']:
        image_id = image['id']
        src_image_path = os.path.join(input_image_dir,image['file_name'])

        # Run Faster R-CNN detection
        img = Image.open(src_image_path).convert("RGB")
        img_tensor = F.to_tensor(img).unsqueeze(0)
        with torch.no_grad():
            predictions = model(img_tensor)

        # Filter detections for airplanes (class 5 in COCO dataset)
        airplane_detections = predictions[0]['boxes'][predictions[0]['labels'] == 5].cpu().numpy()

        if len(airplane_detections) > 0 or image_id in annotations_dict:
            dst_image_path = os.path.join(output_image_dir_with_detections,image['file_name'])
        else:
            dst_image_path = os.path.join(output_image_dir_without_detections,image['file_name'])

        try:
            shutil.copy(src_image_path,dst_image_path)
        except Exception as e:
            print(f"Error copying {src_image_path} to {dst_image_path}: {e}")
            continue

        # Open image to draw bounding boxes
        with Image.open(src_image_path) as img:
            draw = ImageDraw.Draw(img)
            img_width,img_height = img.size

            yolo_labels = []

            # Draw and process Faster R-CNN detections
            for det in airplane_detections:
                x1,y1,x2,y2 = det
                x_center = (x1 + x2) / 2 / img_width
                y_center = (y1 + y2) / 2 / img_height
                width = (x2 - x1) / img_width
                height = (y2 - y1) / img_height

                draw.rectangle([x1,y1,x2,y2],outline="blue",width=2)
                yolo_labels.append(f"0 {x_center} {y_center} {width} {height}\n")

            # Draw ground truth boxes
            if image_id in annotations_dict:
                for annotation in annotations_dict[image_id]:
                    bbox = annotation['bbox']
                    x_min,y_min,width,height = bbox
                    x_max = x_min + width
                    y_max = y_min + height

                    draw.rectangle([x_min,y_min,x_max,y_max],outline="red",width=2)

            # Save the image with bounding boxes
            img.save(dst_image_path)

            # Write YOLO format labels
            label_dir = output_label_dir_with_detections if len(
                yolo_labels) > 0 else output_label_dir_without_detections
            yolo_label_path = os.path.join(label_dir,image['file_name'].replace('.jpg','.txt'))
            with open(yolo_label_path,'w') as label_file:
                label_file.writelines(yolo_labels)

            # Evaluation
            if image_id in annotations_dict:
                ground_truth_boxes = annotations_dict[image_id]
                detected_ground_truths = set()

                for det in airplane_detections:
                    x1,y1,x2,y2 = det
                    detected_box = [x1,y1,x2 - x1,y2 - y1]

                    for i,gt_annotation in enumerate(ground_truth_boxes):
                        gt_bbox = gt_annotation['bbox']
                        iou = calculate_iou(detected_box,gt_bbox)

                        if iou >= 0.5 and i not in detected_ground_truths:
                            total_true_positives += 1
                            detected_ground_truths.add(i)
                            break
                    else:
                        total_false_positives += 1

                total_false_negatives += len(ground_truth_boxes) - len(detected_ground_truths)

    # Calculate evaluation metrics for each split
    precision = total_true_positives / (
                total_true_positives + total_false_positives) if total_true_positives + total_false_positives > 0 else 0
    recall = total_true_positives / (
                total_true_positives + total_false_negatives) if total_true_positives + total_false_negatives > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0

    print(f"Processed annotations for {split_name}")
    print(f"Precision: {precision:.2f}, Recall: {recall:.2f}, F1 Score: {f1_score:.2f}")
    print(f"Total true positives: {total_true_positives}")
    print(f"Total false negatives: {total_false_negatives}")
    print(f"Total false positives: {total_false_positives}")

# Compress the 'faster_R-CNN' directory
shutil.make_archive('faster_R-CNN_dataset','zip','SkyFusion/faster_R-CNN')
print("faster_R-CNN dataset compressed and saved as faster_R-CNN_dataset.zip")
