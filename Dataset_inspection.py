

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os
import random
from collections import Counter

def load_image(path):
    img = Image.open(path).convert('RGB')
    img = img.resize((300,300))  # Resize for consistency
    return np.array(img)

# Function to get random images from a folder
def get_random_images(folder_path,num_images=12):
    image_files = [f for f in os.listdir(folder_path) if
                   f.lower().endswith(('.jpg'))]

    if len(image_files) < num_images:
        print(f"Warning: Only {len(image_files)} images found in the folder.")
        num_images = len(image_files)

    selected_images = random.sample(image_files,num_images)
    return [os.path.join(folder_path,img) for img in selected_images]

def count_files_by_type(folder_path):
    image_types = Counter()
    other_types = Counter()
    image_extensions = {'.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif'}

    for file in os.listdir(folder_path):
        _, ext = os.path.splitext(file)
        ext = ext.lower()
        if ext in image_extensions:
            image_types[ext] += 1
        else:
            other_types[ext if ext else 'no extension'] += 1

    return image_types, other_types

#Train Folder Exploration
# Specify the folder containing your images
folder_path = 'SkyFusion/train'

# Get 12 random image paths
image_paths = get_random_images(folder_path)

# Load the images
images = [load_image(path) for path in image_paths]

# Count files by type
image_counts, other_counts = count_files_by_type(folder_path)

# Create the 3x4 grid and display the images
fig, axes = plt.subplots(3, 4, figsize=(15, 12))
fig.subplots_adjust(hspace=0.3, wspace=0.3)

# Set the title for the entire figure
plt.suptitle('Images from Train Set', fontsize=16)

for i, ax in enumerate(axes.flat):
    if i < len(images):
        ax.imshow(images[i])
        #ax.set_title(os.path.basename(image_paths[i]), fontsize=8)
        ax.axis('off')
    else:
        ax.remove()  # Remove unused subplots

# Add file count information
image_info = "Image file counts:\n" + "\n".join([f"{ext}: {count}" for ext, count in sorted(image_counts.items())])
other_info = "Other file counts:\n" + "\n".join([f"{ext}: {count}" for ext, count in sorted(other_counts.items())])

plt.figtext(0.02, 0.02, image_info, fontsize=10, va="bottom")
plt.figtext(0.5, 0.02, other_info, fontsize=10, va="bottom")

plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to make room for the title
plt.show()


#Test Folder Exploration
# Specify the folder containing your images
folder_path = 'SkyFusion/test'

# Get 12 random image paths
image_paths = get_random_images(folder_path)

# Load the images
images = [load_image(path) for path in image_paths]

# Count files by type
image_counts, other_counts = count_files_by_type(folder_path)

# Create the 3x4 grid and display the images
fig, axes = plt.subplots(3, 4, figsize=(15, 12))
fig.subplots_adjust(hspace=0.3, wspace=0.3)

# Set the title for the entire figure
plt.suptitle('Images from Test Set', fontsize=16)

for i, ax in enumerate(axes.flat):
    if i < len(images):
        ax.imshow(images[i])
        #ax.set_title(os.path.basename(image_paths[i]), fontsize=8)
        ax.axis('off')
    else:
        ax.remove()  # Remove unused subplots

# Add file count information
image_info = "Image file counts:\n" + "\n".join([f"{ext}: {count}" for ext, count in sorted(image_counts.items())])
other_info = "Other file counts:\n" + "\n".join([f"{ext}: {count}" for ext, count in sorted(other_counts.items())])

plt.figtext(0.02, 0.02, image_info, fontsize=10, va="bottom")
plt.figtext(0.5, 0.02, other_info, fontsize=10, va="bottom")

plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to make room for the title
plt.show()


#Validation Folder Exploration
# Specify the folder containing your images
folder_path = 'SkyFusion/valid'

# Get 12 random image paths
image_paths = get_random_images(folder_path)

# Load the images
images = [load_image(path) for path in image_paths]

# Count files by type
image_counts, other_counts = count_files_by_type(folder_path)

# Create the 3x4 grid and display the images
fig, axes = plt.subplots(3, 4, figsize=(15, 12))
fig.subplots_adjust(hspace=0.3, wspace=0.3)

# Set the title for the entire figure
plt.suptitle('Images from Validation Set', fontsize=16)

for i, ax in enumerate(axes.flat):
    if i < len(images):
        ax.imshow(images[i])
        #ax.set_title(os.path.basename(image_paths[i]), fontsize=8)
        ax.axis('off')
    else:
        ax.remove()  # Remove unused subplots

# Add file count information
image_info = "Image file counts:\n" + "\n".join([f"{ext}: {count}" for ext, count in sorted(image_counts.items())])
other_info = "Other file counts:\n" + "\n".join([f"{ext}: {count}" for ext, count in sorted(other_counts.items())])

plt.figtext(0.02, 0.02, image_info, fontsize=10, va="bottom")
plt.figtext(0.5, 0.02, other_info, fontsize=10, va="bottom")

plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to make room for the title
plt.show()