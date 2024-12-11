from PIL import Image
import os

input_dir = 'art/'  # Directory with original large images
output_dir = 'art_resized/'  # Directory where resized images will be saved

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Function to resize while maintaining aspect ratio
def resize_with_aspect_ratio(image, size=500):
    # Get original dimensions
    original_width, original_height = image.size

    # Check which side is longer and scale accordingly
    if original_width > original_height:
        # Landscape orientation: Scale width to 'size'
        new_width = size
        new_height = int((size / original_width) * original_height)
    else:
        # Portrait or square orientation: Scale height to 'size'
        new_height = size
        new_width = int((size / original_height) * original_width)

    return image.resize((new_width, new_height), Image.Resampling.LANCZOS)

# Loop through all images in the input directory
for filename in os.listdir(input_dir):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        # Open the image
        img = Image.open(os.path.join(input_dir, filename))

        # Resize with aspect ratio
        img_resized = resize_with_aspect_ratio(img, size=500)

        # Save the resized image to the output directory
        img_resized.save(os.path.join(output_dir, filename))

print("Images resized successfully to fit within 500 pixels!")