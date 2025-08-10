import os
import cv2
import csv

# Paths
image_folder = r"D:\1. ANPR_CRNN(IMAGES)\data\images"
labeled_output_folder = r"D:\1. ANPR_CRNN(IMAGES)\data\labeled_images"
output_csv = r"D:\1. ANPR_CRNN(IMAGES)\data\labels.csv"

# Create output folder if it doesn't exist
os.makedirs(labeled_output_folder, exist_ok=True)

# Font settings
font = cv2.FONT_HERSHEY_SIMPLEX
initial_scale = 1.2
thickness = 2
color = (0, 255, 0)

# Open CSV to write filename-label pairs
with open(output_csv, mode='w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Filename', 'Label'])  # Header

    # Process each image
    for filename in os.listdir(image_folder):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            label = os.path.splitext(filename)[0]  # Label from filename

            # Write to CSV
            writer.writerow([filename, label])

            # Read and label image
            image_path = os.path.join(image_folder, filename)
            image = cv2.imread(image_path)

            if image is not None:
                img_h, img_w = image.shape[:2]
                font_scale = initial_scale

                # Adjust font scale if text is too wide
                (text_w, text_h), _ = cv2.getTextSize(label, font, font_scale, thickness)
                while text_w > img_w - 20 and font_scale > 0.3:
                    font_scale -= 0.1
                    (text_w, text_h), _ = cv2.getTextSize(label, font, font_scale, thickness)

                # Position the text
                x, y = 10, text_h + 10

                # Draw label on image
                cv2.putText(image, label, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)

                # Save labeled image
                output_path = os.path.join(labeled_output_folder, filename)
                cv2.imwrite(output_path, image)
            else:
                print(f"[WARNING] Failed to read: {image_path}")
