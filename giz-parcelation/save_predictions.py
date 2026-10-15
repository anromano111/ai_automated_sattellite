import os
import pickle
from ultralytics import YOLO
import cv2

# Load the model
model = YOLO('/runs/segment/train9/weights/best.pt')

# Define the directories for input images and output files
input_dir = 'data/images/val/'  # Update this path
output_dir = 'data/predictions'  # Update this path
os.makedirs(output_dir, exist_ok=True)

# Run inference on all images in the directory
for file_name in os.listdir(input_dir):
    if file_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif')):
        file_path = os.path.join(input_dir, file_name)
        print(f"Processing: {file_path}")

        # Perform inference
        results = model.predict(source=file_path, save=False)

        # Save the predictions
        for idx, result in enumerate(results):
            # Save overlaid image
            overlay_path = os.path.join(output_dir, f"{os.path.splitext(file_name)[0]}_result_{idx}.jpg")
            img = result.plot()  # Overlay predictions on the image
            cv2.imwrite(overlay_path, img)
            print(f"Saved overlaid image: {overlay_path}")

            # Save the masks as a pickle file
            mask_path = os.path.join(output_dir, f"{os.path.splitext(file_name)[0]}_mask_{idx}.pkl")
            with open(mask_path, 'wb') as f:
                pickle.dump(result.masks.data.cpu().numpy(), f)  # Save the mask data as a NumPy array
            print(f"Saved mask as pickle: {mask_path}")

print("Inference and saving completed.")
