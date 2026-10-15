import os
import cv2
import pickle
import numpy as np

input_dir = 'data/giz/images/val/'
true_mask_dir = 'data/giz/masks/val/'
output_dir = 'data/giz/predictions'
os.makedirs(output_dir, exist_ok=True)

def burn_in_contours(binary_mask, mode="None"):
    # Extract contours
    contours, _ = cv2.findContours(binary_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter out the small masks
    long_contours = filter(lambda x: cv2.arcLength(x, closed=True) > 500, contours)
    
    # Take the convex hulls
    convex_hulls = [np.array(cv2.convexHull(contour)) for contour in long_contours]
    
    # Select appropriate color
    if mode == "TRUE":
        color = (0, 255, 0)
    elif mode == "PREDICTED":
        color = (255, 165, 0)
    else:
        color = (255, 255, 255)
    
    # Burn in the contours
    cv2.drawContours(image, convex_hulls, -1, color, thickness=4)

for file_name in os.listdir(input_dir):
    if file_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif')):
        
        just_name, just_extension = os.path.splitext(file_name)
        
        image_path = os.path.join(input_dir, file_name)
        mask_path = os.path.join(output_dir, f"{just_name}_mask_0.pkl")
        true_mask_path = os.path.join(true_mask_dir, f"{just_name}.png")

        if os.path.exists(mask_path):
            image = cv2.imread(image_path)
            
            ## Burn in the predicted contours
            with open(mask_path, 'rb') as f:
                mask_data = pickle.load(f)
                
            for p, binary_mask in enumerate(mask_data):
                burn_in_contours(binary_mask, "PREDICTED")
                
            output_path = os.path.join(output_dir, just_name + "_predicted" + just_extension)
            cv2.imwrite(output_path, image)
            
            ## Burn in the ground truth contours
            true_mask_data = cv2.imread(true_mask_path)
            true_unique_classes = np.unique(true_mask_data)
            true_binary_masks = [(true_mask_data == cls).astype(np.uint8)[:, :, 0] for cls in true_unique_classes if cls != 0]
            
            for p, binary_mask in enumerate(true_binary_masks):
                burn_in_contours(binary_mask, "TRUE")
            
            output_path = os.path.join(output_dir, just_name + "_predicted_truth" + just_extension)
            cv2.imwrite(output_path, image)
            
            # Uncomment to see the results!
            # cv2.imshow(f"Burned-in Prediction {file_name}", image)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
