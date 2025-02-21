import os
from blip2.blip2_inference import generate_features

folder_path = './cropped_objects'
if os.path.exists(folder_path) and os.path.isdir(folder_path):
    for filename in os.listdir(folder_path):
        if filename.lower().endswith('.jpg'):
            relative_path = os.path.join(folder_path, filename)
            text_desc = generate_features(relative_path)
            print(text_desc)