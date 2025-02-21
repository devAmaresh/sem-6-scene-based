import os
from PIL import Image
from vector_emb import get_best_match
from object_crop import crop_and_save_objects
from prompt_chain import multi_level_cot
from draw_box import draw_box_on_image
from blip2.blip2_inference import generate_features

import cv2

inputImagePath = "inputImage.jpg"
outputImagePath = "annotated_image.jpg"
logFilePath = "coordinatesLogFile.txt"

def annotateInputImage(image_path, file_path, index, output_path, box_color=(0, 255, 0), box_label=None):
    
    coordinates = None
    with open(file_path, "r") as file:
        for line in file:
            if line.startswith(f"{index}\t"):
                _, coord_str = line.strip().split("\t")
                coordinates = tuple(map(int, coord_str.strip("()").split(", ")))
                break

    if coordinates is None:
        print(f"No coordinates found for index {index}")
        return

    image = cv2.imread(image_path)
    if image is None:
        print(f"Error loading image: {image_path}")
        return

    x1, y1, x2, y2 = coordinates

    cv2.rectangle(image, (x1, y1), (x2, y2), box_color, thickness=2)

    if box_label:
        label_position = (x1, y1 - 10 if y1 - 10 > 10 else y1 + 10)
        cv2.putText(
            image,
            box_label,
            label_position,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            box_color,
            thickness=1,
            lineType=cv2.LINE_AA
        )

    cv2.imwrite(output_path, image)
    print(f"Annotated image saved to: {output_path}")

import ast

def main():
    path = input("enter the path of the image: ")
    output_dir = "cropped_objects"
    # os.makedirs(output_dir, exist_ok=True)
    crop_and_save_objects(path, output_dir)
    prompt = input("Enter the task which needs to be done:")

    required_desc = multi_level_cot(prompt)
    folder_path = "./cropped_objects"

    # Iterate through the folder
    captions = {}
    if os.path.exists(folder_path) and os.path.isdir(folder_path):
        for filename in os.listdir(folder_path):
            if filename.lower().endswith(".jpg"):
                relative_path = os.path.join(folder_path, filename)
                text_desc = generate_features(relative_path)
                num = filename[filename.find("_") + 1 : filename.find(".jpg")]
                captions[text_desc] = num

    print(captions)
    req_ind = get_best_match(list(captions.keys()), required_desc)
    data = open("coordinates.txt")
    coord_dict = ast.literal_eval(data.read())
    data.close()
    coordinates = coord_dict[captions[req_ind]]
    draw_box_on_image(path, coordinates, "selected.jpeg")



if __name__ == "__main__":
    main()
