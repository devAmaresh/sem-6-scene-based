import gradio as gr
import os
import ast
import cv2
from PIL import Image
from vector_emb import get_best_match
from object_crop import crop_and_save_objects
from prompt_chain import multi_level_cot
from draw_box import draw_box_on_image
from blip2.blip2_inference import generate_features

# Function to process the image
def process_image(image, task):
    if image is None or not task.strip():
        return "Please upload an image and enter a task description.", None, None, None

    image_path = "uploaded_image.jpg"
    image.save(image_path)  # Save uploaded image

    output_dir = "cropped_objects"
    os.makedirs(output_dir, exist_ok=True)

    crop_and_save_objects(image_path, output_dir)  # Extract objects

    required_desc = multi_level_cot(task)  # Get required description
    folder_path = "./cropped_objects"

    # Generate captions
    captions = {}
    if os.path.exists(folder_path) and os.path.isdir(folder_path):
        for filename in os.listdir(folder_path):
            if filename.lower().endswith(".jpg"):
                relative_path = os.path.join(folder_path, filename)
                text_desc = generate_features(relative_path)
                num = filename[filename.find("_") + 1 : filename.find(".jpg")]
                captions[text_desc] = num

    if not captions:
        return "No objects detected.", None, None, None

    req_ind = get_best_match(list(captions.keys()), required_desc)

    # Read coordinates
    with open("coordinates.txt") as f:
        coord_dict = ast.literal_eval(f.read())

    if captions[req_ind] not in coord_dict:
        return "Error: Selected object not found in coordinates.", None, None, None

    coordinates = coord_dict[captions[req_ind]]

    # Annotate image
    output_annotated = "annotated_image.jpg"
    draw_box_on_image(image_path, coordinates, output_annotated)

    # Save cropped object
    cropped_image_path = f"{output_dir}/selected_object.jpg"
    x1, y1, x2, y2 = coordinates
    image = Image.open(image_path)
    cropped_object = image.crop((x1, y1, x2, y2))
    cropped_object.save(cropped_image_path)

    return (
        "Processing complete. See results below.",
        output_annotated,
        cropped_image_path,
        req_ind,  # Caption of the cropped object
    )

# Function to enable submit button only when both image and text are provided
def enable_submit(image, task):
    return gr.update(interactive=bool(image and task.strip()))  # This properly updates the button state

# Define Gradio interface
with gr.Blocks() as interface:
    gr.Markdown("## Object Annotation Tool")
    gr.Markdown("Upload an image, describe the object to highlight, and get the annotated result.")

    with gr.Row():
        image_input = gr.Image(type="pil", label="Upload an Image")
        task_input = gr.Textbox(label="Describe the task", placeholder="e.g., Highlight the red apple")

    submit_btn = gr.Button("Submit", interactive=False)

    status_output = gr.Textbox(label="Status")
    annotated_output = gr.Image(type="filepath", label="Annotated Image")
    cropped_output = gr.Image(type="filepath", label="Cropped Object")
    caption_output = gr.Textbox(label="Caption", interactive=False)

    # Enable submit button only when both inputs are given
    image_input.change(enable_submit, inputs=[image_input, task_input], outputs=[submit_btn])
    task_input.change(enable_submit, inputs=[image_input, task_input], outputs=[submit_btn])

    submit_btn.click(
        process_image,
        inputs=[image_input, task_input],
        outputs=[status_output, annotated_output, cropped_output, caption_output]
    )

if __name__ == "__main__":
    interface.launch()
