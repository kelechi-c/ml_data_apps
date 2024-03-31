"""Streamlit app for object detection using the DetrForObjectDetection model.

This app allows the user to select an image file and run the model to detect objects in the image.
The detected objects and their certainties are displayed in the Streamlit interface.
"""

import streamlit as st
from PIL import Image
from transformers import DetrImageProcessor, DetrForObjectDetection
import torch

# Configure the page
st.set_page_config(
    page_title="Object Detection",
    page_icon=":robot:",
)

# Add a title to the page
st.title("Object Detection(Images)")

# Upload an image file
image_file = st.file_uploader(label="Select image", type=["jpg", "jpeg", "png"])

# Initialize the model and processor
"""
Initialize the DetrForObjectDetection model and its processor.
The model is loaded from the 'facebook/detr-resnet-50' model, which is a ResNet-50 model trained on the COCO dataset.
"""
processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50", revision="no_timm")
@st.cache_resource
def load_model():
    """Load the DetrForObjectDetection model from pre-trained checkpoint."""
    model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50", revision="no_timm")
    return model

detection_model = load_model()



if image_file:
    # Load the image
    image = Image.open(image_file)
    # Preprocess the image
    inputs = processor(images=image, return_tensors='pt')
    # Run the model
    if st.button('Run detection model'):
        with st.spinner('Running model...'):
            outputs = detection_model(**inputs)
            # Post-processing
            target_sizes = torch.tensor([image.size[::-1]])
            results = processor.post_process_object_detection(outputs=outputs, threshold=0.9, target_sizes=target_sizes)[0]

            # Display the original image
            st.image(image)

            # Extract the predicted objects and certainties
            objects = [detection_model.config.id2label[label].lower() for label in results['labels']]
            certainties = [round(score.item()*100, 2) for score in results['scores']]
            predictions = zip(objects, certainties)

            # Display the detected objects and certainties
            output = st.chat_message('assistant')
            for object, certainty in predictions:
                output.write(f'{object} present with {certainty}% certainty')

