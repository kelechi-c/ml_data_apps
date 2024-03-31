from transformers import DetrForObjectDetection, DetrImageProcessor
import torch
import streamlit as st
from PIL import Image

image_upload = st.file_uploader(label="Select image", type=["jpg", "jpeg", "png"])
image_upload = Image.open(image_upload)


processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50", revision="no_timm")

@st.cache_resource
def load_model():
    detection_model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50", revision="no_timm")


load_model()


model_input = processor(images=image_upload, return_tensors='pt')

if st.button('Run detection model'):
    model_output = detection_model(**model_input)

    target_sizes = torch.tensor([image_upload.size[::--1]])
    results = processor.post_process_object_detection(outputs=model_output, threshold=0.9, target_sizes=target_sizes)[0]
    
    
    st.image(image_upload, use_column_width=True)

    for score, label, box in zip(results['scores'], results['labels'], results['boxes']):
        box = [round(x, 2) for x in box.tolist()]
        
        st.write(
                f"Detected {detection_model.config.id2label[label.item()]} with confidence "
                f"{round(score.item(), 3)} at location {box}"
        )
        
    