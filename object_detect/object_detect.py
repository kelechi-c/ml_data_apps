from transformers import DetrForObjectDetection, DetrImageProcessor
import torch
import streamlit as st
from PIL import Image

st.set_page_config(
    page_title="Object Detection",
    page_icon=":robot:"
)

st.title("Object Detection(Images)")

image_upload = st.file_uploader(label="Select image", type=["jpg", "jpeg", "png"])

processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50", revision="no_timm")
@st.cache_resource
def load_model():
    model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50", revision="no_timm")
    return model

detection_model = load_model()



if image_upload:
    image_upload = Image.open(image_upload)
    model_input = processor(images=image_upload, return_tensors='pt')
    if st.button('Run detection model'):
        with st.spinner('Running model...'):
            model_output = detection_model(**model_input)

            target_sizes = torch.tensor([image_upload.size[::--1]])
            results = processor.post_process_object_detection(outputs=model_output, threshold=0.9, target_sizes=target_sizes)[0]
            
            
            st.image(image_upload)
            
            object_list = []
            certainties = []

            for score, label in zip(results['scores'], results['labels']):
                object_list.append(detection_model.config.id2label[label.item()])
                certainties.append(100 * round(score.item(), 3))
            
            predictions = zip(object_list, certainties)
            output = st.chat_message('assistant')
            output.write('Detected objects(Certainty of prediction):')
            for object, certainty in predictions:
                output.write(f'{object.upper()}->({certainty:.2f}%)')
            
        