# import streamlit as st
# from transformers import VisionEncoderDecoderModel, ViTFeatureExtractor, AutoTokenizer
# import torch
# from PIL import Image
# import os

# #put a title on the browser tab
# st.set_page_config(page_title='Image Captioning App', layout='centered', initial_sidebar_state='auto')

# # Set paths for saving/loading the model and tokenizer
# model_path = "./model"
# tokenizer_path = "./tokenizer"

# # Check if model and tokenizer checkpoints exist
# model_checkpoint_exists = os.path.isfile(model_path)
# tokenizer_checkpoint_exists = os.path.isfile(tokenizer_path)

# # Load the model and tokenizer or create new instances
# if model_checkpoint_exists and tokenizer_checkpoint_exists:
#     model = VisionEncoderDecoderModel.from_pretrained(model_path)
#     tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
# else:
#     model_name = "nlpconnect/vit-gpt2-image-captioning"
#     model = VisionEncoderDecoderModel.from_pretrained(model_name)
#     tokenizer = AutoTokenizer.from_pretrained(model_name)
#     model.save_pretrained(model_path)
#     tokenizer.save_pretrained(tokenizer_path)

# # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# # model.to(device)

# # Force the use of CPU if on Streamlit Cloud
# device = torch.device("cpu")  # Force CPU if not using GPU on Streamlit Cloud
# model.to(device)


# # Set the maximum length and number of beams for caption generation
# max_length = 16
# num_beams = 4
# gen_kwargs = {
#     "max_length": max_length,
#     "num_beams": num_beams,
# }

# # Initialize the feature extractor
# feature_extractor_path = "./feature_extractor"
# feature_extractor = ViTFeatureExtractor.from_pretrained(feature_extractor_path)

# # Streamlit application
# def main():
#     st.title("Image Captioning App")
#     st.write("Upload an image and get its caption")

#     # Image upload
#     uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

#     if uploaded_file is not None:
#         # Display the uploaded image
#         image = Image.open(uploaded_file)
#         st.image(image, caption="Uploaded Image", use_column_width=True)

#         # Generate caption on button click
#         if st.button("Generate Caption"):
#             caption = generate_caption(image)
#             st.success(caption)

# def generate_caption(image):
#     # Preprocess the image
#     if image.mode != "RGB":
#         image = image.convert("RGB")
#     image = image.resize((224, 224))
#     pixel_values = feature_extractor(images=image, return_tensors="pt").pixel_values.to(device)

#     # Generate caption
#     with torch.no_grad():
#         output_ids = model.generate(pixel_values, **gen_kwargs)
#     caption = tokenizer.decode(output_ids[0], skip_special_tokens=True)

#     return caption

# if __name__ == "__main__":
#     main()

import os
import torch
from PIL import Image
import streamlit as st
from transformers import BlipProcessor, BlipForConditionalGeneration
import io

# Load pre-trained model and processor
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

def generate_caption(image_path):
    """Generate image caption from the given image path."""
    try:
        raw_image = Image.open(image_path).convert('RGB')
        inputs = processor(raw_image, return_tensors="pt")

        # Generate caption
        out = model.generate(**inputs)
        caption = processor.decode(out[0], skip_special_tokens=True)
        return caption
    except Exception as e:
        return f"Error in caption generation: {str(e)}"

def main():
    """Main function to handle Streamlit app logic."""
    st.title("BLIP Image Captioning")

    # File upload
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Save uploaded file to disk
        with open("uploaded_image.jpg", "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Display uploaded image
        st.image(uploaded_file, caption="Uploaded Image.", use_column_width=True)
        st.write("")
        
        # Generate caption
        caption = generate_caption("uploaded_image.jpg")
        
        # Display the caption
        st.write(f"Generated Caption: {caption}")

    # Footer information
    st.markdown("This is an AI-driven image captioning tool using BLIP (Bootstrapping Language-Image Pretraining).")

# Run the Streamlit app
if __name__ == "__main__":
    main()
