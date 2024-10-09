import torch
import streamlit as st 
from PIL import Image
import config
import time
import warnings
from getModels import model_init,model_details
from transform import transform_norm
warnings.filterwarnings("ignore")

# UI configurations
st.set_page_config(page_title="ambideXtrous",
                   page_icon=":bridge_at_night:",
                   layout="centered")


# Reverse the dictionary to map indices to class names
index_to_class = config.INV_LABELS_CLASS


models = model_init()


def inference(cnnmodel,image_path):
    
    if(cnnmodel[1] != None):
       
        image = Image.open(image_path).convert('RGB')
        
        cnnmodel[1].eval()
        input_tensor = transform_norm(image).unsqueeze(0)  # Add batch dimension
        
        # Measure inference time
        start_time = time.time()
        
        with torch.no_grad():
            output = cnnmodel[1](input_tensor)
            probs = torch.exp(output)
            # Get the predicted class index
            index = torch.argmax(output).item()
            
        end_time = time.time()
        inference_time = end_time - start_time
            
        predicted_class = index_to_class[index]
        predprob = round(probs[0][index].item(),2)
        if(predprob < 0.80) : predicted_class = 'None'
        
        st.markdown(f"🚀 Model: **{cnnmodel[0]}**")
        model_details(cnnmodel[1])
        st.markdown(f"🌱 Predicted Class: **{predicted_class}**")
        st.markdown(f"🎃 Accuracy: **{predprob:.2f}**")
        st.markdown(f"🌟 Inference Time: **{inference_time:.4f} seconds**")
            
    else:
        st.markdown(f"🚀 Model: **{cnnmodel[0]}**")
        st.markdown(f"☠️ No Model in Production..!!**")

        

def model_card():
    
    imgpath = st.file_uploader("Upload an Image containing Brand Logo", type=["jpg", "jpeg", "png"])
    
    
    if imgpath is not None:
        
        st.image(imgpath, caption="Uploaded Image", use_column_width=True)

        col1, col2, col3, col4 = st.columns(4)

        # Card 1: Model Name
        with col1:
            inference(models[0],image_path=imgpath)
        

        # Card 2: Model Details
        with col2:
            inference(models[1],image_path=imgpath)
            

        # Card 3: Predicted Class
        with col3:
            inference(models[2],image_path=imgpath)
            

        # Card 4: Inference Time
        with col4:
            inference(models[3],image_path=imgpath)
    
    st.write("""
    ### 🚀 **Exploring Image Classification Models with Transfer Learning!**
    - 🆕 **Models Evaluated**: We've assessed the following state-of-the-art models using the Flick27 dataset:
        - **Xception** 🤖
        - **InceptionV3** 🌈
        - **MobileNetV2** 📱
        - **EfficientNet** ⚡

    ### 🔍 **Performance Aspects Compared**
    - **⚡ Inference Time**: How quickly each model makes predictions.
    - **📦 Model Size**: The storage requirements of each model.
    - **🔢 Number of Parameters**: The complexity and capacity of each model.
    - **📈 Accuracy**: How well each model performs in classifying images.
    - **🏷️ Predicted Class**: The class each model predicts for the input images.

    - 📚 **Understanding Trade-Offs**: These comparisons will help us understand the trade-offs between speed, efficiency, and accuracy for each model.

    💡 **You can train your own model on any dataset following the link below:**
    [Train Your Model](https://github.com/ambideXtrous9/Brand-Logo-Classification-using-TransferLearning-Flickr27/tree/main/Final%20Model)
    
    🚀 **To see How it works..**
    """)
    
    
    

st.title("🚀:rainbow[Brand Logo Recognition] :sunglasses:")

model_card()        
