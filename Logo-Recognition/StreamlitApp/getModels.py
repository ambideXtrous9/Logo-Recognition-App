
from mlflow import MlflowClient
import config
import streamlit as st
import mlflow


mlflow.set_tracking_uri(config.tracking_uri)
client = MlflowClient(tracking_uri=config.tracking_uri)


# Function to calculate and print the model size and number of parameters
def model_details(model):
    # Calculate model size
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024**2  # Convert to megabytes

    # Calculate total number of parameters in millions
    total_params = sum(p.numel() for p in model.parameters())
    total_params_million = total_params / 10**6  # Convert to millions
    

    st.markdown(f"📚 Size: **{size_all_mb:.2f}** MB")
    st.markdown(f"💡 Parameters: **{total_params_million:.2f} M**")
    

def get_model(registry_model_name):

    try:
        # Fetch the latest version of the model that is in "Production" stage
        production_models = client.get_latest_versions(name=registry_model_name, stages=["Production"])
        
        if production_models:
            # Get the production model's version
            model_uri = f"models:/{registry_model_name}/Production"

            # Load the model from MLflow
            model = mlflow.pytorch.load_model(model_uri)

            return model
        else:
            print("No production model found.")
            return None

    except Exception as e:
        # Catch any exceptions that occur and print the error message
        print(f"An error occurred while loading the production model: {str(e)}")
        return None

def model_init():
    model_name = "Xception"
    
    xcep_model = get_model(registry_model_name=model_name)
    
    model_name = "MobileNetV2"
    
    mobile_model = get_model(registry_model_name=model_name)
    
    model_name = "InceptionV3"
    
    incep_model = get_model(registry_model_name=model_name)
    
    model_name = "EfficientNet"
    
    effcent_model = get_model(registry_model_name=model_name)
    
    return [("Xception",xcep_model),("InceptionV3",incep_model),("MobileNetV2", mobile_model),("EfficientNet" ,effcent_model)]
    

# Function to calculate and print the model size and number of parameters
def model_details(model):
    # Calculate model size
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024**2  # Convert to megabytes

    # Calculate total number of parameters in millions
    total_params = sum(p.numel() for p in model.parameters())
    total_params_million = total_params / 10**6  # Convert to millions
    

    st.markdown(f"📚 Size: **{size_all_mb:.2f}** MB")
    st.markdown(f"💡 Parameters: **{total_params_million:.2f} M**")
    