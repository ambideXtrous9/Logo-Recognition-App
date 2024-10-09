import os 


WIDTH = 224
HEIGHT = 224
BATCH_SIZE = 32
VAL_SPLIT = 0.2
NUM_CLASSES = 27
LR = 0.001
QUERY_SET_PATH = 'flickr_logos_dataset/flickr_logos_27_dataset/flickr_logos_27_dataset_query_set_annotation.txt'
DATA_FOLDER = 'flickr_logos_dataset/Cropped_Logos'
MAIN_LOGO_FOLDER = 'flickr_logos_dataset/flickr_logos_27_dataset_images'
MAX_EPOCHS = 100

tracking_uri = "http://34.47.170.249:5000/"

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'mlops-437407-225d42b6661e.json'

