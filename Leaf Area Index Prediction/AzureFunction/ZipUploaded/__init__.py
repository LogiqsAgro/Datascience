import logging

import azure.functions as func
from cv2 import Laplacian
from keras.models import load_model
import pandas as pd
from zipfile import ZipFile
from typing import Tuple
import cv2
import numpy as np
from scipy import ndimage
from joblib import load
from sklearn.preprocessing import StandardScaler
import os
import tensorflow as tf
from io import BytesIO
from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobClient, BlobServiceClient
from datetime import datetime

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
BLOB_STORAGE_URL = "https://leafareaindexuploads.blob.core.windows.net/txt-outputs"

def extract_zip_into_ndarrays(input_zip) -> Tuple[cv2.Mat, cv2.Mat]:
    """
    Extracts zip into ndarrays. Expects that there are 2 PNGs 
    that contain the 'front' and 'top' substring in their filename.
    
    Returns tuple of ndarrays that represent front and top images.
    """
    byte_stream = BytesIO(input_zip.read())
    
    input_zip = ZipFile(byte_stream)
    images_bytes = {name: input_zip.read(name) for name in input_zip.namelist()}
    
    image_bytes_front = [value for key, value in images_bytes.items() if 'front' in key.lower()][0]
    image_bytes_top = [value for key, value in images_bytes.items() if 'top' in key.lower()][0]
    
    nparr_front = np.fromstring(image_bytes_front, np.uint8)
    cv2_image_front = cv2.imdecode(nparr_front, -1)
    
    nparr_top = np.fromstring(image_bytes_top, np.uint8)
    cv2_image_top = cv2.imdecode(nparr_top, -1)
    
    return cv2_image_front, cv2_image_top
    

def calculate_area_size(image) -> float:
    """
    Calculates total green area in image.
    """
    
    img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(img_hsv,(31, 69, 20), (82, 255, 255))
    area_size = np.sum(mask)/255
    
    #centroid = ndimage.center_of_mass(mask) -- left out for this version
    
    return area_size


def load_model_and_scalers() -> Tuple[StandardScaler, StandardScaler, tf.keras.Model]:
    
    """
    Loads models and scalers from their respective (local) directories.
    """
    
    predictors_scaler = load(f"{ROOT_DIR}/scalers/predictors_scaler.bin")
    label_scaler = load(f"{ROOT_DIR}/scalers/label_scaler.bin")
    model = load_model(f"{ROOT_DIR}/lai-prediction-model")
    
    return predictors_scaler, label_scaler, model


def main(myblob: func.InputStream):
    logging.info(f"Python blob trigger function processed blob \n"
                 f"Name: {myblob.name}\n"
                 f"Blob Size: {myblob.length} bytes")
    
    try:
        logging.info("Extracting zip contents into ndarray representations.")
        cv2_image_front, cv2_image_top = extract_zip_into_ndarrays(myblob)
        front_area_size, top_area_size = calculate_area_size(cv2_image_front), calculate_area_size(cv2_image_top)
        
        logging.info("Loading scalers and model.")
        predictors_scaler, label_scaler, model = load_model_and_scalers()
        scaled_area_sizes = predictors_scaler.transform([[top_area_size,front_area_size]])
        
        logging.info("Making prediction.")
        lai_prediction = model.predict(scaled_area_sizes)
        lai_prediction = label_scaler.inverse_transform(lai_prediction)[0][0]
        
        logging.info(f"The LAI prediction for {myblob.name} is {lai_prediction}.")
        
        
        
        upload_name = myblob.name.split("/")[1]

        blob_name =f"{datetime.now()}-{upload_name}.txt"
        
        blob_service_client = BlobServiceClient.from_connection_string(
        conn_str=os.environ["leafareaindexuploads_txtoutputs"],
        )
        
        logging.info("Uploading output to txt_outputs.")        
        blob_client = blob_service_client.get_blob_client(
            container="txt-outputs",
            blob=blob_name
            )
        blob_client.upload_blob(data=str(lai_prediction))
            
    except Exception as e:
        logging.error(f"Something went wrong during the LAI prediction for {myblob.name}.")
        logging.error(e)
        
    logging.info(f"Succesfully uploaded results to blob storage.")
    
