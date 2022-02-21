from fastapi import FastAPI, UploadFile
from fastapi import File
from fastapi.encoders import jsonable_encoder
import uvicorn
from  keras.applications import mobilenet_v2
import numpy as np
from PIL import Image
from io import BytesIO
import tensorflow as tf

from keras.applications.resnet import ResNet50
from keras.applications.imagenet_utils import decode_predictions

from keras import applications 

input_shape=(224,224,3)
model = None
def read_imagefile(file) -> Image.Image:
    image = Image.open(BytesIO(file))
    return image
    
def load_model():
    model =ResNet50(weights="imagenet")
    return model




model=load_model()  

  
def predict(image:np.ndarray):
    
    image = np.asarray(image.resize((224, 224)))[..., :3]
    image = np.expand_dims(image, 0)
    image = image / 127.5 - 1.0
    result = decode_predictions(model.predict(image), 2)[0]
    response={}
    for i, res in enumerate(result):
        resp = {}
        resp["class"] = res[1]
        resp["confidence"] = f"{res[2]*100:0.2f} %"
        response.append(resp)

    return response
       
    
app= FastAPI()
@app.get("/")
def home():
    return{"hello":"world"}
prediction={}
@app.post("/api")
async def predict_image(file:UploadFile=File(...)):
    image=read_imagefile(await file.read())
      
    prediction =predict(image)
    return jsonable_encoder(prediction)
    
@app.get("/api")
def predicted():
    return prediction
    
    

if __name__=="__main__":
    uvicorn.run(app)