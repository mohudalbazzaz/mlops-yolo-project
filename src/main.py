from fastapi import FastAPI, File, UploadFile
import numpy as np

from src.general_utils import preprocess_image, compute_cumulative_ripening
from src.train_model import load_model

model = load_model()

app = FastAPI()

@app.post("/banana_ripeness_classifier")
async def banana_ripeness_classifier(file: UploadFile = File(...)):

    image = await file.read()

    img = preprocess_image(image)
    
    img = np.expand_dims(img, axis=0)  # shape becomes (1, 128, 128, 3)

    prediction = model.predict(img)

    class_names = ["Overripe", "Ripe", "Unripe"]

    predicted_class = class_names[np.argmax(prediction)]

    classification = compute_cumulative_ripening(predicted_class)

    return classification









    



    



