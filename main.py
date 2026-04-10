from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import numpy as np

from src.backend.general_utils import preprocess_image, compute_cumulative_ripening
from src.backend.train_model import load_model

model = load_model()

app = FastAPI()


@app.get("/health")
async def health_check():
    return JSONResponse(content={"status": "healthy"}, status_code=200)


@app.post("/banana_ripeness_classifier")
async def banana_ripeness_classifier(file: UploadFile = File(...)) -> str:
    """Classify banana ripeness from an uploaded image and estimate ripening time.

    This endpoint accepts an image upload, preprocesses it for the model,
    performs inference to determine the ripeness class, and then computes
    the estimated number of days until ripeness or expiry using a
    temperature-based ripening model.

    Args:
        file: An uploaded image file provided by the client.

    Returns:
        A string describing either:
        - days until ripeness (if predicted unripe),
        - days until expiry (if predicted ripe),
        - or expiration status (if predicted overripe).
    """
    image = await file.read()

    img = preprocess_image(image)

    img = np.expand_dims(img, axis=0)  # shape becomes (1, 128, 128, 3)

    prediction = model.predict(img)

    class_names = ["Overripe", "Ripe", "Unripe"]

    predicted_class = class_names[np.argmax(prediction)]

    classification = compute_cumulative_ripening(predicted_class)

    return classification
