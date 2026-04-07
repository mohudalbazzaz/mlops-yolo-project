from PIL import Image
import numpy as np
import io

from src.backend.weather_api import get_weekly_temperature_df

def preprocess_image(file_bytes):

    img = Image.open(io.BytesIO(file_bytes))

    # normalising images for nn
    img = img.convert("RGB")
    img = img.resize((128, 128))
    # scaling pixels to avoid exploding/vanishing gradients 
    return np.array(img) / 255.0 # each pixel becomes a list of 3 numbers [R, G, B]

def compute_cumulative_ripening(initial_classification):

    k_ref = 0.25
    T_ref = 20
    Q10 = 2
    cum_ripeness = 0
    days = 0 

    while True:

        df = get_weekly_temperature_df()

        T_day = df.loc[days, "temperature_2m_max"] + 3 # assuming a kitchen is 3 degrees warmer than the outside

        per_day_ripening = k_ref * (Q10 ** ((T_day - T_ref) / 10))

        cum_ripeness += per_day_ripening
        days += 1

        if initial_classification == "Unripe" and cum_ripeness >= 1:
            return f'{days} days until ripeness'
        
        if initial_classification == "Overripe":
            return f'Past expiration'
        
        if initial_classification == "Ripe" and cum_ripeness >= 1:
            return f'{days} days until expiry'        
