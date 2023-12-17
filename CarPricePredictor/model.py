import xgboost as xgb
import pkg_resources
from .preprocess import preprocess_single_input
import logging

class CarPricePredictor:
    def __init__(self):
        weights_path = pkg_resources.resource_filename('CarPricePredictor', 'weights.json')
        self.model = xgb.Booster()
        self.model.load_model(weights_path)
        logging.log(level=1, msg="Model loaded.")

    def predict_single_input(self, input):
        """
        Function to predict price on a single input of the form : {
                "Car": ["Toyota Camry", "Honda Accord"],
                "Vehicle Type": ["Sedan", "Sedan"],
                "Wheel left/right": ["Left", "Right"],
                "Color": ["Red", "Blue"],
                "Transmission": ["Automatic", "Manual"],
                "Mileage": [50000, 30000],
                "Year": [2018, 2019]
            }
        Input: Dict as shown in example
        usage example :
        sample = {
"Car": "Toyota Camry",
"Vehicle Type": "Sedan",
"Wheel left/right": "Left",
"Color": "Red",
"Transmission": "Automatic",
"Mileage": 50000,
"Year": 2018
            }
        price_predictor = CarPricePredictor()
        predicted_price = price_predictor.predict_single_input(sample)
        """
        preprocessed_input = preprocess_single_input(input)
        dmat_data = xgb.DMatrix(preprocessed_input)
        predictions = self.model.predict(dmat_data)
        return predictions

