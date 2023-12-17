import xgboost as xgb
import pkg_resources
from .preprocess import preprocess_single_input

class CarPricePredictor:
    def __init__(self):
        import pdb; pdb.set_trace()
        weights_path = pkg_resources.resource_filename('CarPricePredictor', 'xgboost_weights.json')
        print(weights_path)
        self.model = xgb.Booster()
        self.model.load_model(weights_path)

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
                "Car": ["Toyota Camry", "Honda Accord"],
                "Vehicle Type": ["Sedan", "Sedan"],
                "Wheel left/right": ["Left", "Right"],
                "Color": ["Red", "Blue"],
                "Transmission": ["Automatic", "Manual"],
                "Mileage": [50000, 30000],
                "Year": [2018, 2019]
            }
        price_predictor = CarPricePredictor()
        predicted_price = price_predictor.predict_single_input(sample)
        """
        preprocessed_input = preprocess_single_input(input)
        predictions = self.model.predict(preprocessed_input)
        return predictions

