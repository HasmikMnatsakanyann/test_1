import xgboost as xgb
from preprocessing import preprocess_input
import pkg_resources

class CarPricePredictor:
    def __init__(self):
        weights_path = pkg_resources.resource_filename('CarPricePredictor', 'xgboost_weights.json')
        print(weights_path)
        self.model = xgb.Booster()
        self.model.load_model(weights_path)

    def predict(self, input):
        """
        Function to predict price with USD:
        Input: ['Car', 'Date Posted', 'Year', 'Mileage', 'Vehicle Type', 'Transmission','Wheel left/right', 'Color']
        output: predicted price
        usage example :
        sample = ['Chevrolet Cruze', '13.02.2020', 2015, 30000.0, 'Sedan', 'Automatic','Left', 'Black']
        price_predictor = CarPricePredictor()
        predicted_price = price_predictor.predict(sample)
        """
        preprocessed_input = preprocess_input(input)
        predictions = self.model.predict(preprocessed_input)
        return predictions


if __name__ == "__main__":
    price_predictor = CarPricePredictor()
    price_predictor.predict()
