import streamlit as st
from model import CarPricePredictor

st.title('Car Price Prediction')
st.write('Enter the car details to predict its price.')


manufacturer = st.selectbox("Select Car Manufacturer", ['BMW', 'ErAZ', 'Chevrolet', 'Ford', 'GAZ', 'IZH', 'Opel', 'Volkswagen', 'Mercedes-Benz', 'Audi', 'Niva', 'Volvo', 'Mitsubishi', 'UAZ', 'ZAZ-Tavria', 'Nissan', 'Seat', 'Renault', 'Suzuki', 'Subaru', 'Toyota', 'Jeep', 'Mazda', 'Lexus', 'Chrysler', 'Hyundai', 'Dodge', 'Eagle', 'Honda', 'Kia', 'Smart', 'Ikco', 'Fiat', 'Peugeot', 'Skoda', 'Isuzu', 'Land', 'Infiniti', 'Mini', 'Chery', 'Citroen', 'Acura', 'Jaguar', 'Cadillac', 'Great', 'Saturn', 'Lincoln', 'Porsche', 'Foton', 'Hummer', 'Buick', 'GMC', 'Alpina', 'Bentley', 'Tesla', 'Rolls'])
model = st.text_input("Model: ")
vehicle_type = st.selectbox("Select Vehicle Type", ["Sedan", "Coupe", "Estate", "Hatchback", "Van", "MPV", "SUV", "Pickup", "Limousine", "Convertible or Roadster"])
wheel_position = st.selectbox("Select Wheel Position", ["Left", "Right", "Changed l->R"])
color = st.selectbox("Select Car Color",['Blue', 'Green', 'Gray', 'Silver', 'White', 'Gold', 'Beige','Azure', 'Black', 'Red', 'Eggplant', 'Other Color', 'Purple','Yellow', 'Cherry','Orange', 'Brown', 'Pink'] )
transmission = st.selectbox("Select Transmission", ['Manual', 'Automatic', 'Variator', 'Semi-Automatic'])
mileage = st.number_input("Enter Mileage (in miles)", min_value=0)
year = st.number_input("Enter Year of Manufacture", max_value=2024)


if st.button('Predict Price'):
    predictor = CarPricePredictor()

    input_data = {
        "Car": manufacturer,
        "Vehicle Type": vehicle_type,
        "Wheel left/right": wheel_position,
        "Color": color,
        "Transmission": transmission,
        "Mileage": mileage,
        "Year": year,
    }
    

    try:
        prediction = predictor.predict_single_input(input_data)
        st.success(f'Predicted Price: ${prediction}')
    except Exception as e:
        st.error(f'Error: {e}')

# Run the Streamlit app with `streamlit run app.py`
