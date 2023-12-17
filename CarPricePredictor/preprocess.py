import numpy as np


Vehicle_Type_DICT = {'Sedan': 0,
 'Coupe': 1,
 'Estate': 2,
 'Hatchback': 3,
 'Van': 4,
 'MPV': 5,
 'SUV': 6,
 'Pickup': 7,
 'Limousine': 8,
 'Convertible or Roadster': 9}
 

COLOR_DICT = {'Blue': 0,
 'Green': 1,
 'Gray': 2,
 'Silver': 3,
 'White': 4,
 'Gold': 5,
 'Beige': 6,
 'Azure': 7,
 'Black': 8,
 'Red': 9,
 'Eggplant': 10,
 'Other Color': 11,
 'Purple': 12,
 'Yellow': 13,
 'Cherry': 14,
 'Orange': 15,
 'Brown': 16,
 'Pink': 17,}


MANUFACTURER = {'VAZ(Lada)': 0,
 'BMW': 1,
 'ErAZ': 2,
 'Chevrolet': 3,
 'Ford': 4,
 'GAZ': 5,
 'IZH': 6,
 'Opel': 7,
 'Volkswagen': 8,
 'Mercedes-Benz': 9,
 'Audi': 10,
 'Niva': 11,
 'Volvo': 12,
 'Mitsubishi': 13,
 'UAZ': 14,
 'ZAZ-Tavria': 15,
 'Nissan': 16,
 'Seat': 17,
 'Renault': 18,
 'Suzuki': 19,
 'Subaru': 20,
 'Toyota': 21,
 'Jeep': 22,
 'Mazda': 23,
 'Lexus': 24,
 'Chrysler': 25,
 'Hyundai': 26,
 'Dodge': 27,
 'Eagle': 28,
 'Honda': 29,
 'Kia': 30,
 'Smart': 31,
 'Ikco': 32,
 'Fiat': 33,
 'Peugeot': 34,
 'Skoda': 35,
 'Isuzu': 36,
 'Land': 37,
 'Infiniti': 38,
 'Mini': 39,
 'Chery': 40,
 'Citroen': 41,
 'Acura': 42,
 'Jaguar': 43,
 'Cadillac': 44,
 'Great': 45,
 'Saturn': 46,
 'Lincoln': 47,
 'Porsche': 48,
 'Foton': 49,
 'Hummer': 50,
 'Buick': 51,
 'GMC': 52,
 'Alpina': 53,
 'Bentley': 54,
 'Tesla': 55,
 'Rolls': 56}


TRANSMISSION = {'Manual': 0, 'Automatic': 1, 'Variator': 2, 'Semi-Automatic': 3}

WHEEL = {'Left': 0, 'Right': 1, 'Changed l->R': 2}



def one_hot_encode_inference(value, encoding_dict):
    one_hoted = np.zeros(len(encoding_dict) + 1)
    one_hoted[encoding_dict.get(value, -1)] = 1
    return one_hoted

def preprocess_single_input(input_dict):
    car_one_hoted = one_hot_encode_inference(input_dict["Car"].split()[0], MANUFACTURER)
    Vehicle_type_one_hoted = one_hot_encode_inference(input_dict["Vehicle Type"], Vehicle_Type_DICT)
    wheel_one_hoted = one_hot_encode_inference(input_dict['Wheel left/right'], WHEEL)
    color_one_hoted = one_hot_encode_inference(input_dict["Color"], COLOR_DICT)
    transmission_one_hoted = one_hot_encode_inference(input_dict["Transmission"], TRANSMISSION)

    mileage = np.array([input_dict.get("Mileage", 0)]) 
    year = np.array([input_dict.get("Year", 0)])  

    inp = np.concatenate((year, mileage, car_one_hoted, Vehicle_type_one_hoted, wheel_one_hoted, color_one_hoted, transmission_one_hoted, ))
    
    return inp.reshape(1, -1)


def preprocess_dataframe(df):
    processed_data = []

    for _, row in df.iterrows():
        car_one_hoted = one_hot_encode_inference(row["Car"].split()[0], MANUFACTURER)
        Vehicle_type_one_hoted = one_hot_encode_inference(row["Vehicle Type"], Vehicle_Type_DICT)
        wheel_one_hoted = one_hot_encode_inference(row['Wheel left/right'], WHEEL)
        color_one_hoted = one_hot_encode_inference(row["Color"], COLOR_DICT)
        transmission_one_hoted = one_hot_encode_inference(row["Transmission"], TRANSMISSION)
        
        mileage = np.array([row.get("Mileage", 0)])
        year = np.array([row.get("Year", 0)])

        inp = np.concatenate((year, mileage, car_one_hoted, Vehicle_type_one_hoted, wheel_one_hoted, color_one_hoted, transmission_one_hoted))
        
        processed_data.append(inp)

    return np.array(processed_data)
