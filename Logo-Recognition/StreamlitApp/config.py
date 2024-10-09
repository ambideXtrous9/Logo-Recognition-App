import os 

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'mlops-437407-225d42b6661e.json'


WIDTH = 224
HEIGHT = 224

# Define the classes dictionary
CLASS_LABELS = {
    'Adidas': 0, 'Apple': 1, 'BMW': 2, 'Citroen': 3, 'Cocacola': 4, 
    'DHL': 5, 'Fedex': 6, 'Ferrari': 7, 'Ford': 8, 'Google': 9, 
    'HP': 10, 'Heineken': 11, 'Intel': 12, 'McDonalds': 13, 'Mini': 14, 
    'Nbc': 15, 'Nike': 16, 'Pepsi': 17, 'Porsche': 18, 'Puma': 19, 
    'RedBull': 20, 'Sprite': 21, 'Starbucks': 22, 'Texaco': 23, 
    'Unicef': 24, 'Vodafone': 25, 'Yahoo': 26
}

tracking_uri = "http://34.47.170.249:5000/"


INV_LABELS_CLASS = {v: k for k, v in CLASS_LABELS.items()}

