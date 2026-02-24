from django.shortcuts import render

# Create your views here.
import os
import joblib
import pandas as pd
from django.shortcuts import render
from django.conf import settings

model_path = os.path.join(settings.BASE_DIR, 'predictor', 'ml_models', 'fair_price_rf_model.pkl')
scaler_path = os.path.join(settings.BASE_DIR, 'predictor', 'ml_models', 'fair_price_scaler.pkl')

model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

expected_columns = [
    'RAM', 'Storage', 'Brand_DELL', 'Brand_Hp', 'Brand_Lenovo',
    'Brand_Toshiba', 'Processor_CORE I5', 'Processor_CORE I7',
    'Processor_CORE i5', 'Processor_Celeron', 'Processor_Core I3',
    'Processor_Core I5', 'Processor_Core I7', 'Processor_Core i5',
    'Processor_Core i7', 'Processor_PENTIUM', 'Processor_Pentium',
    'Processor_RYZEN 7', 'Processor_RYZEN 9', 'Processor_Ryzen 5',
    'Processor_pentium'
]

def home(request):
    return render(request, 'predictor/index.html')

def predict_page(request):
    predicted_price = None
    
    if request.method == 'POST':
        ram = int(request.POST.get('ram'))
        storage = int(request.POST.get('storage'))
        brand = request.POST.get('brand')
        processor = request.POST.get('processor')
        
        input_data = {col: 0 for col in expected_columns}
        
        input_data['RAM'] = ram
        input_data['Storage'] = storage
        
        brand_col = f'Brand_{brand}'
        if brand_col in input_data:
            input_data[brand_col] = 1
            
        processor_col = f'Processor_{processor}'
        if processor_col in input_data:
            input_data[processor_col] = 1
            
        df_input = pd.DataFrame([input_data], columns=expected_columns)
        
        scaled_input = scaler.transform(df_input)
        
        prediction = model.predict(scaled_input)[0]
        
        predicted_price = f"{int(prediction):,}"
        
    return render(request, 'predictor/predict.html', {'predicted_price': predicted_price})