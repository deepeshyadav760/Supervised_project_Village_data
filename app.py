from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import numpy as np
app = Flask(__name__)

# Load the model
# model = joblib.load('Linear_regression.joblib')
model = joblib.load('Linear_regression2.joblib')
model2 = joblib.load('dt_model.joblib')
# Define the API endpoint for predictions
@app.route('/Linear_predict', methods=['POST'])
def Linear_predict():
    # Get data from the request
    data = request.json
    
    # Convert data to DataFrame
    df = pd.DataFrame(data, index=[0])
    
    # Ensure the columns are in the correct order
    df = df[['NUMBER OF SHGS FEDERATED INTO VILLAGE ORGANISATIONS (VOS)', 
             'NUMBER OF SHGS WHICH ACCESSED BANK LOANS', 
             'NUMBER OF BENEFICIARIES RECEIVING BENEFITS UNDER AAYUSHMAN BHARAT-PRADHAN MANTRI JAN AROGYA YOJANA OR ANY STATE GOVT HEALTH SCHEME', 
             'TOTAL NUMBER OF HOUSEHOLDS RECEIVING FOOD GRAINS FROM FAIR PRICE SHOPS ', 
             'TOTAL NUMBER OF FARMERS ', 
             'TOTAL EXPENDITURE APPROVED UNDER NRM IN THE LABOUR BUDGET FOR THE YEAR 2018-19)',
             ]]
    
    # Make prediction
    prediction = model.predict(df)
    
    # Return the prediction
    return jsonify({'prediction': prediction.tolist()})

#  categorical_vars = {
#         'AVAILABILITY OF GOVERNMENT SEED CENTRES': ['Yes', 'No (Nearest facility < 1 km)', 'No (Nearest facility 1-2 kms)', 'No (Nearest facility 2-5 kms)', 'No (Nearest facility 5-10 kms)', 'No (Nearest facilityMore than 10 kms)'],
#         'WHETHER THIS VILLAGE IS A PART OF THE WATERSHED DEVELOPMENT PROJECT': ['Yes', 'No'],
#         'AVAILABILITY OF COMMUNITY RAIN WATER HARVESTING SYSTEM/POND/DAM/CHECK DAM ETC.': ['Yes', 'No'],
#         'AVAILABILITY OF WAREHOUSE FOR FOOD GRAIN STORAGE': ['Yes', 'No (Nearest facility < 1 km)', 'No (Nearest facility 1-2 kms)', 'No (Nearest facility 2-5 kms)', 'No (Nearest facility 5-10 kms)', 'No (Nearest facilityMore than 10 kms)'],
#         'DOES THE VILLAGE HAVE ACCESS TO CUSTOM HIRING CENTRE (AGRI-EQUIPMENTS)': ['Yes', 'No'],
#         'AVAILABILITY OF SOIL TESTING CENTRES': ['Yes', 'No (Nearest facility < 1 km)', 'No (Nearest facility 1-2 kms)', 'No (Nearest facility 2-5 kms)', 'No (Nearest facility 5-10 kms)', 'No (Nearest facilityMore than 10 kms)'],
#         'AVAILABILITY OF FERTILIZER SHOP': ['Yes', 'No (Nearest facility < 1 km)', 'No (Nearest facility 1-2 kms)', 'No (Nearest facility 2-5 kms)', 'No (Nearest facility 5-10 kms)', 'No (Nearest facilityMore than 10 kms)'],
#         'AVAILABILITY OF MILK COLLECTION CENTRE /MILK ROUTES / CHILLING CENTRES': ['Yes', 'No'],
#         'AVAILABILITY OF VETERINARY CLINIC OR HOSPITAL': ['Yes', 'No (Nearest facility < 1 km)', 'No (Nearest facility 1-2 kms)', 'No (Nearest facility 2-5 kms)', 'No (Nearest facility 5-10 kms)', 'No (Nearest facilityMore than 10 kms)'],
#         'AVAILABILITY OF PIPED TAP WATER': ['100% habitations covered', '50 to 100% habitations covered', '<50% habitation covered', 'None (Nearest facilityMore than 10 kms)', 'only one habitation is covered', 'None (Nearest facility2-5 kms)', 'None (Nearest facility5-10 kms)', 'None (Nearest facility1-2 kms)'],
#         'AVAILABILITY OF MARKETS': ['Sub Centre', 'PHC', 'CHC', 'None (Nearest facility < 1 km)', 'None (Nearest facility 1-2 kms)', 'None (Nearest facility 2-5 kms)', 'None (Nearest facility 5-10 kms)', 'None (Nearest facilityMore than 10 kms)'],
#         'BEE KEEPING': ['Yes', 'No'],
#         'SERICULTURE (SILK PRODUCTION)': ['Yes', 'No'],
#         'HANDLOOM': ['Yes', 'No'],
#         'HANDICRAFTS': ['Yes', 'No'],
#         'AVAILABILITY OF COMMUNITY FOREST': ['Yes', 'No'],
#         'AVAILABILITY OF PRIMARY PROCESSING FACILITIES AT THE VILLAGE LEVEL': ['Yes', 'No']
#     }

@app.route('/DT_predict', methods=['POST'])
def DT_predict():
    # Define possible choices for each categorical variable
    print("clicked")
    categorical_vars = {
            'AVAILABILITY OF GOVERNMENT SEED CENTRES': ['Yes', 'No (Nearest facility < 1 km)', 'No (Nearest facility 1-2 kms)', 'No (Nearest facility 2-5 kms)', 'No (Nearest facility 5-10 kms)', 'No (Nearest facilityMore than 10 kms)'],
            'WHETHER THIS VILLAGE IS A PART OF THE WATERSHED DEVELOPMENT PROJECT': ['Yes', 'No'],
            'AVAILABILITY OF COMMUNITY RAIN WATER HARVESTING SYSTEM/POND/DAM/CHECK DAM ETC.': ['Yes', 'No'],
            'AVAILABILITY OF WAREHOUSE FOR FOOD GRAIN STORAGE ': ['Yes', 'No (Nearest facility < 1 km)', 'No (Nearest facility 1-2 kms)', 'No (Nearest facility 2-5 kms)', 'No (Nearest facility 5-10 kms)', 'No (Nearest facilityMore than 10 kms)'],
            'DOES THE VILLAGE HAVE ACCESS TO CUSTOM HIRING CENTRE (AGRI-EQUIPMENTS) ': ['Yes', 'No'],
            'AVAILABILITY OF SOIL TESTING CENTRES': ['Yes', 'No (Nearest facility < 1 km)', 'No (Nearest facility 1-2 kms)', 'No (Nearest facility 2-5 kms)', 'No (Nearest facility 5-10 kms)', 'No (Nearest facilityMore than 10 kms)'],
            'AVAILABILITY OF FERTILIZER SHOP': ['Yes', 'No (Nearest facility < 1 km)', 'No (Nearest facility 1-2 kms)', 'No (Nearest facility 2-5 kms)', 'No (Nearest facility 5-10 kms)', 'No (Nearest facilityMore than 10 kms)'],
            'AVAILABILITY OF MILK COLLECTION CENTRE /MILK ROUTES / CHILLING CENTRES ': ['Yes', 'No'],
            'AVAILABILITY OF VETERINARY CLINIC OR HOSPITAL': ['Yes', 'No (Nearest facility < 1 km)', 'No (Nearest facility 1-2 kms)', 'No (Nearest facility 2-5 kms)', 'No (Nearest facility 5-10 kms)', 'No (Nearest facilityMore than 10 kms)'],
            'AVAILABILITY OF PIPED TAP WATER': ['100% habitations covered', '50 to 100% habitations covered', '<50% habitation covered', 'None (Nearest facilityMore than 10 kms)', 'only one habitation is covered', 'None (Nearest facility2-5 kms)', 'None (Nearest facility5-10 kms)', 'None (Nearest facility1-2 kms)'],
            'AVAILABILITY OF MARKETS': ['Sub Centre', 'PHC', 'CHC', 'None (Nearest facility < 1 km)', 'None (Nearest facility 1-2 kms)', 'None (Nearest facility 2-5 kms)', 'None (Nearest facility 5-10 kms)', 'None (Nearest facilityMore than 10 kms)'],
            'BEE KEEPING': ['Yes', 'No'],
            'SERICULTURE (SILK PRODUCTION)': ['Yes', 'No'],
            'HANDLOOM': ['Yes', 'No'],
            'HANDICRAFTS': ['Yes', 'No'],
            'AVAILABILITY OF COMMUNITY FOREST': ['Yes', 'No'],
            'AVAILABILITY OF PRIMARY PROCESSING FACILITIES AT THE VILLAGE LEVEL': ['Yes', 'No']
        }


    # Extract data from the form
    data = request.form.to_dict()

    # Prepare the model input dictionary with default values for categorical variables
    model_input = {f"{var}_{choice}": 0 for var, choices in categorical_vars.items() for choice in choices}
    
    # Assigning numeric values
    model_input.update({
        'NUMBER OF TOTAL POPULATION': float(data.get('NUMBER OF TOTAL POPULATION', 0)),
        'NUMBER OF HOUSEHOLDS ENGAGED MAJORLY IN FARM ACTIVITIES': float(data.get('NUMBER OF HOUSEHOLDS ENGAGED MAJORLY IN FARM ACTIVITIES', 0)),
        'NUMBER OF HOUSEHOLDS ENGAGED MAJORLY IN NON-FARM ACTIVITIES': float(data.get('NUMBER OF HOUSEHOLDS ENGAGED MAJORLY IN NON-FARM ACTIVITIES', 0)),
        'TOTAL AREA IRRIGATED (IN HECTARE)': float(data.get('TOTAL AREA IRRIGATED (IN HECTARE)', 0)),
        'TOTAL UNIRRIGATED LAND AREA (IN HECTARES)': float(data.get('TOTAL UNIRRIGATED LAND AREA (IN HECTARES)', 0)),
        'AVAILABILITY OF HIGHER/SENIOR SECONDARY SCHOOL': float(data.get('AVAILABILITY OF HIGHER/SENIOR SECONDARY SCHOOL', 0)),
        'NUMBER OF HOUSEHOLDS MOBILIZED INTO SHGS': float(data.get('NUMBER OF HOUSEHOLDS MOBILIZED INTO SHGS', 0)),
        'NUMBER OF HOUSEHOLDS HAVING PIPED WATER CONNECTION': float(data.get('NUMBER OF HOUSEHOLDS HAVING PIPED WATER CONNECTION', 0))
    })


    # One-hot encoding for categorical variables based on received form data
    for var, choices in categorical_vars.items():
        selected_option = data.get(var)
        if selected_option in choices:
            model_input[f"{var}_{selected_option}"] = 1

    # Creating DataFrame from the model input dictionary
    df = pd.DataFrame([model_input])

    # Making prediction using model2
    prediction = model2.predict(df)
    prediction = np.round(prediction, 2)

    # Returning the prediction result
    return jsonify({'prediction': prediction.tolist()})

@app.route('/')
def Linear_regression():
    return render_template('Linear_regression.html')

# @app.route('/dt_classification')
# def dt_classification():
#     return render_template('dt_classification.html')

# # Define the route for the home page with the form
# @app.route('/')
# def home():
#     return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)






