

# # # working
# # from flask import Flask, request, jsonify
# # import joblib
# # import shap
# # import numpy as np
# # import pandas as pd

# # app = Flask(__name__)

# # # Load pre-trained model and encoder
# # pipeline = joblib.load('model/risk_model.pkl')
# # encoder = joblib.load('encoder.pkl')
# # model = pipeline.named_steps['classifier']
# # preprocessor = pipeline.named_steps['preprocessor']

# # # Initialize SHAP explainer
# # num_features = preprocessor.transformers_[0][1].transform(
# #     np.zeros((1, len(preprocessor.transformers_[0][1].feature_names_in_)))
# # ).shape[1]
# # explainer_shap = shap.KernelExplainer(lambda x: model.predict_proba(x)[:, 1], np.zeros((1, num_features)))

# # def get_risk_level(risk_score):
# #     """Determine risk level based on risk score."""
# #     if risk_score <= 30:
# #         return 'Low'
# #     elif risk_score <= 70:
# #         return 'Moderate'
# #     else:
# #         return 'High'

# # @app.route('/api/predict', methods=['POST'])
# # def predict():
# #     data = request.json
# #     features = data['features']
# #     print("Input Features:", features)  # Debugging
    
# #     # Create DataFrame
# #     new_data = pd.DataFrame([features])
    
# #     # Encode categorical variables using the same encoder
# #     encoded_categorical = encoder.transform(new_data[["countryCode", "usageType", "domain", "isp"]])
# #     encoded_df = pd.DataFrame(
# #         encoded_categorical, columns=["countryCode", "usageType", "domain", "isp"]
# #     )
    
# #     # Concatenate encoded features with the rest of the DataFrame
# #     new_data_encoded = pd.concat(
# #         [new_data.drop(["countryCode", "usageType", "domain", "isp"], axis=1), encoded_df], axis=1
# #     )
    
# #     # Transform input
# #     new_data_transformed = preprocessor.transform(new_data_encoded)
    
# #     # Predict risk score
# #     risk_score_prob = model.predict_proba(new_data_transformed)[0][1]
# #     risk_score = int(risk_score_prob * 100)
    
# #     # Get risk level
# #     risk_level = get_risk_level(risk_score)

# #     # SHAP explanation
# #     shap_values = explainer_shap.shap_values(new_data_transformed)
# #     shap_explanation = shap_values[0].tolist()

# #     # SHAP summary (top N features)
# #     N = 10
# #     feature_names = preprocessor.transformers_[0][1].feature_names_in_
# #     shap_summary = sorted(zip(feature_names, shap_explanation), key=lambda x: abs(x[1]), reverse=True)[:N]

# #     # Format SHAP output and convert numpy types to native Python types
# #     formatted_shap_explanation = [
# #         {"feature": feature, "value": new_data[feature].iloc[0] if feature in new_data else "N/A", "impact": "increases risk" if value > 0 else "decreases risk" if value < 0 else "no impact"}
# #         for feature, value in shap_summary
# #     ]
# #     formatted_shap_explanation = [{k: (int(v) if isinstance(v, (np.int64, np.int32)) else 
# #                                        float(v) if isinstance(v, (np.float64, np.float32)) else 
# #                                        bool(v) if isinstance(v, np.bool_) else v) 
# #                                    for k, v in item.items()} for item in formatted_shap_explanation]
    
# #     return jsonify({'riskScore': int(risk_score), 'riskLevel': risk_level, 'shapSummary': formatted_shap_explanation})

# # if __name__ == '__main__':
# #     app.run(debug=True, port=5001)
# #!/usr/bin/env python
# # from flask import Flask, request, jsonify
# # import joblib
# # import shap
# # import numpy as np
# # import pandas as pd
# # import logging
# # import os

# # app = Flask(__name__)
# # logging.basicConfig(level=logging.INFO)

# # # Load the pre-trained pipeline and encoder.
# # pipeline = joblib.load('model/risk_model.pkl')
# # encoder = joblib.load('encoder.pkl')
# # # Extract classifier and preprocessor from the pipeline.
# # model = pipeline.named_steps['classifier']
# # preprocessor = pipeline.named_steps['preprocessor']

# # # Define the expected features (order must match training).
# # expected_numeric_features = ["totalReports", "numDistinctUsers", "lastReportedAt", "isWhitelisted", "isTor"]
# # expected_categorical_features = ["countryCode", "usageType", "domain", "isp"]
# # expected_features = expected_numeric_features + expected_categorical_features

# # # Prepare a background dataset for SHAP.
# # try:
# #     df_encoded = pd.read_csv("synthetic_data_encoded.csv")
# #     # Keep only the expected features.
# #     X_background = df_encoded[expected_features].iloc[:100]
# #     X_background_transformed = preprocessor.transform(X_background)
# # except Exception as e:
# #     logging.error("Error loading background sample: %s", e)
# #     X_background_transformed = np.zeros((1, len(expected_features)))

# # # Define a prediction function that returns the probability for class 1.
# # def predict_prob(x):
# #     return model.predict_proba(x)[:, 1]

# # # Create SHAP KernelExplainer using the defined prediction function.
# # explainer_shap = shap.KernelExplainer(predict_prob, X_background_transformed)

# # def get_risk_level(risk_score):
# #     """Determine risk level based on risk score."""
# #     if risk_score <= 30:
# #         return 'Low'
# #     elif risk_score <= 70:
# #         return 'Moderate'
# #     else:
# #         return 'High'

# # @app.route('/api/predict', methods=['POST'])
# # def predict():
# #     try:
# #         data = request.json
# #         features = data.get('features')
# #         if features is None:
# #             raise ValueError("Missing 'features' in JSON payload.")
# #         logging.info("Input Features: %s", features)
        
# #         # Create a DataFrame from input features.
# #         df_input = pd.DataFrame([features])
        
# #         # (Optional) If isWhitelisted is null, treat it as False.
# #         if df_input['isWhitelisted'].isnull().any():
# #             df_input['isWhitelisted'] = df_input['isWhitelisted'].fillna(False)
        
# #         # Keep only the expected features (drop any extra ones).
# #         df_input = df_input[expected_features]
        
# #         # Process categorical features using the encoder.
# #         encoded_categorical = encoder.transform(df_input[expected_categorical_features])
# #         encoded_df = pd.DataFrame(encoded_categorical, columns=expected_categorical_features)
        
# #         # Extract numeric features.
# #         df_numeric = df_input[expected_numeric_features]
        
# #         # Combine numeric and encoded categorical features.
# #         new_data_encoded = pd.concat([df_numeric, encoded_df], axis=1)
        
# #         # Transform using the preprocessor (applies StandardScaler on all features).
# #         new_data_transformed = preprocessor.transform(new_data_encoded)
        
# #         # If the IP is whitelisted, immediately return low risk.
# #         if df_input['isWhitelisted'].iloc[0]:
# #             risk_score_prob = 0.0
# #             risk_score = 0
# #             risk_level = 'Low'
# #             base_value = None
# #             shap_summary = []
# #         else:
# #             # Get predicted probability for abusive class.
# #             risk_score_prob = predict_prob(new_data_transformed)[0]
# #             risk_score = int(risk_score_prob * 100)
# #             risk_level = get_risk_level(risk_score)
            
# #             # SHAP explanation.
# #             # Since our predict_prob returns a scalar, explainer_shap.expected_value is a scalar.
# #             base_value = float(explainer_shap.expected_value)
# #             shap_values = explainer_shap.shap_values(new_data_transformed)
# #             # shap_values is a 2D array: shape (1, n_features).
# #             # These values should sum to (risk_score_prob - base_value).
            
# #             # Prepare SHAP summary: pair feature names with their contributions.
# #             shap_summary = sorted(
# #                 zip(new_data_encoded.columns.tolist(), shap_values[0]),
# #                 key=lambda x: abs(x[1]),
# #                 reverse=True
# #             )
            
# #             # Format the SHAP summary.
# #             formatted_shap_summary = []
# #             for feature, contribution in shap_summary:
# #                 if abs(contribution) > 1e-6:
# #                     # Get the raw input value for this feature (if available).
# #                     raw_value = features.get(feature, "N/A")
# #                     formatted_shap_summary.append({
# #                         "feature": feature,
# #                         "value": raw_value,
# #                         "contribution": float(contribution)
# #                     })
# #             shap_summary = formatted_shap_summary
        
# #         # Build a calculation summary string for transparency.
# #         calc_summary = (
# #             f"risk_score_prob = model.predict_proba(x)[:,1] -> {risk_score_prob:.4f}, "
# #             f"riskScore = int(risk_score_prob * 100) = {risk_score}, "
# #             f"base_value = {base_value if base_value is not None else 'N/A'}, "
# #             f"sum(SHAP values) = {np.sum(shap_values[0]) if not df_input['isWhitelisted'].iloc[0] else 'N/A'} "
# #             f"(base_value + sum(SHAP) should equal risk_score_prob)"
# #         )
        
# #         response = {
# #             'riskScore': risk_score,
# #             'riskLevel': risk_level,
# #             'baseValue': base_value,
# #             'shapSummary': shap_summary,
# #             'calculation': calc_summary
# #         }
# #         return jsonify(response)
# #     except Exception as e:
# #         logging.error("Error in prediction: %s", e)
# #         return jsonify({'error': str(e)}), 500

# # if __name__ == '__main__':
# #     app.run(debug=True, port=5001)


# from flask import Flask, request, jsonify
# import joblib
# import shap
# import numpy as np
# import pandas as pd
# import logging
# import os

# app = Flask(__name__)
# logging.basicConfig(level=logging.INFO)

# # Load the pre-trained pipeline and encoder.
# pipeline = joblib.load('model/risk_model.pkl')
# encoder = joblib.load('encoder.pkl')

# # Extract classifier and preprocessor from the pipeline.
# model = pipeline.named_steps['classifier']
# preprocessor = pipeline.named_steps['preprocessor']

# # Define the expected features (order must match training).
# expected_numeric_features = ["totalReports", "numDistinctUsers", "lastReportedAt", "isWhitelisted", "isTor"]
# expected_categorical_features = ["countryCode", "usageType", "domain", "isp"]
# expected_features = expected_numeric_features + expected_categorical_features

# # Prepare a background dataset for SHAP.
# try:
#     df_encoded = pd.read_csv("synthetic_data_encoded.csv")
#     # Keep only the expected features.
#     X_background = df_encoded[expected_features].iloc[:100]
#     X_background_transformed = preprocessor.transform(X_background)
# except Exception as e:
#     logging.error("Error loading background sample: %s", e)
#     X_background_transformed = np.zeros((1, len(expected_features)))

# # Define a prediction function that returns the probability for class 1.
# def predict_prob(x):
#     return model.predict_proba(x)[:, 1]

# # Create SHAP KernelExplainer using the defined prediction function.
# explainer_shap = shap.KernelExplainer(predict_prob, X_background_transformed)

# def get_risk_level(risk_score):
#     """Determine risk level based on risk score."""
#     # Using dictionary mapping for clarity without explicit if/else.
#     risk_map = {range(0, 31): 'Low', range(31, 71): 'Moderate', range(71, 101): 'High'}
#     return next(val for key, val in risk_map.items() if risk_score in key)

# def get_custom_explanation(feature, value, contribution):
#     """
#     Dynamically generates an explanation using template dictionaries and lookup tables,
#     without explicit if/else branches.
#     """
#     # Template dictionary for explanations.
#     explanation_templates = {
#         "domain": "The domain '{value}' is usually linked with established organizations and its contribution of {contribution:.4f} tends to {verb} the overall risk.",
#         "lastReportedAt": "The timestamp '{value}' shows when the IP was last reported, which tends to {verb} risk by {contribution:.4f}.",
#         "isp": "The ISP '{value}' is a recognized provider, and its characteristics tend to {verb} the risk (contribution: {contribution:.4f}).",
#         "countryCode": "Country code '{value}' often reflects regional cybersecurity trends that {verb} risk, as shown by a contribution of {contribution:.4f}.",
#         "usageType": "Usage type '{value}' is associated with common usage patterns that typically {verb} risk (contribution: {contribution:.4f}).",
#         "numDistinctUsers": "Having {value} distinct users reporting this IP suggests widespread observation that {verb} risk by {contribution:.4f}.",
#         "totalReports": "A total report count of {value} generally tends to {verb} the risk (contribution: {contribution:.4f}).",
#         "isWhitelisted": "The whitelist status '{value}' usually reduces risk (contribution: {contribution:.4f}).",
#         "isTor": "TOR node status '{value}' is known to be associated with anonymity and can {verb} risk (contribution: {contribution:.4f})."
#     }
    
#     # Verb mapping based on sign of the contribution (using np.sign; no explicit if/else).
#     verb_mapping = {1: "increase", -1: "decrease", 0: "have no impact on"}
#     verb = verb_mapping[int(np.sign(contribution))]
    
#     # Get the template for the feature; use a default template if not found.
#     template = explanation_templates.get(
#         feature,
#         "The feature '{value}' contributes {contribution:.4f} and tends to {verb} the overall risk."
#     )
    
#     return template.format(value=value, contribution=contribution, verb=verb)

# @app.route('/api/predict', methods=['POST'])
# def predict():
#     try:
#         data = request.json
#         features = data.get('features')
#         if features is None:
#             raise ValueError("Missing 'features' in JSON payload.")
#         logging.info("Input Features: %s", features)
        
#         # Create a DataFrame from input features.
#         df_input = pd.DataFrame([features])
        
#         # (Optional) If isWhitelisted is null, treat it as False.
#         df_input['isWhitelisted'] = df_input.get('isWhitelisted', False)
        
#         # Keep only the expected features.
#         df_input = df_input[expected_features]
        
#         # Process categorical features using the encoder.
#         encoded_categorical = encoder.transform(df_input[expected_categorical_features])
#         encoded_df = pd.DataFrame(encoded_categorical, columns=expected_categorical_features)
        
#         # Extract numeric features.
#         df_numeric = df_input[expected_numeric_features]
        
#         # Combine numeric and encoded categorical features.
#         new_data_encoded = pd.concat([df_numeric, encoded_df], axis=1)
        
#         # Transform using the preprocessor.
#         new_data_transformed = preprocessor.transform(new_data_encoded)
        
#         # If the IP is whitelisted, immediately return low risk.
#         if df_input['isWhitelisted'].iloc[0]:
#             risk_score_prob = 0.0
#             risk_score = 0
#             risk_level = 'Low'
#             base_value = 0
#             shap_summary = []
#         else:
#             # Get predicted probability for abusive class.
#             risk_score_prob = predict_prob(new_data_transformed)[0]
#             risk_score = int(risk_score_prob * 100)
#             risk_level = get_risk_level(risk_score)
            
#             # SHAP explanation.
#             base_value = float(explainer_shap.expected_value)
#             shap_values = explainer_shap.shap_values(new_data_transformed)
            
#             # Pair feature names with their contributions dynamically.
#             shap_summary = [
#                 {
#                     "feature": feature,
#                     "value": features.get(feature, "N/A"),
#                     "contribution": float(contribution),
#                     "explanation": get_custom_explanation(feature, features.get(feature, "N/A"), contribution)
#                 }
#                 for feature, contribution in zip(new_data_encoded.columns.tolist(), shap_values[0])
#                 if abs(contribution) > 1e-6
#             ]
        
#         # Build a calculation summary string for transparency.
#         calc_summary = (
#             f"risk_score_prob = model.predict_proba(x)[:,1] -> {risk_score_prob:.4f}, "
#             f"riskScore = int(risk_score_prob * 100) = {risk_score}, "
#             f"base_value = {base_value if base_value is not None else 'N/A'}, "
#             f"sum(SHAP values) = {np.sum(shap_values[0]) if not df_input['isWhitelisted'].iloc[0] else 'N/A'} "
#             f"(base_value + sum(SHAP) should equal risk_score_prob)"
#         )
        
#         response = {
#             'riskScore': risk_score,
#             'riskLevel': risk_level,
#             'baseValue': base_value,
#             'shapSummary': shap_summary,
#             'calculation': calc_summary
#         }
#         return jsonify(response)
#     except Exception as e:
#         logging.error("Error in prediction: %s", e)
#         return jsonify({'error': str(e)}), 500

# if __name__ == '__main__':
#     app.run(debug=True, port=5001)


from flask import Flask, request, jsonify
import joblib
import shap
import numpy as np
import pandas as pd
import logging
import os

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# Load the pre-trained pipeline and encoder.
pipeline = joblib.load('model/risk_model.pkl')
encoder = joblib.load('encoder.pkl')

# Extract classifier and preprocessor from the pipeline.
model = pipeline.named_steps['classifier']
preprocessor = pipeline.named_steps['preprocessor']

# Define the expected features (order must match training).
expected_numeric_features = ["totalReports", "numDistinctUsers", "lastReportedAt", "isWhitelisted", "isTor"]
expected_categorical_features = ["countryCode", "usageType", "domain", "isp"]
expected_features = expected_numeric_features + expected_categorical_features

# Load the initial dataset
df_encoded = pd.read_csv("synthetic_data_encoded.csv")

# Prepare a background dataset for SHAP.
try:
    df_encoded = pd.read_csv("synthetic_data_encoded.csv")
    # Keep only the expected features.
    X_background = df_encoded[expected_features].iloc[:100]
    X_background_transformed = preprocessor.transform(X_background)
except Exception as e:
    logging.error("Error loading background sample: %s", e)
    X_background_transformed = np.zeros((1, len(expected_features)))

# Define a prediction function that returns the probability for class 1.
def predict_prob(x):
    return model.predict_proba(x)[:, 1]

# Create SHAP KernelExplainer using the defined prediction function.
explainer_shap = shap.KernelExplainer(predict_prob, X_background_transformed)

# Function to update the dataset with new IP data
def update_dataset(new_data, max_size=10000):
    global df_encoded
    df_encoded = pd.concat([df_encoded, new_data], ignore_index=True)
    if len(df_encoded) > max_size:
        df_encoded = df_encoded.iloc[-max_size:]

    # Save the updated dataset
    df_encoded.to_csv("synthetic_data_encoded.csv", index=False)

def get_risk_level(risk_score):
    """Determine risk level based on risk score."""
    # Using dictionary mapping for clarity without explicit if/else.
    risk_map = {range(0, 31): 'Low', range(31, 71): 'Moderate', range(71, 101): 'High'}
    return next(val for key, val in risk_map.items() if risk_score in key)

def get_custom_explanation(feature, value, contribution):
    """
    Dynamically generates an explanation using template dictionaries and lookup tables,
    without explicit if/else branches.
    """
    # Template dictionary for explanations.
    explanation_templates = {
        "domain": "The domain '{value}' is usually linked with established organizations and its contribution of {contribution:.4f} tends to {verb} the overall risk.",
        "lastReportedAt": "The timestamp '{value}' shows when the IP was last reported, which tends to {verb} risk by {contribution:.4f}.",
        "isp": "The ISP '{value}' is a recognized provider, and its characteristics tend to {verb} the risk (contribution: {contribution:.4f}).",
        "countryCode": "Country code '{value}' often reflects regional cybersecurity trends that {verb} risk, as shown by a contribution of {contribution:.4f}.",
        "usageType": "Usage type '{value}' is associated with common usage patterns that typically {verb} risk (contribution: {contribution:.4f}).",
        "numDistinctUsers": "Having {value} distinct users reporting this IP suggests widespread observation that {verb} risk by {contribution:.4f}.",
        "totalReports": "A total report count of {value} generally tends to {verb} the risk (contribution: {contribution:.4f}).",
        "isWhitelisted": "The whitelist status '{value}' usually reduces risk (contribution: {contribution:.4f}).",
        "isTor": "TOR node status '{value}' is known to be associated with anonymity and can {verb} risk (contribution: {contribution:.4f})."
    }
    
    # Verb mapping based on sign of the contribution (using np.sign; no explicit if/else).
    verb_mapping = {1: "increase", -1: "decrease", 0: "have no impact on"}
    verb = verb_mapping[int(np.sign(contribution))]
    
    # Get the template for the feature; use a default template if not found.
    template = explanation_templates.get(
        feature,
        "The feature '{value}' contributes {contribution:.4f} and tends to {verb} the overall risk."
    )
    
    return template.format(value=value, contribution=contribution, verb=verb)

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        features = data.get('features')
        if features is None:
            raise ValueError("Missing 'features' in JSON payload.")
        logging.info("Input Features: %s", features)
        
        # Create a DataFrame from input features.
        df_input = pd.DataFrame([features])
        
        # (Optional) If isWhitelisted is null, treat it as False.
        df_input['isWhitelisted'] = df_input.get('isWhitelisted', False)
        
        # Keep only the expected features.
        df_input = df_input[expected_features]
        
        # Process categorical features using the encoder.
        encoded_categorical = encoder.transform(df_input[expected_categorical_features])
        encoded_df = pd.DataFrame(encoded_categorical, columns=expected_categorical_features)
        
        # Extract numeric features.
        df_numeric = df_input[expected_numeric_features]
        
        # Combine numeric and encoded categorical features.
        new_data_encoded = pd.concat([df_numeric, encoded_df], axis=1)
        
        # Update the dataset with the new data dynamically
        update_dataset(new_data_encoded)
        
        # Transform using the preprocessor.
        new_data_transformed = preprocessor.transform(new_data_encoded)
        
        # If the IP is whitelisted, immediately return low risk.
        if df_input['isWhitelisted'].iloc[0]:
            risk_score_prob = 0.0
            risk_score = 0
            risk_level = 'Low'
            base_value = 0
            shap_summary = []
        else:
            # Get predicted probability for abusive class.
            risk_score_prob = predict_prob(new_data_transformed)[0]
            risk_score = int(risk_score_prob * 100)
            risk_level = get_risk_level(risk_score)
            
            # SHAP explanation.
            base_value = float(explainer_shap.expected_value)
            shap_values = explainer_shap.shap_values(new_data_transformed)
            
            # Pair feature names with their contributions dynamically.
            shap_summary = [
                {
                    "feature": feature,
                    "value": features.get(feature, "N/A"),
                    "contribution": float(contribution),
                    "explanation": get_custom_explanation(feature, features.get(feature, "N/A"), contribution)
                }
                for feature, contribution in zip(new_data_encoded.columns.tolist(), shap_values[0])
                if abs(contribution) > 1e-6
            ]
        
        # Build a calculation summary string for transparency.
        calc_summary = (
            f"risk_score_prob = model.predict_proba(x)[:,1] -> {risk_score_prob:.4f}, "
            f"riskScore = int(risk_score_prob * 100) = {risk_score}, "
            f"base_value = {base_value if base_value is not None else 'N/A'}, "
            f"sum(SHAP values) = {np.sum(shap_values[0]) if not df_input['isWhitelisted'].iloc[0] else 'N/A'} "
            f"(base_value + sum(SHAP) should equal risk_score_prob)"
        )
        
        response = {
            'riskScore': risk_score,
            'riskLevel': risk_level,
            'baseValue': base_value,
            'shapSummary': shap_summary,
            'calculation': calc_summary
        }
        return jsonify(response)
    except Exception as e:
        logging.error("Error in prediction: %s", e)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5001)