from flask import Flask, request, render_template, jsonify
import numpy as np
import pandas as pd
import pickle
import openai
import os
from dotenv import load_dotenv
import traceback
import logging
from typing import List, Dict, Tuple, Optional
from openai import OpenAI

# Load environment variables and initialize OpenAI
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Flask app
app = Flask(__name__)

# Initialize OpenAI client
try:
    client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    logger.info("OpenAI client initialized successfully")
except Exception as e:
    logger.error(f"Error initializing OpenAI client: {str(e)}")
    raise

# Load datasets
try:
    sym_des = pd.read_csv("datasets/symptoms_df.csv")
    precautions = pd.read_csv("datasets/precautions_df.csv")
    workout = pd.read_csv("datasets/workout_df.csv")
    description = pd.read_csv("datasets/description.csv")
    medications = pd.read_csv('datasets/medications.csv')
    diets = pd.read_csv("datasets/diets.csv")
    training_data = pd.read_csv('datasets/minimal_symptoms.csv')
    severity_data = pd.read_csv('datasets/minimal_severity.csv')
    logger.info("Successfully loaded all datasets")
except Exception as e:
    logger.error(f"Error loading datasets: {str(e)}")
    raise

# Load model
try:
    svc = pickle.load(open('models/svc.pkl','rb'))
    logger.info("Successfully loaded SVC model")
except Exception as e:
    logger.error(f"Error loading SVC model: {str(e)}")
    raise

# Create symptoms dictionary
all_symptoms = []
for col in training_data.columns[1:]:
    all_symptoms.extend(training_data[col].unique())
symptoms_dict = {symptom: idx for idx, symptom in enumerate(set(all_symptoms))}

# Create severity dictionary
severity_dict = dict(zip(severity_data.iloc[:, 0], severity_data.iloc[:, 1]))

def calculate_severity_score(symptoms: List[str]) -> int:
    """Calculate severity score based on symptoms"""
    score = 0
    for symptom in symptoms:
        if symptom in severity_dict:
            score += severity_dict[symptom]
    return score

def get_severity_level(score: int) -> str:
    """Determine severity level based on score"""
    if score <= 3:
        return "Low"
    elif score <= 6:
        return "Moderate"
    else:
        return "High"

def get_predicted_value(user_symptoms: List[str]) -> Tuple[str, float]:
    """Match user symptoms with diseases and return prediction with confidence"""
    try:
        # Validate input
        if len(user_symptoms) < 2:
            raise ValueError("Minimum 2 symptoms required")
        if len(user_symptoms) > 10:
            raise ValueError("Maximum 10 symptoms allowed")
            
        # Convert user symptoms to lowercase for matching
        user_symptoms = [s.strip().lower() for s in user_symptoms]
        logger.info(f"Processing symptoms: {user_symptoms}")
        
        # First attempt: Match against minimal_symptoms.csv
        disease_matches = {}
        for _, row in training_data.iterrows():
            disease = row['Disease']
            disease_symptoms = [
                str(row['Symptom_1']).lower(),
                str(row['Symptom_2']).lower(),
                str(row['Symptom_3']).lower(),
                str(row['Symptom_4']).lower()
            ]
            matches = len(set(user_symptoms) & set(disease_symptoms))
            if matches > 0:
                disease_matches[disease] = matches
        
        if disease_matches:
            predicted_disease = max(disease_matches.items(), key=lambda x: x[1])[0]
            confidence = max(disease_matches.values()) / len(user_symptoms)
            logger.info(f"Found match in dataset: {predicted_disease} with confidence {confidence}")
            return predicted_disease, confidence
        
        # Second attempt: Use SVC model
        try:
            input_vector = np.zeros(len(symptoms_dict))
            for symptom in user_symptoms:
                if symptom in symptoms_dict:
                    input_vector[symptoms_dict[symptom]] = 1
            
            prediction = svc.predict([input_vector])[0]
            confidence = max(svc.predict_proba([input_vector])[0])
            logger.info(f"SVC prediction: {prediction} with confidence {confidence}")
            return prediction, confidence
        except Exception as e:
            logger.warning(f"SVC prediction failed: {str(e)}")
        
        raise ValueError("Could not generate prediction")
        
    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}")
        raise

def generate_disease_description(disease_name: str) -> str:
    """Generate disease description using GPT-4"""
    try:
        logger.info(f"Generating description for: {disease_name}")
        
        # Ensure API key is set
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OpenAI API key not found")
            
        # Initialize OpenAI client
        client = OpenAI(api_key=api_key)
        
        # Create the prompt
        prompt = f"""As a medical expert, provide a clear and concise description of {disease_name}.
        Include:
        - Definition and nature of the condition
        - Key symptoms and characteristics
        - Common causes if known
        - How it affects the body
        
        Keep it informative yet understandable for patients. Use 2-3 sentences."""
        
        # Make API call with new format
        completion = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a medical expert. Provide accurate, clear, and concise disease descriptions."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=200,
            temperature=0.3
        )
        
        # Extract description from response
        if completion.choices and hasattr(completion.choices[0].message, 'content'):
            description = completion.choices[0].message.content.strip()
            if description:
                logger.info(f"Successfully generated description: {description}")
                return description
            
        raise ValueError("No valid description generated")
        
    except Exception as e:
        logger.error(f"Error generating description: {str(e)}")
        traceback.print_exc()
        
        # Fallback to database description
        try:
            desc_df = pd.read_csv("datasets/description.csv")
            if disease_name.lower() in desc_df['Disease'].str.lower().values:
                desc = desc_df[desc_df['Disease'].str.lower() == disease_name.lower()]['Description'].iloc[0]
                if pd.notna(desc) and desc.strip():
                    logger.info(f"Using database description for {disease_name}")
                    return desc
        except Exception as db_error:
            logger.error(f"Database fallback failed: {str(db_error)}")
        
        return "We apologize, but we're having trouble generating a description at the moment. Please try again."

def helper(dis: str) -> Dict[str, List[str]]:
    """Get all recommendations for a disease"""
    try:
        result = {
            'precautions': [],
            'medications': [],
            'diet': [],
            'workout': []
        }
        
        # Get precautions
        if dis in precautions['Disease'].values:
            prec = precautions[precautions['Disease'] == dis].iloc[0]
            result['precautions'] = [prec[f'Precaution_{i}'] for i in range(1, 5) if pd.notna(prec[f'Precaution_{i}'])]
        
        # Get medications
        if dis in medications['Disease'].values:
            med = medications[medications['Disease'] == dis].iloc[0]
            try:
                if isinstance(med['Medication'], str):
                    med_list = eval(med['Medication'])
                    if isinstance(med_list, list):
                        result['medications'] = [m.strip() for m in med_list if m and isinstance(m, str)]
            except:
                med_str = med['Medication'].strip('[]').replace("'", "").replace('"', '')
                result['medications'] = [m.strip() for m in med_str.split(',') if m.strip()]
        
        # Get diet recommendations
        if dis in diets['Disease'].values:
            diet = diets[diets['Disease'] == dis].iloc[0]
            try:
                if isinstance(diet['Diet'], str):
                    diet_list = eval(diet['Diet'])
                    if isinstance(diet_list, list):
                        result['diet'] = [d.strip() for d in diet_list if d and isinstance(d, str)]
            except:
                diet_str = diet['Diet'].strip('[]').replace("'", "").replace('"', '')
                result['diet'] = [d.strip() for d in diet_str.split(',') if d.strip()]
        
        # Get workout recommendations
        if dis in workout['disease'].values:
            workout_recs = workout[workout['disease'] == dis]['workout'].tolist()
            result['workout'] = [w.strip() for w in workout_recs if w and isinstance(w, str)]
        
        logger.info(f"Retrieved recommendations for {dis}")
        return result
        
    except Exception as e:
        logger.error(f"Error getting recommendations: {str(e)}")
        return {
            'precautions': ["Recommendations not available"],
            'medications': ["Recommendations not available"],
            'diet': ["Recommendations not available"],
            'workout': ["Recommendations not available"]
        }

@app.route("/")
def landing():
    """Render the landing page"""
    return render_template('landing.html')

@app.route("/analyze")
def analyze():
    """Render the analysis page"""
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    """Handle disease prediction requests"""
    try:
        if request.method == 'POST':
            symptoms = request.form.get('symptoms', '').split(',')
            
            # Validate input
            if len(symptoms) < 2:
                return render_template('index.html', error="Please enter at least 2 symptoms")
            if len(symptoms) > 10:
                return render_template('index.html', error="Maximum 10 symptoms allowed")
            
            # Process symptoms
            user_symptoms = [s.strip().lower() for s in symptoms]
            
            # Get prediction
            predicted_disease, confidence = get_predicted_value(user_symptoms)
            
            if not predicted_disease:
                return render_template('index.html', error="Could not make a prediction")
            
            # Get severity
            severity_score = calculate_severity_score(user_symptoms)
            severity_level = get_severity_level(severity_score)
            
            # Always generate description using ChatGPT-4o
            desc = generate_disease_description(predicted_disease)
            
            # Get recommendations
            recommendations = helper(predicted_disease)
            
            # Log the results
            logger.info(f"Prediction results for {predicted_disease}:")
            logger.info(f"Description: {desc}")
            logger.info(f"Precautions: {recommendations['precautions']}")
            logger.info(f"Medications: {recommendations['medications']}")
            logger.info(f"Diet: {recommendations['diet']}")
            logger.info(f"Workout: {recommendations['workout']}")
            
            return render_template('index.html',
                                predicted_disease=predicted_disease,
                                description=desc,
                                precautions=recommendations['precautions'],
                                medications=recommendations['medications'],
                                diet=recommendations['diet'],
                                workout=recommendations['workout'],
                                severity_level=severity_level,
                                severity_score=severity_score,
                                confidence=confidence)
        
        return render_template('index.html')
    except Exception as e:
        logger.error(f"Error in prediction route: {str(e)}")
        traceback.print_exc()
        return render_template('index.html', error="An error occurred. Please try again.")

@app.route('/about')
def about():
    """Render the about page"""
    return render_template('about.html')

@app.route('/contact')
def contact():
    """Render the contact page"""
    return render_template('contact.html')

@app.route('/developer')
def developer():
    """Render the developer page"""
    return render_template('developer.html')

@app.route('/blog')
def blog():
    """Render the blog page"""
    return render_template('blog.html')

if __name__ == '__main__':
    app.run(debug=True, port=5001)