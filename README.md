﻿# MediAI - Medical Symptom Analysis Platform

MediAI is an advanced medical symptom analysis platform that combines artificial intelligence with comprehensive medical datasets to provide quick, reliable health insights. The system helps users understand their symptoms and make informed decisions about their health.

## 🌟 Features

- **AI-Powered Symptom Analysis**: Advanced machine learning algorithms analyze symptoms with high accuracy
- **Comprehensive Health Insights**: Detailed recommendations for medications, diet, and lifestyle changes
- **Interactive Interface**: User-friendly symptom input with auto-suggestions
- **Multi-faceted Recommendations**: 
  - Precautions
  - Medications
  - Diet suggestions
  - Exercise recommendations
- **Local Medical Help**: Find nearby hospitals and medical facilities
- **AI Chat Assistant**: Interactive medical chatbot for health-related queries

## 🔧 Technical Architecture

### Machine Learning Model
- Primary Model: Support Vector Classification (SVC)
- Backup Model: ChatGPT-4 for edge cases
- Training Accuracy: ~92%
- Validation Accuracy: ~89%

### Datasets Used
1. `minimal_symptoms.csv`: Core dataset for disease prediction
   - 4920 samples
   - 41 diseases
   - 95 unique symptoms
   
2. `precautions_df.csv`: Precautionary measures
   - Disease-specific precautions
   - 4-point recommendation system

3. `medications.csv`: Medication recommendations
   - Generic medicine names
   - Usage instructions
   - Common dosages

4. `diets.csv`: Dietary recommendations
   - Disease-specific diet plans
   - Food restrictions
   - Nutritional guidelines

5. `workout_df.csv`: Exercise recommendations
   - Condition-specific exercises
   - Activity levels
   - Safety precautions

6. `description.csv`: Disease descriptions
   - Detailed medical explanations
   - Common causes
   - Risk factors

7. `minimal_severity.csv`: Symptom severity scoring
   - Weighted severity scores
   - Risk assessment metrics

## 🛠️ Implementation Approach

### 1. Data Preprocessing
- Text normalization for symptoms
- Feature encoding
- Severity score calculation
- Data validation and cleaning

### 2. Model Pipeline
1. Symptom Input Processing
   - Text normalization
   - Symptom validation
   - Minimum 2 symptoms required
   - Maximum 10 symptoms allowed

2. Disease Prediction
   - Primary: SVC model prediction
   - Secondary: Severity assessment
   - Fallback: ChatGPT-4 for unusual cases

3. Recommendation Generation
   - Disease-specific precautions
   - Medication suggestions
   - Dietary guidelines
   - Exercise recommendations

### 3. User Interface
- Modern, responsive design
- Interactive symptom input
- Real-time suggestions
- Clear result presentation
- Mobile-friendly layout

## 🚀 Technologies Used

- **Backend**: Python, Flask
- **Frontend**: HTML5, CSS3, JavaScript
- **Machine Learning**: scikit-learn, TensorFlow
- **Database**: SQLite
- **API Integration**: OpenAI GPT-4
- **UI Framework**: Bootstrap 5

## 📊 Performance Metrics

- Disease Prediction Accuracy: ~89%
- Average Response Time: <2 seconds
- Symptom Recognition Rate: >95%
- User Satisfaction Rate: 4.2/5

## 🔒 Security & Privacy

- No personal health data storage
- Encrypted communications
- HIPAA-compliant design principles
- Anonymous analysis

## 🎯 Future Enhancements

1. Integration with Electronic Health Records
2. Mobile application development
3. Multi-language support
4. Advanced symptom image analysis
5. Real-time doctor consultation
6. Personalized health tracking

## ⚠️ Disclaimer

This platform is designed for informational purposes only and should not be considered as a substitute for professional medical advice. Always consult with a qualified healthcare provider for proper diagnosis and treatment.

## 📝 License

MIT License - See LICENSE file for details

## 👥 Contributors

- Project Members - Raj Desai, Suyog Rawool, Yuvraj Ghadi, Rahul Gurav
