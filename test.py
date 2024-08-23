import streamlit as st
import joblib
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk


# Load the saved model and label encoders
model_filename = 'health_chatbot_model.pkl'
label_encoder_filename = 'label_encoders.pkl'

model = joblib.load(model_filename)
le_symptom, le_condition = joblib.load(label_encoder_filename)

# Load the data
df = pd.read_csv('symptoms_conditions1.csv')

# Create a dictionary of symptoms
symptom_dict = dict(zip(df['Symptom'], le_symptom.transform(df['Symptom'])))

def extract_symptoms(text):
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    symptoms = [word.capitalize() for word in tokens if word.capitalize() in symptom_dict]
    return symptoms

def predict_treatment(text):
    symptoms = extract_symptoms(text)
    if not symptoms:
        return "No recognized symptoms found in the text."

    results = []
    for symptom in symptoms:
        encoded_symptom = symptom_dict[symptom]

        # Find the most common condition for this symptom
        condition = df[df['Symptom'] == symptom]['Condition'].mode().iloc[0]
        encoded_condition = le_condition.transform([condition])[0]

        prediction = model.predict([[encoded_symptom, encoded_condition]])[0]

        results.append({
            'Symptom': symptom,
            'Condition': condition,
            'Medicine': prediction[0],
            'Precaution': prediction[1],
            'Treatment': prediction[2],
            'Faculty': prediction[3],
            'Location': prediction[4]
        })

    return results

# Streamlit App
st.title("Health Chatbot")

st.write("Enter your symptoms below:")

user_input = st.text_input("Symptoms:")

if st.button("Predict"):
    if user_input:
        results = predict_treatment(user_input)
        if isinstance(results, str):
            st.write(results)
        else:
            for r in results:
                st.write(f"**Symptom:** {r['Symptom']}")
                st.write(f"**Condition:** {r['Condition']}")
                st.write(f"**Medicine:** {r['Medicine']}")
                st.write(f"**Precaution:** {r['Precaution']}")
                st.write(f"**Treatment:** {r['Treatment']}")
                st.write(f"**Faculty:** {r['Faculty']}")
                st.write(f"**Location:** {r['Location']}")
                st.write("---")
    else:
        st.write("Please enter some symptoms.")
