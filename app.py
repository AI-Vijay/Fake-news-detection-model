import streamlit as st
import joblib

# Load the saved model and vectorizer
model1 = joblib.load('LR_model.pkl')
model2 = joblib.load('DT_model.pkl')
model3 = joblib.load('GB_model.pkl')
model4 = joblib.load('RF_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# Streamlit interface
st.title("Fake News Detection")

# Input text box for users to enter news content
user_input = st.text_area("Enter news content to classify:")

if st.button('Classify'):
    # Preprocess the user input using the same vectorizer
    input_tfidf = vectorizer.transform([user_input])
    
    # Make the prediction
    prediction = model1.predict(input_tfidf)
    # Display the result
    if prediction == 1:
        st.write("This is Fake News!")
    else:
        st.write("This is Real News!")

    # Make the prediction
    prediction = model2.predict(input_tfidf)
    # Display the result
    if prediction == 1:
        st.write("This is Fake News!")
    else:
        st.write("This is Real News!")
    
     # Make the prediction
    prediction = model3.predict(input_tfidf)
    # Display the result
    if prediction == 1:
        st.write("This is Fake News!")
    else:
        st.write("This is Real News!")
        
 # Make the prediction
    prediction = model4.predict(input_tfidf)
    # Display the result
    if prediction == 1:
        st.write("This is Fake News!")
    else:
        st.write("This is Real News!")

