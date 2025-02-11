import streamlit as st
from src.pipeline.predict_pipeline import CustomData,PredictionPipeline
from src.logger import logging

# Title of the app
st.title("Students' Performance Prediction")

st.text("Enter the details below to predict performance")
# Input fields
gender = st.text_input("Gender")
race_ethnicity = st.text_input("Race-Ethnicity")
parental_level_of_education = st.text_input("Parent's Education Level")
lunch = st.text_input("Type of Lunch")
test_preparation_course = st.text_input("test_preparation_course")
reading_score = st.text_input("Reading Score")
writing_score = st.text_input("Writing Score")

logging.info(f"The values are:{gender,race_ethnicity,parental_level_of_education,lunch,test_preparation_course,reading_score,writing_score}")
print(gender,race_ethnicity,parental_level_of_education,lunch,test_preparation_course,reading_score,writing_score)

if gender and race_ethnicity and parental_level_of_education and lunch and test_preparation_course and reading_score and writing_score:

    input_data = CustomData(
                     gender=gender.strip(),
                     race_ethnicity=race_ethnicity.strip(),
                     parental_level_of_education=parental_level_of_education.strip(),
                     lunch=lunch.strip(),
                     test_preparation_course=test_preparation_course.strip(),
                     reading_score=int(reading_score),
                     writing_score=int(writing_score)
    )

    # input as dataframe
    inp_df = input_data.get_data_as_dataframe()

    #Prediction
    prediction = PredictionPipeline()
    result = prediction.predict(features=inp_df)

    # Button to trigger an action
    if st.button("Predict"):
        st.session_state.submitted = True
        st.success(f"Prediction is : {result[0]}")
    else:
        st.error("Something is Fishy !!!!")

else:
    print("One or more inputs are empty. Please fill all fields.")