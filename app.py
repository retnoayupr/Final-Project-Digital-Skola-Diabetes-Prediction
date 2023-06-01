import streamlit as st
import pickle


def main():
    background = """<div style ='background-color:black; padding:13px'>
                    <h1 style ='color:white'>Diabetes Prediction</h1>
                </div>"""
    st.markdown(background, unsafe_allow_html=True)

    st.write('Enter the following details to predict whether you have diabetes or not.')

    # Create input fields for the user to enter the details
    left, right = st.columns((2,2))
    pregnancies = left.number_input('Pregnancies', min_value=0, max_value=17, step=1, value=0)
    glucose = right.number_input('Glucose', min_value=0, max_value=199, value=0)
    blood_pressure = left.number_input('Blood Pressure', min_value=0, max_value=122, value=0)
    skin_thickness = right.number_input('Skin Thickness', min_value=0, max_value=99, value=0)
    insulin = left.number_input('Insulin', min_value=0, max_value=846, value=0)
    bmi = right.number_input('BMI', min_value=0.0, max_value=67.1, value=0.0)
    diabetes_pedigree_function = left.number_input('Diabetes Pedigree Function', min_value=0.0, max_value=2.42, value=0.0)
    age = right.number_input('Age', min_value=0, max_value=81, value=0)
    button = st.button('Predict')

    #kalau button di klik
    if button:
        #mengecek apakah ada nilai 0
        if glucose == 0 or blood_pressure == 0 or skin_thickness == 0 or insulin == 0 or bmi == 0.0 or diabetes_pedigree_function == 0.0 or age == 0:
            st.error('Not allowed to enter a value of 0 except for pregnancies.')
        else:
            result = predict_diabetes(pregnancies, glucose, blood_pressure, skin_thickness, insulin, 
                                      bmi, diabetes_pedigree_function, age)
            st.success(f'You {result} diabetes.')

def predict_diabetes(pregnancies, glucose, blood_pressure, skin_thickness, insulin, 
                    bmi, diabetes_pedigree_function, age):
    
    #Load the trained model
    with open ('model/model_xgb.pkl', 'rb') as file:
        XGB_Model = pickle.load(file)

    #Membuat prediksi
    prediction = XGB_Model.predict([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, 
                    bmi, diabetes_pedigree_function, age]])
    verdict = 'do not have' if prediction == 0 else 'have'
    
    return verdict

if __name__ == "__main__":
    main()                
  



