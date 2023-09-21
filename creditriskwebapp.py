import pickle
import sklearn
import numpy as np
import streamlit as st
# To Display Images
from PIL import Image

# loading the saved model
loaded_model = pickle.load(open('creditrisk.sav', 'rb'))


# creating a function for Prediction

def credit_prediction(input_data):
    # changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)

    if (prediction[0] == 0):
        return 'The person defaulted'
    else:
        return 'The person paid back'


def main():
    # display image
    img = Image.open("creditpic.png")
    new_image = img.resize((700, 200))
    st.image(new_image)
    # let's display
    # st.image(img, width=700)

    # giving a title
    st.title('Credit Risk Prediction Web App')

    # getting the input data from the user

    Age = st.number_input('Input your Age')
    Income = st.number_input('What is your Income')
    Home = st.number_input('Home Type: 3 = OTHER, 2 = RENT, 1 = OWN, 0 = MORTGAGE')
    Intent = st.number_input('Loan: 5 = EDUCATION, 4 = MEDICAL, 3 = VENTURE, 2 = PERSONAl, 1 = HOMEIMPROVEMENT, 0 = DEBTCONSOLIDATION')
    Amount = st.number_input('How much loan do you need')


    # code for Prediction
    creditrisk = ''

    # creating a button for Prediction

    if st.button('Credit Risk Result'):
        creditrisk = credit_prediction([Age, Income, Home, Intent, Amount])

    st.success(creditrisk)

if __name__ == '__main__':
    main()
