import streamlit as st
import requests


def main():

    st.title("Email Classification")
    message = st.text_input('Enter Text to Classify')

    if st.button('Predict'):
        payload = {
            "Text": message
        }
        print(payload)
        res = requests.post(f"http://service:8000/predict/",json=payload )
        with st.spinner('Classifying, please wait....'):
            st.write(res.json())




if __name__ == '__main__':
    main()