import joblib as jb
import pandas as pd
import streamlit as st

model = jb.load(r'D:\Codes\MachineLearning\models\SalesModel.pkl')
st.title('Advertising Budget Prediction')
tv = st.number_input('TV', step=1)
radio = st.number_input('Radio', step=1)
newspaper = st.number_input('Newspaper', step=1)

new_data = pd.DataFrame({'TV': tv, 'radio': radio, 'newspaper': newspaper}, index=[0])

result=''

if st.button('Predict'):
  result = model.predict(new_data)
  st.subheader("Predicted Sale of items (in thousand units)")
  st.subheader(result)
else:
  st.subheader("Enter ad Budget and click Predict button.")


# streamlit run 24.streamlit.py
