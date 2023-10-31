import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_model import ARIMA

# Muat model ARIMA dari file .sav
model_file = 'forecast-corn-ar.sav'
model = pickle.load(open(model_file, 'rb'))

# Muat dataset AirPassengers.csv
data_file = pd.read_csv("/content/corn2015-2017/corn2013-2017.txt",sep=',',header=None, names=['date','price'])
data = pd.read_csv(data_file)
data['date'] = pd.to_datetime(data['date'])
data.set_index('Month', inplace=True)

# Judul aplikasi
st.title('ARIMA Forecasting App')

# Slider untuk menentukan jumlah bulan yang akan diprediksi
forecast_steps = st.slider('Jumlah Bulan Prediksi', 1, 36, 12)

# Tombol "Prediksi"
if st.button('Prediksi'):
    # Prediksi dengan model ARIMA
    forecast = model.forecast(steps=forecast_steps)
    
    # Tampilkan data asli
    st.subheader('Data Asli')
    st.line_chart(data)

    # Tampilkan hasil prediksi
    st.subheader('Hasil Prediksi')
    st.line_chart(forecast)
        
    # Tampilkan grafik data asli dengan hasil prediksi
    st.subheader('Grafik Data Asli dengan Hasil Prediksi')
    fig, ax = plt.subplots()
    data['#Passengers'].plot(style='--', color='gray', legend=True, label='Data Asli', ax=ax)
    forecast.plot(color='b', legend=True, label='Prediksi', ax=ax)
    st.pyplot(fig)
    
    # Tampilkan tabel hasil prediksi
    st.subheader('Tabel Hasil Prediksi')
    forecast_df = pd.DataFrame({
    'Tanggal': pd.date_range(start=data.index[-1], periods=forecast_steps, freq='M'),
    'Prediksi': forecast
    })
    st.dataframe(forecast_df)
