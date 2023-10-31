# Laporan Proyek Machine Learning
### Nama : Alan Hermawan
### Nim : 211351010
### Kelas : IF Malam B

## Domain Proyek

Projek ini dibuat untuk membantu petani dan pembeli jagung dalam mendapatkan informasi terkait harga jagung kedepannya berdasarkan index waktu

## Business Understanding

Prediksi harga jagung ini diharapkan dapat membantu memberikan informasi terkait harga jagung kedepannya sehingga petani maupun penjual atau pembeli memiliki informasi terkait harga jagung kedepannya

Bagian laporan ini mencakup:

### Problem Statements

Menjelaskan pernyataan masalah latar belakang:
- Harga jagung yang selalu berubah dari hari ke hari

### Goals

Menjelaskan tujuan dari pernyataan masalah:
- Dapat membantu siapapun yang membutuhkan informasi terkait perubahan harga jagung

    ### Solution statements
    - Pembuatan sistem yang mempermudah dalam memberikan informasi harga jagung yang berubah-ubah berdasarkan index waktu parminggunya sehingga siapapun yang membutuhkan informasi tersebut.
    - Sistem yang dibuat menggunakan dataset yang diambil dari kaggle dan diproses menggunakan 3 algoritma yang berbeda yang mana selanjutnya akan dipilih algoritma terbaik untuk dipakai didalam aplikasi tersebut

## Data Understanding
Dataset yang diambil dari kaggle ini berisi 2 kolom yaitu tanggal dan harga jagung. 

Dataset: [Weekly Corn Price](https://www.kaggle.com/datasets/nickwong64/corn2015-2017).

Dalam proses data understanding ini tahapan pertama yang dilakukan adalah:
1. import dataset

dikarenakan dataset diambil dari kaggle maka kita perlu import token kaggle:
```
from google.colab import files
files.upload()
```
lalu buat directory:
```
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
!ls ~/.kaggle
```
selanjutnya download datasetnya:
```
!kaggle datasets download -d nickwong64/corn2015-2017
```
setelah itu kita unzip file yang sudah di download:
```
!mkdir corn2015-2017
!unzip corn2015-2017.zip -d corn2015-2017
!ls corn2015-2017
```
import library yang akan digunakan:
```
import pandas as pd
import numpy as np

# library untuk visualisasi
import matplotlib.pyplot as plt
import seaborn as sns

# library untuk analisis time series
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# library yang digunakan untuk forecasting
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from statsmodels.tsa.statespace.sarimax import SARIMAX
```
Setelah itu baru kita import datasetnya:
```
df = pd.read_csv("/content/corn2015-2017/corn2013-2017.txt",sep=',',header=None, names=['date','price'])
```
2. menampilkan 5 baris pertama dataset
```
df.head()
```
3. cek tipe data
```
df.info()
```
4. cek ukuran dataset
```
df.shape
```
(248, 2)
dataset tersebut berisi 144 baris dengan 2 kolom
5. Null Check
```
df.isnull().sum()
```
Tidak terdapat data yang kosong

Selanjutnya uraikanlah seluruh variabel atau fitur pada data. Sebagai contoh:  

### Variabel-variabel pada Heart Failure Prediction Dataset adalah sebagai berikut:
- date : Tanggal (Tahun-Bulan) ```object```
- price : harga jagung ```float64```

## Data Preparation
Dikarenakan kolom Month memiliki tipe data object maka harus kita convert menjadi datetime:
```
df['date'] = pd.to_datetime(df['date'])
```
lalu kita set index dari kolom month untuk menjadi acuan dalam melakukan forecasting:
```
df.set_index("date",inplace=True)
```
kita tampilkan juga bagaimana grafik dari perubahan jumlah penumpang pasawatnya:
```
df['price'].plot(figsize=(12,5));
```
![image](https://github.com/alanhrmwn/harga-jagung/assets/148874522/92be6692-09c4-40b2-ab37-9783b0a2396a)

cek index max dan min:
```
df.index.min(), df.index.max()
```
(Timestamp('2013-01-06 00:00:00', freq='W-SUN'),
 Timestamp('2017-10-01 00:00:00', freq='W-SUN'))

Selanjutnya kita bagi terlebih dahulu antara train dan test data:
```
train = df.iloc[:170]
test = df.iloc[171:]
```
Sekarang mari kita lihat selisih antara setiap dua poin data berturut-turut dalam sebuah rangkaian data deret waktu (time series):
```
diff_df = df.diff()
diff_df.head()
```
lalu kita hapus kolom yang berisi null values:
```
diff_df.dropna(inplace=True)
```
Sekarang kita lakukan uji adfuller yaitu uji statistik yang digunakan untuk mengevaluasi apakah sebuah deret waktu stasioner atau tidak:
```
result = adfuller(diff_df)
# The result is a tuple that contains various test statistics and p-values
# You can access specific values as follows:
adf_statistic = result[0]
p_value = result[1]

# Print the results
print(f'ADF Statistic: {adf_statistic}')
print(f'p-value: {p_value}')
```
ADF Statistic: -13.496765258501243<br>
p-value: 3.0312210481240657e-25

Selanjutnya kita cek korelasi dari deret waktunya:
```
plot_acf(diff_df)
plot_pacf(diff_df)
```
![image](https://github.com/alanhrmwn/harga-jagung/assets/148874522/4064193e-26ce-4399-b14b-61f783f61504)

## Modeling
Ditahap modeling ini kita akan menggunakan 3 algoritma yang mana akan kita bandingkan algoritma terbaik yang selanjutnya akan dipakai untuk aplikasi tersebut.

kita akan coba untuk memprediksi 77 minggu kedepan:

  ### Single Exponential Smoothing
```
single_exp = SimpleExpSmoothing(train).fit()
single_exp_train_pred = single_exp.fittedvalues
single_exp_test_pred = single_exp.forecast(77)
```
```
train['price'].plot(style='--', color='gray', legend=True, label='train')
test['price'].plot(style='--', color='r', legend=True, label='test')
single_exp_test_pred.plot(color='b', legend=True, label='Prediction')
```
![image](https://github.com/alanhrmwn/harga-jagung/assets/148874522/56fb96a5-1465-43b2-9940-5c9b245e3530)


```
Train_RMSE_SES = mean_squared_error(train, single_exp_train_pred)**0.5
Test_RMSE_SES = mean_squared_error(test, single_exp_test_pred)**0.5
Train_MAPE_SES = mean_absolute_percentage_error(train, single_exp_train_pred)
Test_MAPE_SES = mean_absolute_percentage_error(test, single_exp_test_pred)

print('Train RMSE :',Train_RMSE_SES)
print('Test RMSE :', Test_RMSE_SES)
print('Train MAPE :', Train_MAPE_SES)
print('Test MAPE :', Test_MAPE_SES)
```
Train RMSE : 0.16140010141419225<br>
Test RMSE : 0.41554115341252534<br>
Train MAPE : 0.018756005557876605<br>
Test MAPE : 0.09890883033731582<br>

  ## Double Exponential Smoothing
```
double_exp = ExponentialSmoothing(train, trend=None, initialization_method='heuristic', seasonal='add', seasonal_periods=29, damped_trend=False).fit()
double_exp_train_pred = double_exp.fittedvalues
double_exp_test_pred = double_exp.forecast(77)
```
```
train['price'].plot(style='--', color='gray', legend=True, label='train')
test['price'].plot(style='--', color='r', legend=True, label='test')
double_exp_test_pred.plot(color='b', legend=True, label='Prediction')
```
![image](https://github.com/alanhrmwn/harga-jagung/assets/148874522/6b55ab9f-d1de-46f0-97b3-78c1d8bab8df)


```
Train_RMSE_DES = mean_squared_error(train, double_exp_train_pred)**0.5
Test_RMSE_DES = mean_squared_error(test, double_exp_test_pred)**0.5
Train_MAPE_DES = mean_absolute_percentage_error(train, double_exp_train_pred)
Test_MAPE_DES = mean_absolute_percentage_error(test, double_exp_test_pred)

print('Train RMSE :',Train_RMSE_DES)
print('Test RMSE :', Test_RMSE_DES)
print('Train MAPE :', Train_MAPE_DES)
print('Test MAPE :', Test_MAPE_DES)
```
Train RMSE : 0.16205098371914398<br>
Test RMSE : 0.4488396674833522<br>
Train MAPE : 0.020458390748657466<br>
Test MAPE : 0.1016007247389691<br>

  ## ARIMA
```
ar = ARIMA(train, order=(15,1,15)).fit()
ar_train_pred = ar.fittedvalues
ar_test_pred = ar.forecast(77)
```
```
train['price'].plot(style='--', color='gray', legend=True, label='train')
test['price'].plot(style='--', color='r', legend=True, label='test')
ar_test_pred.plot(color='b', legend=True, label='Prediction')
```
![image](https://github.com/alanhrmwn/harga-jagung/assets/148874522/2232e70a-cb6a-4467-a23f-936fd4f2861d)

```
Train_RMSE_AR = mean_squared_error(train, ar_train_pred)**0.5
Test_RMSE_AR = mean_squared_error(test, ar_test_pred)**0.5
Train_MAPE_AR = mean_absolute_percentage_error(train, ar_train_pred)
Test_MAPE_AR = mean_absolute_percentage_error(test, ar_test_pred)

print('Train RMSE :',Train_RMSE_AR)
print('Test RMSE :', Test_RMSE_AR)
print('Train MAPE :', Train_MAPE_AR)
print('Test MAPE :', Test_MAPE_AR)
```
Train RMSE : 0.6153128426935569<br>
Test RMSE : 0.38981683508614867<br>
Train MAPE : 0.02442687290853558<br>
Test MAPE : 0.09223393707251747<br>

Selanjutnya mari kita evaluasi 3 algoritma tersebut

## Evaluation
Pada tahap evaluasi ini kita akan membandingkan nilai Root Mean Squared Error (RMSE) dan Mean Absolute Percentage Error (MAPE) yang mana selanjutnya akan kita urutkan mana nilai RMSE yang paling kecil maka algoritma tersebut yang akan kita pakai:
```
comparision_df = pd.DataFrame(data=[
    ['Single Exp Smoothing', Test_RMSE_SES, Test_MAPE_SES],
    ['Double Exp Smoothing', Test_RMSE_DES, Test_MAPE_DES],
    ['ARIMA', Test_RMSE_AR, Test_MAPE_AR]
    ],
    columns=['Model', 'RMSE', 'MAPE'])
comparision_df.set_index('Model', inplace=True)
```
```
comparision_df.sort_values(by='RMSE')
```
<img width="222" alt="image" src="https://github.com/alanhrmwn/harga-jagung/assets/148874522/1c9bdb4e-33c6-4f66-8061-0aeac188521e">


dapat dilihat jika nilai RMSE dan MAPE ada pada algoritma ARIMA, maka dari itu algoritma yang akan dipakai adalah algoritma ARIMA

## Deployment
Link Aplikasi: [Forecasting harga jagung](https://forecast-jagung.streamlit.app/)

