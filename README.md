# Ex.No:04   FIT ARMA MODEL FOR TIME SERIES
# Date:12/05/2026



### AIM:
To implement ARMA model in python.
### ALGORITHM:
1. Import necessary libraries.
2. Set up matplotlib settings for figure size.
3. Define an ARMA(1,1) process with coefficients ar1 and ma1, and generate a sample of 1000

data points using the ArmaProcess class. Plot the generated time series and set the title and x-
axis limits.

4. Display the autocorrelation and partial autocorrelation plots for the ARMA(1,1) process using
plot_acf and plot_pacf.
5. Define an ARMA(2,2) process with coefficients ar2 and ma2, and generate a sample of 10000

data points using the ArmaProcess class. Plot the generated time series and set the title and x-
axis limits.

6. Display the autocorrelation and partial autocorrelation plots for the ARMA(2,2) process using
plot_acf and plot_pacf.
### PROGRAM:
```
# Import Required Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# -----------------------------------
# Load Salary Prediction Dataset
# -----------------------------------
data = pd.read_csv('/content/salary_prediction_data.csv')

# Display First 5 Rows
print(data.head())

# -----------------------------------
# Select Salary Column
# -----------------------------------
X = data['Salary']

# Declare Required Variables
N = 1000

# Set Figure Size
plt.rcParams['figure.figsize'] = [12, 6]

# -----------------------------------
# Visualize Original Salary Data
# -----------------------------------
plt.plot(X)
plt.title('Original Salary Data')
plt.xlabel('Index')
plt.ylabel('Salary')
plt.show()

# -----------------------------------
# Plot ACF and PACF of Original Data
# -----------------------------------
plt.subplot(2, 1, 1)

plot_pacf(X, lags=30, ax=plt.gca())
plt.title('Original Salary PACF')

plt.subplot(2, 1, 2)

plot_acf(X, lags=30, ax=plt.gca())
plt.title('Original Salary ACF')

plt.tight_layout()
plt.show()

# ===================================
# ARMA(1,1) MODEL
# ===================================

# Fit ARMA(1,1) Model
arma11_model = ARIMA(X, order=(1, 0, 1)).fit()

# Get Parameters
phi1_arma11 = arma11_model.params['ar.L1']
theta1_arma11 = arma11_model.params['ma.L1']

# -----------------------------------
# Simulate ARMA(1,1) Process
# -----------------------------------
ar1 = np.array([1, -phi1_arma11])
ma1 = np.array([1, theta1_arma11])

ARMA_1 = ArmaProcess(ar1, ma1).generate_sample(nsample=N)

# Plot Simulated Process
plt.plot(ARMA_1)
plt.title('Simulated ARMA(1,1) Process')
plt.xlim([0, 500])
plt.show()

# -----------------------------------
# Plot PACF and ACF for ARMA(1,1)
# -----------------------------------
plt.subplot(2,1,1)

plot_pacf(ARMA_1, lags=30, ax=plt.gca())
plt.title('Partial Autocorrelation')

plt.subplot(2,1,2)

plot_acf(ARMA_1, lags=30, ax=plt.gca())
plt.title('Autocorrelation')

plt.tight_layout()
plt.show()

# ===================================
# ARMA(2,2) MODEL
# ===================================

# Fit ARMA(2,2) Model
arma22_model = ARIMA(X, order=(2, 0, 2)).fit()

# Get Parameters
phi1_arma22 = arma22_model.params['ar.L1']
phi2_arma22 = arma22_model.params['ar.L2']

theta1_arma22 = arma22_model.params['ma.L1']
theta2_arma22 = arma22_model.params['ma.L2']

# -----------------------------------
# Simulate ARMA(2,2) Process
# -----------------------------------
ar2 = np.array([1, -phi1_arma22, -phi2_arma22])

ma2 = np.array([1, theta1_arma22, theta2_arma22])

ARMA_2 = ArmaProcess(ar2, ma2).generate_sample(nsample=N)

# Plot Simulated Process
plt.plot(ARMA_2)
plt.title('Simulated ARMA(2,2) Process')
plt.xlim([0, 500])
plt.show()

# -----------------------------------
# Plot PACF and ACF for ARMA(2,2)
# -----------------------------------
plt.subplot(2,1,1)

plot_pacf(ARMA_2, lags=30, ax=plt.gca())
plt.title('Partial Autocorrelation')

plt.subplot(2,1,2)

plot_acf(ARMA_2, lags=30, ax=plt.gca())
plt.title('Autocorrelation')

plt.tight_layout()
plt.show()

# ===================================
# RESULT
# ===================================
print("RESULT:")
print("Thus, a python program is created to fit ARMA models using Salary Prediction Dataset successfully.")
```

OUTPUT:
ORGINAL SALARY DATA:

<img width="1030" height="550" alt="image" src="https://github.com/user-attachments/assets/0c8e2d79-518a-444f-bfc2-4760bda256b6" />

ORGINAL SALARY PACF AND ACF:


<img width="1120" height="552" alt="image" src="https://github.com/user-attachments/assets/ae697e97-a5f5-43ac-b529-4fd3a33f1cf6" />


SIMULATED ARMA(1,1) PROCESS:

<img width="927" height="492" alt="image" src="https://github.com/user-attachments/assets/281836f1-3280-4747-aad3-ad5aafebf8f4" />


Partial Autocorrelation

<img width="1121" height="275" alt="image" src="https://github.com/user-attachments/assets/bd165f73-39f4-4c87-8a03-4bfe78ea53bf" />


Autocorrelation

<img width="1121" height="287" alt="image" src="https://github.com/user-attachments/assets/9d441250-9205-4e76-bf44-2d1068c093c1" />


SIMULATED ARMA(2,2) PROCESS:

<img width="931" height="502" alt="image" src="https://github.com/user-attachments/assets/ab89b816-56f5-429c-83bd-f6e368167104" />


Partial Autocorrelation

<img width="1122" height="272" alt="image" src="https://github.com/user-attachments/assets/a6a96332-d8ea-48f3-93a6-42cb28331d3f" />


Autocorrelation

<img width="1121" height="272" alt="image" src="https://github.com/user-attachments/assets/a47015c5-15b1-4269-ab3f-76e52ed1c471" />


RESULT:
Thus, a python program is created to fir ARMA Model successfully.
