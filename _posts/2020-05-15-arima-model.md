---
layout: post
title: "Introduction to ARIMA Model"
---

Author: [Jinit Sanghvi](https://www.linkedin.com/in/jinit-sanghvi-4329a016b/)

# Introduction to ARIMA Model

## A brief introduction followed by implementation
<center>
<img src="https://www.freestockcharts.com/help/Content/Resources/Images/timeseriesforecast.png">
</center>
<p style="text-align:center;">Taken from <a href="https://www.freestockcharts.com/">freestockcharts.com</a></p>

ARIMA or Autoregressive Integrated Moving Average is a purely-regressive model and this model is used for forecasting values of a time series. A time series is essentially a sequence of data points or observations taken at different instances. Time Series are very common to find given how time-dependent most of the worldly schemes and variables are. To better understand this, take a look at the Stock Market or weather reports gathered over a timeline and you'll observe patterns which are highly time-dependent.
<br>
In this post, we'll first cover the theory and then move on to the code

### Theory

One of the best ways to be introduced to the model is to understand why we're using it, especially when other effective regression models like linear regression or multivariate regression exists. For time-series forecasting, why do we prefer the ARIMA model? Most of the time-series data available online focuses on the dependent variable, and how the dependent variable changes with time. For models like linear regression, we need independent variables to map a function from dependent variables to dependent variables for prediction.
However, this is not always possible because:-
<br>
<ol>
    <li>Independent Variables are not always available.</li>
        <li>In quite a few scenarios, too many independent variables exist and we may find it difficult to find enough dependent variables to sufficiently explain the behavior of the time-series.</li>
</ol>
<i>On the other hand, ARIMA leverages the correlation between the values of the time-series and time, allowing it to better understand the relation between independent variables and time. Since the model focuses more on patterns and behavior of the time-series instead of finding out which factors affect these dependent variables, the model is able to forecast on the basis of these recognized patterns and behavior.</i>
<hr>
<i>To better understand this model, let's break it down into 2 parts and then integrate those 2 parts in the end.</i>

## AR or Autoregressive Model

The intuition behind this model is that observations at different time-steps are related to the previous value of the variable. For example, on a hot sunny day, you predict that the next day will be hotter because you've noticed increasing temperatures. Similarly, the AR model finds the correlation between future time-steps and past time-steps to forecast future values.

$$Y_t  = \alpha + \beta_1Y_{t-1} + \beta_2Y_{t-2} + \dots + \beta_pY_{t-p} + \epsilon_1$$

<!--![AR Equation](https://latex.codecogs.com/gif.latex?Y_t%20%3D%20%5Calpha%20&plus;%20%5Cbeta_1*Y_t_-_1%20&plus;%20%5Cbeta_2*Y_t_-_2%20..%20&plus;%20%5Cbeta_p*Y_t_-_p%20&plus;%20%5Cepsilon_1)-->
_By the above equation, we can see how we can reduce this to a regression problem and statistically find the correlation between future values and earlier time-steps._

<hr>

## Moving Average Model

The intuition behind this model in nature is of reinforcement, i.e, a moving average model tries to learn from the previous errors it has committed and tries to tweak itself accordingly. To better understand this, take a look at the equation below:

$$Y_t = \alpha + \epsilon_t + \phi_1\epsilon_{t - 1} + \phi_2\epsilon_{t-2} + \dots + \phi_q\epsilon_{t-q}$$

<small>But, what does $$\epsilon$$ signify? Simply put, it is the error or the difference between the actual value and the predicted value.</small>
<br>

<i>Since this model tries to learn from its mistakes, it is better able to account for unpredictable changes in value and is able to correct itself to provide more accurate results and predictions.</i>
<hr>
Now that you've understood the Autoregressive Model and the Moving Averages model, it's time to learn about ARIMA. When the Autoregressive Terms and the Moving Average terms are combined together with differencing to make the time-series stationary(more on this later), we get the ARIMA Model! Since the equation is regressive in nature, we can find the respective weights of the terms in the equation using regression techniques.


$$Y_t = \alpha + \beta_1Y_{t-1} + \beta_2Y_{t-2} + \dots + \beta_pY_{t-p} + \phi_1\epsilon_{t - 1} + \phi_2\epsilon_{t-2} + \dots + \phi_q\epsilon_{t-q}$$
<!--![ARIMA Equation](https://latex.codecogs.com/gif.latex?Y_t%20%3D%20%5Calpha%20&plus;%20%5Cbeta%20_1Y_t_-_1%20&plus;%20%5Cbeta%20_2Y_t_-_2%20&plus;%20....%20&plus;%20%5Cbeta%20_pY_t_-_p%20&plus;%20%5Cphi%20_1%5Cepsilon_t_-_1%20&plus;%20%5Cphi%20_2%5Cepsilon_t_-_2%20&plus;%20....%20&plus;%20%5Cphi%20_q%5Cepsilon_t_-_q)-->

<hr>

> So far, we've understood the basic intuition behind the ARIMA Model. Let's dig a bit deeper and understand the parameters of an ARIMA model.

> Consider a list below, and assume that every successive element of the list is a successive time-step or observation.

$$[1, 3, 5, 4]$$

> Now, when we difference the list, we subtract the nth value of the series with the (n-1)th value of the series. For a better understanding:

> After First Differencing:

$$[3–1, 5–3, 4–5] = [2, 2, -1]$$

> After Second Differencing:

$$[2–2, -1–2] = [0, -3]$$

<p>The reason why we difference the time-series is to make the time-series stationary, i.e, the mean and the variance of the time-series remains constant/stable over time which allows us to reduce components like trend and seasonality(illustrated later). This is important because ARIMA expects the time-series to be stationary. Thus, we keep differencing the time-series till it becomes stationary.</p>

<i>Now that we have understood all the underlying concepts that we're going to utilize, let's learn how to find these parameters and dive right into the code!</i>


> To begin with, let's start with some basic imports.


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['text.color'] = 'k'
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
%matplotlib inline
```

For this post, I'm going to use a time-series from the <em>Huge Stock Market Dataset</em>. This is just a sample time-series used to further ground the concepts you read about. At the end of this post, you'll be able to apply these concepts to any other time-series you want to.


```python
aus1 = pd.read_csv("a.us.txt",sep=',', index_col=0, parse_dates=True, squeeze=True) 
```


```python
aus1.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Open</th>
      <th>High</th>
      <th>Low</th>
      <th>Close</th>
      <th>Volume</th>
      <th>OpenInt</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1999-11-18</th>
      <td>30.713</td>
      <td>33.754</td>
      <td>27.002</td>
      <td>29.702</td>
      <td>66277506</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1999-11-19</th>
      <td>28.986</td>
      <td>29.027</td>
      <td>26.872</td>
      <td>27.257</td>
      <td>16142920</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1999-11-22</th>
      <td>27.886</td>
      <td>29.702</td>
      <td>27.044</td>
      <td>29.702</td>
      <td>6970266</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1999-11-23</th>
      <td>28.688</td>
      <td>29.446</td>
      <td>27.002</td>
      <td>27.002</td>
      <td>6332082</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1999-11-24</th>
      <td>27.083</td>
      <td>28.309</td>
      <td>27.002</td>
      <td>27.717</td>
      <td>5132147</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
aus1.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Open</th>
      <th>High</th>
      <th>Low</th>
      <th>Close</th>
      <th>Volume</th>
      <th>OpenInt</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>4521.000000</td>
      <td>4521.000000</td>
      <td>4521.000000</td>
      <td>4521.000000</td>
      <td>4.521000e+03</td>
      <td>4521.0</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>27.856296</td>
      <td>28.270442</td>
      <td>27.452486</td>
      <td>27.871357</td>
      <td>3.993503e+06</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>std</th>
      <td>12.940880</td>
      <td>13.176000</td>
      <td>12.711735</td>
      <td>12.944389</td>
      <td>2.665730e+06</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>min</th>
      <td>7.223100</td>
      <td>7.513900</td>
      <td>7.087800</td>
      <td>7.323800</td>
      <td>0.000000e+00</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>19.117000</td>
      <td>19.435000</td>
      <td>18.780000</td>
      <td>19.089000</td>
      <td>2.407862e+06</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>24.456000</td>
      <td>24.809000</td>
      <td>24.159000</td>
      <td>24.490000</td>
      <td>3.460621e+06</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>36.502000</td>
      <td>37.046000</td>
      <td>35.877000</td>
      <td>36.521000</td>
      <td>4.849809e+06</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>max</th>
      <td>105.300000</td>
      <td>109.370000</td>
      <td>97.881000</td>
      <td>107.320000</td>
      <td>6.627751e+07</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
aus1.index
```




    DatetimeIndex(['1999-11-18', '1999-11-19', '1999-11-22', '1999-11-23',
                   '1999-11-24', '1999-11-26', '1999-11-29', '1999-11-30',
                   '1999-12-01', '1999-12-02',
                   ...
                   '2017-10-30', '2017-10-31', '2017-11-01', '2017-11-02',
                   '2017-11-03', '2017-11-06', '2017-11-07', '2017-11-08',
                   '2017-11-09', '2017-11-10'],
                  dtype='datetime64[ns]', name='Date', length=4521, freq=None)



> Before we use this time-series, we first need to resample it into a time-series wherein each observation differs by a month.


```python
y1 = aus1['Open'].resample('MS').mean()
y2 = aus1['Close'].resample('MS').mean()
y1.index
```




    DatetimeIndex(['1999-11-01', '1999-12-01', '2000-01-01', '2000-02-01',
                   '2000-03-01', '2000-04-01', '2000-05-01', '2000-06-01',
                   '2000-07-01', '2000-08-01',
                   ...
                   '2017-02-01', '2017-03-01', '2017-04-01', '2017-05-01',
                   '2017-06-01', '2017-07-01', '2017-08-01', '2017-09-01',
                   '2017-10-01', '2017-11-01'],
                  dtype='datetime64[ns]', name='Date', length=217, freq='MS')



> So far, so good! Let's plot the time-series and see what it looks like!


```python
y1.plot(figsize=(15, 6))
plt.show()
```


![png](/assets/notebooks/2020-05-15-arima-code_files/2020-05-15-arima-code_10_0.png)


> Now that we've seen what it looks like, let's try breaking it down into its components. Let's try extracting the trend, the seasonality and the residual of our time-series.


```python
from pylab import rcParams
rcParams['figure.figsize'] = 18, 8
decomposition = sm.tsa.seasonal_decompose(y1, model='additive')
fig = decomposition.plot()
plt.show()
```


![png](/assets/notebooks/2020-05-15-arima-code_files/2020-05-15-arima-code_12_0.png)


<em>Our inference from breaking the time-series down into components like seasonality and trend is that the given time-series is not stationary. Since the trend and seasonality of the time-series affect its values at different time-steps, this time-series is not stationary as its mean and variance keeps changing. Thus, we have to difference the time-series at least once to make it stationary.</em>

If you remember, we discussed the three parameters of the ARIMA Model. In order to determine these 3 parameters and finding the best combination of these 3 parameters in order to yield the best results, we'll use a combination of statistical tools and iteration (trying out different models to check which achieves the best results). 


Let's start by determining $$d$$. As we mentioned earlier, we'll keep differencing the time-series until it becomes stationary. Let's try differencing it once and use the Augmented Dickey Fuller Test to find if it is stationary or not.


```python
from statsmodels.tsa.stattools import adfuller
from numpy import log
y1_d = y1.diff()
y1_d = y1_d.dropna()
result = adfuller(y1_d)
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
```

    ADF Statistic: -2.955284
    p-value: 0.039287
    

<em>As we see that the $$p$$-value obtained is less than 0.05, we can reject the null hypothesis and say that the series is stationary. We can also see that the series has become more stationary from the plot above. Thus, obtaining a $$p$$-value of less than 0.05 using ADF is a good indicator that our time-series has become stationary.</em>


```python
y1_d.plot(figsize=(15, 6))
plt.show()
```


![png](/assets/notebooks/2020-05-15-arima-code_files/2020-05-15-arima-code_18_0.png)


In order to determine $$p$$, we need to first understand what an ACF plot is. Autocorrelation Function plot or ACF plot shows how correlated the variable is with its previous time-steps. For example, while predicting stock prices, the ACF plot tries to show how correlated the stock prices of March are to the stock prices of February, how likely are the stock prices in March to follow the behavior of stock prices in February. To better understand this, let's plot it.


```python
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
```


```python
plot_acf(y1_d,lags=10)

```




![png](/assets/notebooks/2020-05-15-arima-code_files/2020-05-15-arima-code_21_0.png)




![png](/assets/notebooks/2020-05-15-arima-code_files/2020-05-15-arima-code_21_1.png)


<small>We can see that the auto-correlation is strong for the first 2 lags and then it decays and oscillates in the blue region.</small>

<em>Inferring from this, we can see that the first 1–2 lags show high correlation and values keep decreasing in the blue region.</em><br>
Thus, we can say that the value of $$p$$ will lie in the range spanning from 1 to 2.
<hr>

Let's try determining $$q$$ now. The process of determining $$q$$ is very similar to the process of determining $$p$$. Instead of using ACF, we'll now use a PACF, or a Partial Autocorrelation Function. A PACF plot shows how correlated a variable is with itself at a previous time-step, ignoring all linear dependencies it has on time-steps that lie between the time-step against which correlation is to be found and the current time-step. Also called a conditional correlation fucntion, PACF aims to find the correlation between two time-steps independent of all other time-steps in between. Let's understand this better with a plot.


```python
plot_pacf(y1_d,lags=10)
```




![png](/assets/notebooks/2020-05-15-arima-code_files/2020-05-15-arima-code_25_0.png)




![png](/assets/notebooks/2020-05-15-arima-code_files/2020-05-15-arima-code_25_1.png)


<em>From the plot we can see that the plot cuts off to zero at the second lag so we can estimate that the value of $$q$$ will be lesser than 2.</em><br>
Thus, we can say that the value $$q$$ shall lie in the range spanning from 0 to 1.
<hr>

> You've finally learnt the necessary building blocks required to create an ARIMA model in order to perform time-series forecasting! Let's jump right into it!

We divide our time-frame into a training and a testing set so that we can later gauge how well our model is performing. Training data is 70% of the original time-series data and testing data is 30% of the original time-series data.


```python
size = int(len(y1) * 0.7)
train, test = y1[0:size], y1[size:len(y1)]
series = [y1 for y1 in train]
predictions = []
```

> Alright, now let's start training the model! Let's try using values of $$(p,d,q)$$ equal to (2,1,1) and see how the model performs!


```python
for y in range(len(test)):
    arima_model = ARIMA(series, order=(2,1,1))
    model_fit = arima_model.fit(disp=0)
    preds = model_fit.forecast()
    pred = preds[0]
    predictions.append(pred)
    actual_val = test[y]
    series.append(actual_val)
error = mean_squared_error(test, predictions)
print('Test error is {}'.format(error))
plt.plot(predictions,'r')
plt.plot(np.array(test))
```

    Test error is 3.273019781729989
    




    [<matplotlib.lines.Line2D at 0x1d39598f668>]




![png](/assets/notebooks/2020-05-15-arima-code_files/2020-05-15-arima-code_31_2.png)


<em>We can see that the model has learnt the nature and behavior of the time-series, and is performing pretty well on the test set!</em>

> Let's try out new values of $$(p,d,q)$$. Let's try out (1,1,1) this time and see if we obtain better results.


```python
size = int(len(y1) * 0.7)
train, test = y1[0:size], y1[size:len(y1)]
series = [y1 for y1 in train]
predictions = []
```


```python
for y in range(len(test)):
    arima_model = ARIMA(series, order=(1,1,1)) 
    model_fit = arima_model.fit(disp=0)
    preds = model_fit.forecast()
    pred = preds[0]
    predictions.append(pred)
    actual_val = test[y]
    series.append(actual_val)
error = mean_squared_error(test, predictions)
print('Test error is {}'.format(error))
plt.plot(predictions,'r')
plt.plot(np.array(test))
```

    Test error is 2.355571525013936
    




    [<matplotlib.lines.Line2D at 0x1d3959f32b0>]




![png](/assets/notebooks/2020-05-15-arima-code_files/2020-05-15-arima-code_35_2.png)


> Great, we have obtained better results by changing the parameters of the model! By using the statistical methods above and by trying out different values, you can achieve great results on various different time-series.


<strong>I hope you're now acquainted with various concepts of Time-Series Forecasting and the ARIMA Model!</strong>


You can get the jupyter notebook corresponding to this blog [here](/assets/notebooks/2020-05-15-arima-code.ipynb)
