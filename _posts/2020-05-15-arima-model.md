---
layout: post
title: "Introduction to ARIMA Model"
---

Author: [Jinit Sanghvi](www.linkedin.com/in/jinit-sanghvi-4329a016b)

### Introduction to ARIMA Model

<h3>A brief introduction followed by implementation.</h3>
<img src="https://images.app.goo.gl/iCGZ1GFnqkZxW18B8">
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
<h2>AR or Autoregressive Model:</h2>
<p>The intuition behind this model is that observations at different time-steps are related to the previous value of the variable. For example, on a hot sunny day, you predict that the next day will be hotter because you've noticed increasing temperatures. Similarly, the AR model finds the correlation between future time-steps and past time-steps to forecast future values.<p>

![AR Equation](https://latex.codecogs.com/gif.latex?Y_t%20%3D%20%5Calpha%20&plus;%20%5Cbeta_1*Y_t_-_1%20&plus;%20%5Cbeta_2*Y_t_-_2%20..%20&plus;%20%5Cbeta_p*Y_t_-_p%20&plus;%20%5Cepsilon_1)

<i>By the above equation, we can see how we can reduce this to a regression problem and statistically find the correlation between future values and earlier time-steps.</i>
<hr>
<h2>Moving Average Model:</h2>
<p>The intuition behind this model in nature is of reinforcement, i.e, a moving average model tries to learn from the previous errors it has committed and tries to tweak itself accordingly. To better understand this, take a look at the equation below:</p>
![MA_Equation](https://latex.codecogs.com/gif.latex?Y_t%20%3D%20%5Calpha%20&plus;%20%5Cepsilon_t%20&plus;%20%5Cphi_1%20%5Cepsilon_t-1%20&plus;%20%5Cphi_2%20%5Cepsilon_t-2%20&plus;%20...%20&plus;%20%5Cphi_q%20%5Cepsilon_t-q)
<small>But, what does epsilon signify? Simply put, it is the error or the difference between the actual value and the predicted value.</small>
<br>

<i>Since this model tries to learn from its mistakes, it is better able to account for unpredictable changes in value and is able to correct itself to provide more accurate results and predictions.</i>
<hr>
Now that you've understood the Autoregressive Model and the Moving Averages model, it's time to learn about ARIMA. When the Autoregressive Terms and the Moving Average terms are combined together with differencing to make the time-series stationary(more on this later), we get the ARIMA Model! Since the equation is regressive in nature, we can find the respective weights of the terms in the equation using regression techniques.

![ARIMA Equation](https://latex.codecogs.com/gif.latex?Y_t%20%3D%20%5Calpha%20&plus;%20%5Cbeta%20_1Y_t_-_1%20&plus;%20%5Cbeta%20_2Y_t_-_2%20&plus;%20....%20&plus;%20%5Cbeta%20_pY_t_-_p%20&plus;%20%5Cphi%20_1%5Cepsilon_t_-_1%20&plus;%20%5Cphi%20_2%5Cepsilon_t_-_2%20&plus;%20....%20&plus;%20%5Cphi%20_q%5Cepsilon_t_-_q)

<hr>

<h3>So far, we've understood the basic intuition behind the ARIMA Model. Let's dig a bit deeper and understand the parameters of an ARIMA model.</h3>

<h2 style="font-style:italic;">Consider a list below, and assume that every successive element of the list is a successive time-step or observation.</h2>
<h3>[1,3,5,4]</h3>

<h2 style="font-style:italic;">Now, when we difference the list, we subtract the nth value of the series with the (n-1)th value of the series. For a better understanding:</h2>

<h3>After First Differencing:</h3>
<h3>[3–1,5–3,4–5] = [2,2,-1]</h3>
<h3>After Second Differencing:</h3>
<h3>[2–2,-1–2] = [0,-3]</h3>

<p>The reason why we difference the time-series is to make the time-series stationary, i.e, the mean and the variance of the time-series remains constant/stable over time which allows us to reduce components like trend and seasonality(illustrated later). This is important because ARIMA expects the time-series to be stationary. Thus, we keep differencing the time-series till it becomes stationary.</p>

<i>Now that we have understood all the underlying concepts that we're going to utilize, let's learn how to find these parameters and dive right into the code!</i>
