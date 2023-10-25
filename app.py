import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
from prophet.diagnostics import performance_metrics
from prophet.diagnostics import cross_validation
from prophet.plot import plot_cross_validation_metric
from prophet.plot import plot_plotly, plot_components_plotly
from prophet.plot import add_changepoints_to_plot
from prophet.plot import plot_yearly
import base64

with st.sidebar:
        """
         With the help of this web application, you can forecast key healthcare metrics such as patient admissions, drug sales, or disease outbreak trends. Designed for healthcare professionals, institutions, or researchers with data sets ready for analysis and forecasting, the steps include:
          1. Importing the dataset
          2. Specifying the forecast interval
          3. Analyzing data visualizations
          4. Downloading the forecast

        """         
        
# st.image('healthcare_forecast.jpg') 

st.title('Automated Prediction for Healthcare Metrics')

"""
This application aids in crafting precise forecasts tailored to healthcare. Whether you're forecasting patient admissions, medical equipment demand, or disease trends, accurate data is the foundation. Start by gathering a coherent set of data for your healthcare institution or research. Afterward, follow the 4 steps in this web application to derive a forecast rooted in your data.
"""                     


"""
### Step 1: Import the dataset
"""
df = st.file_uploader('Upload your dataset here. The dataset should be structured with date of records and the desired metric. The date column must be labeled "ds" and follow this format: YYYY-MM-DD (Example: 2019-05-20). The metric column should be named "y", representing the numerical value you wish to forecast. Acceptable file format: csv. See the sample file structure below.', type='csv')
st.info(
            f"""
                👆 First, upload the .csv file. [Example pedestrians covid](https://raw.githubusercontent.com/BohdanTarchanin/healthcare-metrics-prediction/master/example_pedestrians_covid.csv)
                """
        )

if df is not None:
    data = pd.read_csv(df)
    data['ds'] = pd.to_datetime(data['ds'],errors='coerce') 
    
    st.write(data)
    
    max_date = data['ds'].max()
    st.write(max_date)
    st.success('👆 This marks the latest date in your dataset')

"""
### Step 2: Specify the duration for the forecast

"""

periods_input = st.number_input('How many days would you like the forecast to cover? Input a number between 1 and 365 then press Enter.',
min_value = 1, max_value = 365)

if df is not None:
    m = Prophet()
    m.fit(data)
   
            
"""
### Step 3: Visualization of forecasted data

The table below presents the forecasting results for the target metric "y", based on your historical dataset.
* ds - forecast dates
* yhat - predicted value for the metric
* yhat_lower - probable lower bound for the prediction
* yhat_upper - probable upper bound for the prediction

"""
if df is not None:
    future = m.make_future_dataframe(periods=periods_input)
    
    forecast = m.predict(future)
    fcst = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]

    fcst_filtered =  fcst[fcst['ds'] > max_date]    
    st.write(fcst_filtered)
    
    """
    The bar chart below illustrates the date (ds) against the target metric (y) and the prediction interval. The black dots represent the actual figures from the dataset, while the blue line is the forecast.
    """
    fig1 = m.plot(forecast)
    st.write(fig1)
    
    st.info('Graph showcasing date dependency on the target metric alongside the forecast.')

    """
    Up next, the graphs display the trend of the target metric against date and its seasonal correlation.
    """
    fig2 = m.plot_components(forecast)
    st.write(fig2)
      
    st.info('Graphs visualizing the trend and seasonal patterns.')

    """    
    Through interactive graphs, you can:
     * Zoom into specific timeframes for detailed insights
     * Opt for different viewing intervals (weekly, monthly, yearly)
     * Download the visual as an image
    """
        
    fig3 = plot_plotly(m, forecast)
    st.write(fig3)
    
    st.info('Interactive chart detailing the date influence on the target metric and its forecast.')

    fig4 = plot_components_plotly(m, forecast)
    st.write(fig4)
      
    st.info('Interactive charts elucidating trend and seasonal patterns.')              


"""
### Step 4: Download your healthcare metric forecast

To retain a copy of the forecast, click the "Download forecast" button and save the results to your device.
"""
if df is not None:
    csv_exp = fcst_filtered.to_csv(index=False)
    b64 = base64.b64encode(csv_exp.encode()).decode()  
    st.download_button(label="Download forecast", data=csv_exp, file_name='healthcare_metric_forecast.csv', mime='text/csv')
