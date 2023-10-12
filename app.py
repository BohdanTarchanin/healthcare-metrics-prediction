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
         With the help of this web application, you will be able to get a forecast for your retail business. The program is designed for small and medium-sized businesses that already have a certain set of data about their sales and want to use it for analysis and forecasting.
          To do this, you need to take the following steps:
          1. Import the dataset
          2. Enter the forecast interval
          3. Familiarize yourself with data visualization
          4. Download the forecast

        """         
        
st.image('Sales-forecast.jpg')

st.title('Automated forecasting for retail')

"""
 This program will help you independently make the forecast you need
specifically for your business. It can be a forecast of income, expenses, profit or the amount of goods sold. In order to make a high-quality forecast, it is necessary to collect
a set of sales data in your business. And after that go through 4 steps in this web application to get a forecast based on your data.
"""                     


"""
### Step 1: Import the dataset
"""
df = st.file_uploader('Import your dataset here. A dataset is a table of sales data, in which the row is the date of sale and the column is various characteristics. The date column should be called ds and formatted in the following format: YYYY-MM-DD (Example: 2019-05-20). And in order for the program to understand what exactly you want to predict, mark this column y. The y column must be numeric and represent the measurement we want to predict. The file format is csv. Below will be a link to a design example.', type='csv')
st.info(
            f"""
                👆 First load the file .csv. [File sample](https://raw.githubusercontent.com/facebook/prophet/main/examples/example_air_passengers.csv)
                """
        )

if df is not None:
    data = pd.read_csv(df)
    data['ds'] = pd.to_datetime(data['ds'],errors='coerce') 
    
    st.write(data)
    
    max_date = data['ds'].max()
    st.write(max_date)
    st.success('👆 This is the end date in your data set')

"""
### Step 2: Enter the number of days for the forecast

"""

periods_input = st.number_input('For how many days do you want to make a forecast? Enter a number from 1 to 365 and click Enter',
min_value = 1, max_value = 365)

if df is not None:
    m = Prophet()
    m.fit(data)
   
            
"""
### Step 3: Visualization of forecast data

The following table shows the result of the program - a forecast for the target column y, made on the basis of a historical data set.
* ds - column with forecast dates
* yhat is the predicted value of the target characteristic
* yhat_lower - the probable lower limit of the predicted value
* yhat_upper - the probable upper limit of the predicted value

"""
if df is not None:
    future = m.make_future_dataframe(periods=periods_input)
    
    forecast = m.predict(future)
    fcst = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]

    fcst_filtered =  fcst[fcst['ds'] > max_date]    
    st.write(fcst_filtered)
    
    """
    A bar chart displays the dependence of the date (ds) on the target characteristic (y) and the predicted interval.
     The black dots are the actual figures from the historical data set, and the blue line is the forecast display.
    """
    fig1 = m.plot(forecast)
    st.write(fig1)
    
    st.info('The graph of the dependence of the date on the target value and the forecast')


    """
    The following graphs show the trend for the target characteristic against the date and the seasonal correlation.
    """
    fig2 = m.plot_components(forecast)
    st.write(fig2)
      
    st.info('Graphs of trend and periodic correlation')

    """    
    With the help of interactive graphs, you can:
     * visually increase the required period for a more detailed review
     * choose an interval for viewing (week, month, year)
     * download the graph as an image
    """
        
    fig3 = plot_plotly(m, forecast)
    st.write(fig3)
    
    st.info('Interactive graph of the dependence of the date on the target value and the forecast')


    fig4 = plot_components_plotly(m, forecast)
    st.write(fig4)
      
    st.info('Interactive graphs of trend and periodic correlation')              


"""
### Step 4: Download the forecast for your business

Click the "Download forecast" button and save the result of the program to your computer.
"""
if df is not None:
    csv_exp = fcst_filtered.to_csv(index=False)
    # When no file name is given, pandas returns the CSV as a string, nice.
    b64 = base64.b64encode(csv_exp.encode()).decode()  # some strings <-> bytes conversions necessary here
    #href = f'<a href="data:file/csv;base64,{b64}">Download CSV File</a> (right-click and save as ** &lt;forecast_name&gt;.csv**)'
    #st.markdown(href, unsafe_allow_html=True)
    st.download_button(label="Download forecast", data=csv_exp, file_name='prediction_data.csv', mime='text/csv')


