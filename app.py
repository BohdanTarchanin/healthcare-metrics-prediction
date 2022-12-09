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
         За допомогою цього веб-застосунку Ви зможете отримати прогноз для бізнесу роздрібної торгівлі. Програма розрахована на малий та середній бізнес, який уже має певний набір даних про свої продажі і хоче використати його для аналізу та прогнозування. 
         Для цього потрібно зробити наступні кроки: 
         1. Імпортувати набір даних 
         2. Ввести інтервал прогнозу 
         3. Ознайомитися із візуалізацією даних 
         4. Завантажити прогноз

        """         
        
st.image('Sales-forecast.jpg')

st.title('Автоматизоване прогнозування для роздрібної торгівлі')

"""
 Ця програма допоможе Вам самостійно зробити прогноз, який потрібен. 
саме для Вашого бізнесу. Це може бути прогноз доходу, розходів, прибутку або кількість проданого товару. Для того, щоб зробити якісний прогноз необхідно зібрати
набір даних про продажі у Вашому бізнесі. І після цього пройти 4 кроки у цьому веб-застосунку, щоб отримати прогноз на основі Ваших даних.

"""                     


"""
### Крок 1. Імпортуйте набір даних
"""
df = st.file_uploader('Імпортуйте свій набір даних сюди. Набір даних - це таблиця з даними про продажі, в якій рядок - це дата продажу, а стовпець - різні характеристики. Стовпець з датою необхідно назвати ds і оформити у такому форматі: РРРР-ММ-ДД (Приклад: 2019-05-20). І для того, щоб програма зрозуміла, що саме Ви хочете спрогнозувати, то позначте цей стовпець y. Стовпець y має бути числовим і представляти вимірювання, яке ми хочемо спрогнозувати. Формат файлу - csv. Внизу буде посилання на приклад оформлення.', type='csv')

st.info(
            f"""
                👆 Спочатку завантажте файл .csv. [Зразок оформлення](https://raw.githubusercontent.com/BohdanTarchanin/streamlit-sales-pred-7/master/example_retail_sales.csv?token=GHSAT0AAAAAAB3ZH36Q3IR62ASJEVDUTQOAY4J4M3A)
                """
        )

if df is not None:
    data = pd.read_csv(df)
    data['ds'] = pd.to_datetime(data['ds'],errors='coerce') 
    
    st.write(data)
    
    max_date = data['ds'].max()
    st.write(max_date)
    st.success('👆 Це кінцева дата у Вашому наборі даних')

"""
### Крок 2: Введіть кількість днів для прогнозу

"""

periods_input = st.number_input('На скільки днів ви хочете зробити прогноз? Введіть число від 1 до 365 і натисніть Enter',
min_value = 1, max_value = 365)

if df is not None:
    m = Prophet()
    m.fit(data)
   
            
"""
### Крок 3: Візуалізація даних прогнозу

Наведена нижче таблиця відображає результат роботи програми - прогноз для цільового стовпця у, зроблений на основі історичного набору даних.
* ds - стовпець з датами прогнозування
* yhat - прогнозоване значення цільової характеристики
* yhat_lower - імовірна нижня межа  прогнозованого значення 
* yhat_upper - імовірна верхня межа прогнозованого значення

"""
if df is not None:
    future = m.make_future_dataframe(periods=periods_input)
    
    forecast = m.predict(future)
    fcst = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]

    fcst_filtered =  fcst[fcst['ds'] > max_date]    
    st.write(fcst_filtered)
    
    """
    Насупний графік відорбражає залежність дати (ds) від цільової характеристики (y) і прогнозований інтервал.
    Чорні крапки - фактичні показники з історичного набору даних, а синя лінія - відображення прогнозу.
    """
    fig1 = m.plot(forecast)
    st.write(fig1)
    
    st.info('Графік залежності дати від цільового значення і прогнозу')


    """
    Наступні графіки відображають тренд для цільової характеристики відносно дати і сезонну кореляцію.
    """
    fig2 = m.plot_components(forecast)
    st.write(fig2)
      
    st.info('Графіки тренду і періодичної кореляції')

    """    
    За допомогою інтерактивних графіків Ви можете:
    * візульно збільшити необхідний період для детальнішого перегляду
    * обрати інтервал для перегляду (тиждень, місяць, рік)
    * завантажити графік у вигляді зображення
    """
        
    fig3 = plot_plotly(m, forecast)
    st.write(fig3)
    
    st.info('Інтерактивний графік залежності дати від цільового значення і прогнозу')


    fig4 = plot_components_plotly(m, forecast)
    st.write(fig4)
      
    st.info('Інтерактивні графіки тренду і періодичної кореляції')              


"""
### Крок 4: Завантажуйте прогноз для Вашого бізнесу

Натискайте кнопку "Завантажити прогноз" і зберігайте результат виконання програми собі на комп'ютер.
"""
if df is not None:
    csv_exp = fcst_filtered.to_csv(index=False)
    # When no file name is given, pandas returns the CSV as a string, nice.
    b64 = base64.b64encode(csv_exp.encode()).decode()  # some strings <-> bytes conversions necessary here
    #href = f'<a href="data:file/csv;base64,{b64}">Download CSV File</a> (right-click and save as ** &lt;forecast_name&gt;.csv**)'
    #st.markdown(href, unsafe_allow_html=True)
    st.download_button(label="Завантажити прогноз", data=csv_exp, file_name='prediction_data.csv', mime='text/csv')


