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
        st.write("This code will be printed to the sidebar.")

st.image('Sales-forecast.jpg')

st.title('Автоматизоване прогнозування для роздрібної торгівлі')

"""
За допомогою цього веб-застосунку Ви зможете отримати прогноз для бізнесу роздрібної торгівлі. Програма розрахована на малий та середній бізнес, який уже має
певний набір даних про свої продажі, але не використовує його для аналізу та прогнозування. За допомгою неї Ви зможете самостійно зробити прогноз, який потрібен 
саме для Вашого бізнесу. Це може бути прогноз доходу, розходів, прибутку або кількість проданого товару. Для того, щоб зробити якісний прогноз необхідно зібрати
набір даних про продажі у Вашому бізнесі. І після цього пройти Х кроків у цьому веб-застосунку, щоб отримати прогноз на основі Ваших даних.

"""

"""
### Крок 1. Імпорт даних
"""
df = st.file_uploader('Імпортуйте файл csv часового ряду сюди. Стовпці повинні мати позначки ds і y. Вхідними даними для Prophet завжди є фрейм даних із двома стовпцями: ds і y. Стовпець ds (штамп дати) має мати формат, очікуваний Pandas, в ідеалі РРРР-ММ-ДД для дати або РРРР-ММ-ДД ГГ:ХХ:СС для позначки часу. Стовпець y має бути числовим і представляти вимірювання, яке ми хочемо спрогнозувати.', type='csv')

st.info(
            f"""
                👆 Спочатку завантажте файл .csv. Зразок: [peyton_manning_wiki_ts.csv](https://raw.githubusercontent.com/zachrenwick/streamlit_forecasting_app/master/example_data/example_wp_log_peyton_manning.csv)
                """
        )

if df is not None:
    data = pd.read_csv(df)
    data['ds'] = pd.to_datetime(data['ds'],errors='coerce') 
    
    st.write(data)
    
    max_date = data['ds'].max()
    st.write(max_date)

"""
### Крок 2: Виберіть горизонт прогнозу

Майте на увазі, що прогнози стають менш точними з більшими горизонтами прогнозування.
"""

periods_input = st.number_input('На скільки періодів ви б хотіли спрогнозувати майбутнє?',
min_value = 1, max_value = 365)

if df is not None:
    m = Prophet()
    m.fit(data)
   
            
"""
### Крок 3: візуалізуйте дані прогнозу

Наведене нижче зображення показує майбутні прогнозовані значення. «це» є прогнозованим значенням, а верхня та нижня межі (за замовчуванням) становлять 80% довірчий інтервал.
"""
if df is not None:
    future = m.make_future_dataframe(periods=periods_input)
    
    forecast = m.predict(future)
    fcst = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]

    fcst_filtered =  fcst[fcst['ds'] > max_date]    
    st.write(fcst_filtered)
    
    """
    Наступне зображення показує фактичні (чорні крапки) і прогнозовані (синя лінія) значення з часом.
    """
    fig1 = m.plot(forecast)
    st.write(fig1)


    """
    Наступні кілька візуальних зображень показують високорівневу тенденцію прогнозованих значень, тенденції днів тижня та річні тенденції (якщо набір даних охоплює кілька років). Заштрихована блакитним кольором зона представляє верхній і нижній довірчі інтервали.
    """
    fig2 = m.plot_components(forecast)
    st.write(fig2)

    fig3 = plot_plotly(m, forecast)
    st.write(fig3)

    fig4 = plot_components_plotly(m, forecast)
    st.write(fig4)
            


"""
### Крок 4: Завантажте даний прогноз

Посилання нижче дозволяє завантажити щойно створений прогноз на ваш комп’ютер для подальшого аналізу та використання.
"""
if df is not None:
    csv_exp = fcst_filtered.to_csv(index=False)
    # When no file name is given, pandas returns the CSV as a string, nice.
    b64 = base64.b64encode(csv_exp.encode()).decode()  # some strings <-> bytes conversions necessary here
    #href = f'<a href="data:file/csv;base64,{b64}">Download CSV File</a> (right-click and save as ** &lt;forecast_name&gt;.csv**)'
    #st.markdown(href, unsafe_allow_html=True)
    st.download_button(label="Завантажити прогноз", data=csv_exp, file_name='prediction_data.csv', mime='text/csv')


