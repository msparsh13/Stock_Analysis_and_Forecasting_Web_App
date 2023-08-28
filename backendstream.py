import streamlit as st
import plotly.graph_objects as go
from io import StringIO
import pandas as pd
import string
import numpy as np
import datetime
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import math
from sklearn.metrics import mean_squared_error


st.title("Web App For Stock Analysis")
uploaded_file = st.file_uploader("****Choose a  '.csv'  file****")
if uploaded_file is not None:
    x= pd.read_csv(uploaded_file)
   
   
    for column in x.columns:
       x.rename(columns={column: column.title() } , inplace=True)
 
    fig = go.Figure(data=[go.Candlestick(x=x['Date'],
                       open=x['Open'], close=x['Close'],
                       low=x['Low'], high=x['High'],
               increasing_line_color= 'green', decreasing_line_color= 'red')])

    st.plotly_chart(fig)
    
    choice = st.selectbox(
    'Which **Line Graph** would you like to see',
    (None , 'Open' ,'Close' , 'High'  , 'Low' ))
    number = st.number_input(label="Enter Rolling for Moving Average" ,   min_value=0)
    if choice is not None:
       
       if(number >0):
          z = x.rolling(number).mean()
          
          fig = go.Figure(data=[go.Line(x=x['Date'] , y=x[choice]) , go.Line(x=x['Date'] , y=z[choice])] )
       else:
          fig = go.Figure(data=[go.Line(x=x['Date'] , y=x[choice])])
       fig.update_layout(showlegend=False)
           
       st.plotly_chart(fig)

      #deep learning model

    model = load_model(r'C:\Users\Sparsh Mahajan\OneDrive\Documents\c progams\.vscode\.vscode\backend\stock anlysis and prediction\ta.h5')



    abc = st.selectbox(
    'Which Value to predict',
    ( 'Open' ,'Close' , 'High'  , 'Low' , None ))
    
    if abc is not None:                  
        scaler = MinMaxScaler()
        scaled_data=scaler.fit_transform(np.array(x[abc]).reshape(-1,1))
        n_past = 60
        train_size = int(len(scaled_data)*0.80)  
        train_data = scaled_data[:train_size]
        # Prepare sequences for LSTM
        X_train, y_train = [], []
        for i in range(n_past, len(train_data)):
            X_train.append(train_data[i-n_past: i , 0])
            y_train.append(train_data[i:i , 0])
        X_train, y_train = np.array(X_train), np.array(y_train)
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
       
        st.header("These results are based on LSTM model")
        Y_ = model.predict(X_train[-60:]).reshape(-1, 1)
        Y_ = scaler.inverse_transform(Y_)
        x['Date'] = pd.to_datetime(x['Date'])
        df_future = pd.DataFrame(columns=['Date', 'Actual', 'Forecast'])
        df_future['Date'] = pd.date_range(start=x['Date'].iloc[-1] + pd.Timedelta(days=1), periods=n_past)
        df_future['Forecast'] = Y_.flatten()
        df_future['Actual'] = np.nan
        results = x.append(df_future).set_index('Date')
        st.write(df_future[['Date' , 'Forecast']] )

        
        fig3 = go.Figure(data=[go.Line(x=results.index , y=x[abc] ) , go.Line(x=results.index , y=results['Forecast'])  ] )
        fig3['data'][0]['name'] = 'Actual'
        fig3['data'][1]['name'] = 'Forecast'
        st.subheader("Forecasting Graph") 
        st.plotly_chart(fig3)
      
    
 

       



   
    
