##importing a few general use case libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import *
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from flask import Flask,request,render_template
from model_files.model import custom_df,data_imputer,scale_transf,final_model

df = pd.read_csv('/Users/saptarshipal/Documents/data/pred_maint_v.2.1/pred_maint_v.2.1/data/ai4i2020.csv')

app = Flask(__name__,template_folder='templates')



## Route for a home page

@app.route('/',methods=['GET','POST'])  
def predict_datapoint():
            if request.method=='GET':
                return render_template('home.html')
            else:

                Type=float(request.form["Product Type"])
                Air_temperature=float(request.form.get("Air temperature"))
                Process_temperature=float(request.form.get("Process temperature"))
                Rotational_speed = float(request.form.get("Rotational speed"))
                Torque = float(request.form.get("Torque"))
                Tool_wear = float(request.form.get("Tool wear"))


                input_data = [[Type
                              ,Air_temperature
                              ,Process_temperature
                              ,Rotational_speed
                              ,Torque
                              ,Tool_wear
                              ]]
                            
                df_input_data = pd.DataFrame(input_data,columns=['Type'
                              ,'Air_temperature'
                              ,'Process_temperature'
                              ,'Rotational_speed'
                              ,'Torque'
                              ,'Tool_wear'])
                
                
                data = custom_df(df)
                
                data = data_imputer(data)

                X = data.drop(columns = ['Machine failure'], axis =1)
                y = data['Machine failure']

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size =0.2,random_state= 67 )

                X_train = pd.DataFrame(X_train)
                X_test = pd.DataFrame(X_test)   
                
                y_train = pd.DataFrame(y_train)
                y_test = pd.DataFrame(y_test)

                X_train_Air_temperature_max = X_train.iloc[:,1].max()
                X_train_Process_temperature_max = X_train.iloc[:,2].max()
                X_train_Rotational_speed_max = X_train.iloc[:,3].max()
                X_train_Torque_max = X_train.iloc[:,4].max()

                X_train_Air_temperature_min = X_train.iloc[:,1].min()
                X_train_Process_temperature_min = X_train.iloc[:,2].min()
                X_train_Rotational_speed_min = X_train.iloc[:,3].min()
                X_train_Torque_min = X_train.iloc[:,4].min()

                X_train_trans = X_train.copy()
                X_test_trans = X_test.copy()

                scale_transf(X_train_trans,X_test_trans)

                X_train_trans = pd.DataFrame(X_train_trans)
                X_test_trans = pd.DataFrame(X_test_trans)                 


                final_model.fit(X_train_trans, y_train)

                df_input_data['Air_temperature'] = (df_input_data['Air_temperature']-X_train_Air_temperature_min)/(X_train_Air_temperature_max - X_train_Air_temperature_min)
                df_input_data['Process_temperature'] = (df_input_data['Process_temperature']-X_train_Process_temperature_min)/(X_train_Process_temperature_max - X_train_Air_temperature_min)
                df_input_data['Rotational_speed'] = (df_input_data['Rotational_speed']-X_train_Rotational_speed_min)/(X_train_Rotational_speed_max - X_train_Rotational_speed_min)
                df_input_data['Torque'] = (df_input_data['Torque']-X_train_Torque_min)/(X_train_Torque_max - X_train_Torque_min)
                
                proc_input_data = df_input_data.values.tolist()


                results = final_model.predict(proc_input_data)


            return render_template('home.html',results=results[0])

if __name__ == '__main__':
    app.run(debug=True,host="0.0.0.0",port=9696)


