##importing a few general use case libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split,cross_val_score,GridSearchCV
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report,confusion_matrix,ConfusionMatrixDisplay


from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier

import warnings
warnings.filterwarnings('ignore')




df = pd.read_csv('saptarshipal1/pred_maint_v.2.1/data/ai4i2020.csv')

def custom_df(df):
    data = df.copy()

    data.rename(columns = {'Air temperature [K]':'Air temperature','Process temperature [K]':'Process temperature','Rotational speed [rpm]':'Rotational speed','Torque [Nm]':'Torque','Tool wear [min]':'Tool wear'},inplace = True)

    data.drop(columns = ['UDI','Product ID','TWF', 'HDF', 'PWF', 'OSF','RNF'], axis = 1,inplace =True)

    data['Type'] = data['Type'].map({'L':0,'M':1,'H':2})

    return data

def data_imputer(data):

    imputer = SimpleImputer()

    cols = ['Type', 'Air temperature', 'Process temperature', 'Rotational speed',
        'Torque', 'Tool wear']

    for col in cols:
        data[cols] = imputer.fit_transform(data[cols])

    return data


def split_data(data):

    X = data.drop(columns = ['Machine failure'], axis =1)
    y = data['Machine failure']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size =0.2,random_state= 67 )

    X_train = pd.DataFrame(X_train)
    X_test = pd.DataFrame(X_test)   
    
    y_train = pd.DataFrame(y_train)
    y_test = pd.DataFrame(y_test)


    return X_train, X_test, y_train, y_test


#Created variables to keep reference points for future data point scaling


# X_train_Air_temperature_max = X_train.iloc[:,1].max()
# X_train_Process_temperature_max = X_train.iloc[:,2].max()
# X_train_Rotational_speed_max = X_train.iloc[:,3].max()
# X_train_Torque_max = X_train.iloc[:,4].max()

# X_train_Air_temperature_min = X_train.iloc[:,1].min()
# X_train_Process_temperature_min = X_train.iloc[:,2].min()
# X_train_Rotational_speed_min = X_train.iloc[:,3].min()
# X_train_Torque_min = X_train.iloc[:,4].min()

# X_train_trans = X_train.copy()
# X_test_trans = X_test.copy()

def scale_transf(X_train_trans,X_test_trans):

    trans = MinMaxScaler()

    X_train_trans[['Air temperature', 'Process temperature', 'Rotational speed','Torque', 'Tool wear']] = trans.fit_transform(X_train_trans[['Air temperature', 'Process temperature', 'Rotational speed','Torque', 'Tool wear']])
    X_test_trans[['Air temperature', 'Process temperature', 'Rotational speed','Torque', 'Tool wear']] = trans.fit_transform(X_test_trans[['Air temperature', 'Process temperature', 'Rotational speed','Torque', 'Tool wear']])

    return X_train_trans,X_test_trans

final_model=MLPClassifier(activation= 'relu',
 alpha= 0.0001,
 hidden_layer_sizes= (50, 100, 50),
 learning_rate= 'constant',
 solver= 'adam')



        










