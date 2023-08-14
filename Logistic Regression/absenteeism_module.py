# Imports
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin

# Custom scaler
class CustomScaler (BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.scaler = StandardScaler()
        self.columns = columns
        self.mean_ = None
        self.var_ = None

    def fit(self, X, y = None):
        self.scaler.fit(X[self.columns], y)
        self.mean_ = np.array(np.mean(X[self.columns]))
        self.var_ = np.array(np.var(X[self.columns]))
        return self
    
    def transform(self, X, y = None, copy = None):
        init_col_order = X.columns
        X_scaled = pd.DataFrame(self.scaler.transform(X[self.columns]), columns = self.columns)
        X_not_scaled = X.loc[:, ~X.columns.isin(self.columns)]
        return pd.concat([X_not_scaled, X_scaled], axis = 1)[init_col_order]
    
# Class for making predictions
class absenteeism_model():
    def __init__(self, model_file, scaler_file):
        # read the 'model' and 'scaler' files that were saved earlier
        with open('model', 'rb') as model_file, open('scaler', 'rb') as scaler_file:
            self.reg = pickle.load(model_file)
            self.scaler = pickle.load(scaler_file)
            self.data = None

    def load_and_clean_data(self, data_file):
        # import data
        df = pd.read_csv(data_file, delimiter = ',')
        # store data in new variable for later
        self.df_with_predictions = df.copy()
        # drop ID column
        df = df.drop(['ID'], axis = 1)
        df['Absenteeism Time in Hours'] = 'NaN'

        # Create new dataframe for dummies
        reason_columns = pd.get_dummies(df['Reason for Absence'], drop_first= True, dtype=int)

        reason_type_1 = reason_columns.loc[:, 1:14].max(axis = 1)
        reason_type_2 = reason_columns.loc[:, 15:17].max(axis = 1)
        reason_type_3 = reason_columns.loc[:, 18:21].max(axis = 1)
        reason_type_4 = reason_columns.loc[:, 22:].max(axis = 1)

        # Drop reasons and add dummies
        df = df.drop(['Reason for Absence'], axis = 1)
        df = pd.concat([df, reason_type_1, reason_type_2, reason_type_3, reason_type_4], axis = 1)

        # Assign column names
        column_names = ['Date', 'Transportation Expense', 'Distance to Work', 'Age',
                        'Daily Work Load Average', 'Body Mass Index', 'Education', 'Children',
                        'Pets', 'Absenteeism Time in Hours', 'Reason_1', 'Reason_2', 'Reason_3', 'Reason_4']
        df.columns = column_names

        # Reorder columns
        column_names_reordered = ['Reason_1', 'Reason_2', 'Reason_3', 'Reason_4', 'Date', 'Transportation Expense',
                                  'Distance to Work', 'Age', 'Daily Work Load Average', 'Body Mass Index', 'Education',
                                  'Children', 'Pets', 'Absenteeism Time in Hours']
        df = df[column_names_reordered]

        # Convert date to datetime
        df['Date'] = pd.to_datetime(df['Date'], format = '%d/%m/%Y')

        # List of month values
        list_months = []
        for i in range(df.shape[0]):
            list_months.append(df['Date'][i].month)

        # Insert month values into dataframe
        df['Month Value'] = list_months

        # Create day of the week feature
        df['Day of the Week'] = df['Date'].apply(lambda x: x.weekday())

        # Drop Date column
        df = df.drop(['Date'], axis = 1)

        # Reorder columns
        column_names_upd = ['Reason_1', 'Reason_2', 'Reason_3', 'Reason_4', 'Month Value', 'Day of the Week',
                            'Transportation Expense', 'Distance to Work', 'Age', 'Daily Work Load Average',
                             'Body Mass Index', 'Education', 'Children', 'Pets', 'Absenteeism Time in Hours']
        df = df[column_names_upd]

        # map 'Eduction' variable; resultas are dummies
        df['Education'] = df['Education'].map({1:0, 2:1, 3:1, 4:1})
        
        # Replace NaN values with 0
        df = df.fillna(value = 0)

        # Drop Absenteeism time
        df = df.drop(['Absenteeism Time in Hours'], axis = 1)

        # Drop variable that were not needed, because they didn't have a significant impact on the regression
        df = df.drop(['Day of the Week', 'Daily Work Load Average', 'Distance to Work'], axis = 1)

        # Create a variable so we can call the preprocessed data
        self.preprocessed_data = df.copy()

        # Need this variable for the next functions
        self.data = self.scaler.transform(df)

    # Outputs the probability of a data point being 1
    def predicted_probability(self):
        if (self.data is not None):
            pred = self.reg.predict_proba(self.data)[:, 1]
            return pred

    # Output 0 or 1 based on the model
    def predict_output_category(self):
        if (self.data is not None):
            pred_outputs = self.reg.predict(self.data)
            return pred_outputs
        
    # Predict outputs and probabilities, add these columns to the end of the dataframe
    def predicted_outputs(self):
        if (self.data is not None):
            self.preprocessed_data['Probability'] = self.reg.predict_proba(self.data)[:, 1]
            self.preprocessed_data['Prediction'] = self.reg.predict(self.data)
            return self.preprocessed_data