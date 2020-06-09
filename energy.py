import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
from sklearn.metrics import precision_score, recall_score 
import io
from io import StringIO
import matplotlib
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn.svm import SVR
from collections import Counter
import os.path
from os import path

import seaborn as sns
import statsmodels.api as sm
from patsy import dmatrices
import statsmodels.graphics.api as smg
from pandas.plotting import scatter_matrix
from pandas.plotting._misc import scatter_matrix
import webbrowser

def main():
    st.title("Energy Modelling Machine Learning Web App Using Stream lit ")
    #st.markdown([UCI Dataset Link ](https://archive.ics.uci.edu/ml/datasets/Appliances+energy+prediction))
    UCI_url = 'https://archive.ics.uci.edu/ml/datasets/Appliances+energy+prediction'

    if st.button('Download DataSet From UCI'):
        webbrowser.open_new_tab(UCI_url)
    st.sidebar.title("Machine Learning Web Application")
    st.markdown("⚡ Lets Analyze/Model Electric ⚡ Usage of a house  ⚡")
    st.sidebar.markdown("How do you use your electricity ⚡ at home ? ⚡")

    DATA_URL = ("/home/ec2-user/energydata_complete.csv")
    #@st.cache(persist=True, suppress_st_warning=True)
    def load_data():
        # Check if local path exist 
        if path.exists("D:/University/Coursera/Deployments/Appliance_Energy/energydata_complete.csv"):
            data = pd.read_csv(r'D:/University/Coursera/Deployments/Appliance_Energy/energydata_complete.csv', parse_dates=True )
            return data
        
        # For AWS 
        if path.exists(DATA_URL):
            data = pd.read_csv(DATA_URL, parse_dates=True )
            return data
        else: 
            # If local file does not exist, please upload the file
            uploaded_file = st.file_uploader("Relaod Data", type=["csv", "xls"])
            if uploaded_file is not None:
                data = pd.read_csv(uploaded_file)
                return data
        
            # Future addition: take a path from user. 
            
    # Hard Coded data description, describing names of the columns & units 
    data_description = {
        "Appliances": "Appliances, energy use in Wh",
        "lights": "energy use of light fixtures in the house in Wh",
        "T1": "Kitchen Temperature in Celsius",
        "RH_1": "Humidity in kitchen area, in %",
        "T2": "Temperature in living room area, in Celsius",
        "RH_2": "Humidity in living room area, in %",
        "T3": "Temperature in laundry room area",
        "RH_3": "Humidity in laundry room area, in %",
        "T4": "Temperature in office room, in Celsius",
        "RH_4": "Humidity in office room, in %",
        "T5": "Temperature in bathroom, in Celsius",
        "RH_5": "Humidity in bathroom, in %",
        "T6": "Temperature outside the building (north side), in Celsius",
        "RH_6": "Humidity outside the building (north side), in %",
        "T7" :"Temperature in ironing room , in Celsius",
        "RH_7": "Humidity in ironing room, in %",
        "T8": "Temperature in teenager room 2, in Celsius",
        "RH_8": "Humidity in teenager room 2, in %",
        "T9":" Temperature in parents room, in Celsius",
        "RH_9": "Humidity in parents room, in %",        
        "To":" Temperature outside (from Chièvres weather station), in Celsius",
       "Pressure ":"(from Chièvres weather station), in mm Hg",
       "RH_out":" Humidity outside (from Chièvres weather station), in %",
       "Windspeed ":" Windspeed(from Chièvres weather station), in m/s",
       "Visibility ":"Visibility (from Chièvres weather station), in km",
       "Tdewpoint ":"(from Chièvres weather station), °C",
       "rv1":" Random variable 1, nondimensional",
       "rv2":" Rnadom variable 2, nondimensional"
    }
    
    # Purpose of this function is to show the statistics about data on the website
    # Number of nulls for each column 
    # Data type 
    # https://stackoverflow.com/questions/43976830/pandas-info-to-html
    def analyze_dataframe(content: pd.DataFrame):
        content_info  = io.StringIO()
        content.info(buf=content_info)
        str_ = content_info.getvalue()
        
        lines = str_.split("\n")
        table = StringIO("\n".join(lines[3:-3]))
        datatypes = pd.read_table(table, delim_whitespace=True,names=["#","column", "Non-Null", "Count", "Dtype"])
        datatypes.set_index("#", inplace=True)
        info = "\n".join(lines[0:2] + lines[-2:-1])
        return datatypes
        
    
    # Test/Train set
    #@st.cache(persist=True)
    def split(df):
        # Sklearn treain/test 
        y = df.Appliances
        x = df.drop(columns =["rv1", "rv2", "date", "Appliances"], axis =1)
        x_train,x_test,y_train,y_test= train_test_split(x,y, test_size = 0.3,random_state = 42)
        return x_train,x_test,y_train,y_test
    
    def standardization(df):
        sc=StandardScaler()
        df_scaled = sc.fit_transform(df)
        return df_scaled
    
    def plot_metrics(metrics_list):
        if 'Score' in metrics_list:
            st.write("Accuracy: ", accuracy.round(2))
        if 'MSE' in metrics_list:
            st.write("MSE:  ",mean_squared_error(y_test, y_pred))
            st.write("RMSE:  ",np.sqrt(mean_squared_error(y_test, y_pred)))
            
        # Classification Metrics 
        if 'Confusion Matrix' in metrics_list:
            st.subheader("Confusion Matrix")
            plot_confusion_matrix(model, x_test, y_test_enc, display_labels=class_names)
            #plot_confusion_matrix(model, x_test, y_test_enc)
            st.pyplot()
        if 'ROC Curve' in metrics_list:
            st.subheader("ROC Curve")
            plot_roc_curve(model, x_test, y_test_enc)
            st.pyplot()
        if 'Precision-Recall Curve' in metrics_list:
            st.subheader("Precision-Recall Curve")
            plot_precision_recall_curve(model, x_test,y_test_enc)
            st.pyplot()
            
    def ecdf_plot(a):
        st.write("ECDF Plot")
        x = np.sort(a)
        y = np.arange(len(x))/float(len(x))
        plt.plot(x, y)
        st.pyplot()
    
    
    def encode(y, threshold):
        if threshold== 'mean':
            Y = (y>y.mean())*1
        if threshold== 'median':
            Y = (y>y.median())*1
        return Y
    
    
    # def load_data_via_streamlit(file_upload):
    #     if uploaded_file is not None:
    #         data = pd.read_csv(uploaded_file)
    #         return data
        
            

        
        
    # File upload widget
    #uploaded_file = st.file_uploader("Realod Data", type=["csv", "xls"])
    #df = load_data_via_streamlit(uploaded_file)

    df = load_data()

    if df is not None:
        #st.write('Data Upload Successful')
        x_train,x_test,y_train,y_test = split(df)
        
        # Standardizing train & test data separately 
        x_train = pd.DataFrame(standardization(x_train), columns = x_train.columns, index = x_train.index)
        x_test = pd.DataFrame(standardization(x_test), columns = x_test.columns, index = x_test.index)
        
        # Choose what type of activity you would like to do with 
        activity = st.sidebar.radio("Choose Required Work ", ('EDA','regression', 'Classification', 'OLS'), key='activity')
        
        if activity =="EDA":
            if st.sidebar.checkbox("Show Dataset Head", False):
                st.subheader("Dataset head")
                #st.write(df) 
                st.dataframe(df.head())
            if st.sidebar.checkbox("Show Full data", False):
                st.write(df) 
            if st.sidebar.checkbox("Data Shape/Dimensions", False):
                st.subheader("Data Shape")
                st.subheader( df.shape)
            # column names 
            if st.sidebar.checkbox("Show Column Names", False):
                test = pd.DataFrame(data_description,index=[0])
                st.subheader("Detailed Column Names & Units")
                st.write(test.T)   
            # Dataframe stats
            if st.sidebar.checkbox("Show Descriptive Statistics of the Data", False):
                st.subheader("Energy Data Descriptive Statistics ")
                st.write(df.describe())
                st.subheader("Null & Data Type Statistics  ")
                st.write(analyze_dataframe(df))
            
            if st.sidebar.checkbox("Column Distributions  "):
                st.subheader("Analyzing the distributions")
                column_name = st.selectbox("Select the (Numerical) feature for viewing its histogram ",list(df.columns))     
                st.write(df[column_name].describe())
                print(df.columns.dtype)
                if df[column_name].dtype != object:
                #print(type(df['date']))
                #print(df['date'].dtype)
                #print(type(df['Appliances']))
                #bin_values = np.arange(start=np.min(df[column_name]), stop=np.max(df[column_name]), step=100)
                    plt.hist(df[column_name],bins='auto',  color='steelblue')
                    plt.style.use('seaborn-white')
                    st.pyplot()
                elif df[column_name].dtype != object:
                    st.write('The Column type is Non-numeric ')
                    
            if st.sidebar.checkbox("Energy/Time Plot"):
                st.subheader("Appliance Energy Usage Against time, data obtained in 10 minutes interval ")
                fig = px.line(df, x='date', y='Appliances')
                st.write(fig)
           
            if st.sidebar.checkbox("Standardized Train Data"):
                st.subheader("Standardized train")
                st.write(x_train.head(2))
            
            # Plot against weekdays
            if st.sidebar.checkbox("Appliance Energy: Weekdays vs Weekends", False):
                st.subheader("Weekdays Appliance Energy Usage Vs Weekend Appliance Energy Usage")
                        
                temp_df = df.copy()
                temp_df['WEEKDAY'] = ((pd.to_datetime(temp_df['date']).dt.dayofweek)// 5 == 1).astype(float)
                temp_weekday =  temp_df[temp_df['WEEKDAY'] == 0]
                visData = go.Scatter( x= temp_weekday.date  ,  mode = "lines", y = temp_weekday.Appliances )
                layout = go.Layout(title = 'Appliance energy consumption measurement on weekdays' , xaxis=dict(title='Date'), yaxis=dict(title='(Wh)'))
                weekday_fig = go.Figure(data=[visData],layout=layout)
                st.write(weekday_fig)
                
                temp_weekend =  temp_df[temp_df['WEEKDAY'] == 1]
                visData_weekend = go.Scatter( x= temp_weekend.date  ,  mode = "lines", y = temp_weekend.Appliances )
                layout = go.Layout(title = 'Appliance energy consumption measurement on weekends' , xaxis=dict(title='Date'), yaxis=dict(title='(Wh)'))
                weekend_fig = go.Figure(data=[visData_weekend],layout=layout)
                st.write(weekend_fig)

            # Plotting Correlation For training data
            if st.sidebar.checkbox("Correlation ", False):
                st.subheader("Correlation Matrix")
                st.dataframe(x_train.corr())
                #cor_df.style.background_gradient(cmap='coolwarm')
                #cor_df.style.background_gradient(cmap='coolwarm').set_precision(2)
                #st.pyplot(cor_df)
                

                
        if activity == 'regression':
            st.write('You chose Regression, so be it  ')
            #st.markdown("![Alt Text](https://media.giphy.com/media/vFKqnCdLPNOKc/giphy.gif)")
            st.markdown("![Alt Text](https://media.giphy.com/media/t7sEnf5w7wJ1CEPyy7/giphy.gif)")
            st.sidebar.subheader("Choose Regressor")
            Regressor = st.sidebar.selectbox("Regressor", ("Linear Regression", "RandomForest", "Support Vector Machine Regression (SVM)"))
            
            if Regressor == "Linear Regression":
                metrics = st.sidebar.multiselect("What metrics would you like to see? ",('Score', 'MSE'))                                                                 
                if st.sidebar.button("Regress", key = 'regression'):
                    model = reg = LinearRegression()
                    st.subheader("Linear Regression Results")
                    model.fit(x_train, y_train)
                    accuracy = model.score(x_test,y_test)
                    y_pred = reg.predict(x_test)
                    plot_metrics(metrics)
        
            ###
            if Regressor == "RandomForest":
                st.sidebar.subheader("Model Huperparameters")
                n_estimators = st.sidebar.number_input("n_estimators (The number of trees in the forest.)", 100, 5000, step = 100, key = 'n_estimators')
                criterion = st.sidebar.radio("criterion", ("mse", "mae"), key='criterion')
                max_depth = st.sidebar.number_input("max_depth (Max Depth of the tree.)", 1, 20, step = 1, key = 'max_depth')
                min_samples_leaf = st.sidebar.number_input("min_samples_leaf.)", 1, 10, step = 1, key = 'min_samples_leaf')
            
                # Re-assigning model variable every time for each model. 
                model = RandomForestRegressor(n_estimators=n_estimators,max_depth=max_depth,criterion=criterion, min_samples_leaf=min_samples_leaf, random_state=0)
                metrics = st.sidebar.multiselect("What metrics would you like to see? ",('Score', 'MSE'))
                                                                                    
                if st.sidebar.button("Regress", key = 'regression'):
                    st.subheader("Random Forest Results")
                    model.fit(x_train, y_train)
                    accuracy = model.score(x_test,y_test)
                    y_pred = model.predict(x_test)
                    plot_metrics(metrics)
                    #st.write(model.feature_importances_)
                    important_features = pd.Series(data=model.feature_importances_,index=x_train.columns)
                    important_features.sort_values(ascending=False,inplace=True)
                    st.write(important_features)
                
            ###
            
            if Regressor == "Support Vector Machine Regression (SVM)":
                st.sidebar.subheader("Model Huperparameters")
                kernel = st.sidebar.radio("Kernel", ("rbf", "linear","poly","sigmoid"), key='kernel')
                C = st.sidebar.number_input("C (Regularization parameter)", 1.0, step = 0.5, key = 'C')
                cache_size = st.sidebar.number_input("cache_size (Specify the size of the kernel cache (in MB).)", 200, 700, step =100, key = 'cache_size')
                gamma = st.sidebar.radio("Gamma (Kernel Coefficient)",("scale", "auto"), key = 'svm_gamma')
                                        
        
                model = SVR(kernel=kernel,C=C,gamma=gamma,cache_size=cache_size)
                metrics = st.sidebar.multiselect("What metrics would you like to see? ",('Score', 'MSE'))
                                                                                
                if st.sidebar.button("Regress", key = 'regression'):
                    st.subheader("Support Vector Machine Regression (SVM)")
                    model.fit(x_train, y_train)
                    accuracy = model.score(x_test,y_test)
                    y_pred = model.predict(x_test)
                    plot_metrics(metrics)
                    #st.write(model.coef_)   
        
        
        ### 
        ###
        # The purpose is to use stats model library for increasing explainability of the data
        # Endogenour
        if activity == 'OLS':
            #metrics = st.sidebar.multiselect("What metrics would you like to see? ",('Score', 'MSE'))
            OLS_df = df.drop(['date'],axis =1)
            Dependent_variable = st.sidebar.selectbox("Select the Dependent variable  ",(list(OLS_df.columns)))
            features_name = st.sidebar.multiselect("Select features for regression  ",(list(OLS_df.columns)))
            
            
            #plt.matshow(OLS_df.corr())
            #corr = OLS_df.corr()
            #plt.figure(figsize=(10,10)) 
            #sns.heatmap(corr,xticklabels=corr.columns, yticklabels=corr.columns)

            #st.pyplot()
            if features_name:
                model_equation = Dependent_variable+'~'+'+'.join(features_name)
                st.write('Model Equation: ',model_equation)
                y_stats, X_stats = dmatrices(model_equation, data=OLS_df, return_type='dataframe')
                st.write("Dependent Variable Sample: ",y_stats[:3] )
                st.write("Independent Variable Sample: ",X_stats[:3] )

                mod = sm.OLS(y_stats, X_stats)
                res = mod.fit()
                res_summary = res.summary()
                st.subheader("OLS Summary Statistics Tables")
                st.write(pd.read_html(res_summary.tables[0].as_html(), header=0, index_col=0)[0])
                st.write(pd.read_html(res_summary.tables[1].as_html(), header=0, index_col=0)[0])
                st.write(pd.read_html(res_summary.tables[2].as_html(), header=0, index_col=0)[0])
                
                # Summary 2 
                st.write("Statsmodel provides two methods of model summary sharing. ")
                res_summary_2 = res.summary2()
                st.subheader("Summary 2 ")
                st.write(res_summary_2.tables[0])
                st.write(res_summary_2.tables[1])
                st.write(res_summary_2.tables[2])
                
        # Classification 
        if activity == 'Classification':
            st.subheader('Classification')
            
            st.write('Mean of our target variable is: ',np.mean(y_train))
            st.write('Median of our target variable is: ',np.median(y_train))
            
            threshold = st.radio("Choose class threshold",('mean', 'median'), key = 'threshold')
            ecdf_plot(y_train)
            st.write('To predict high/low electric usage we need to encode the variable, by viewing the ECDF plot we can analyze the distribution of our Target variable. We can than choose median or mean as our cut point for High electric usage or low electric usage')
            # Encoding y 
            y_train_enc = encode(y_train, threshold)
            y_test_enc = encode(y_test, threshold)
            

            class_names = ['Low Usage', 'High Usage']
            
            
            classifier = st.sidebar.selectbox("Classifier", ("Support Vector Machine (SVM)", "Logistic Regression", "Random Forest"))
            
            if classifier == 'Support Vector Machine (SVM)':
                st.sidebar.subheader("Model Huperparameters")
                C = st.sidebar.number_input("C (Regularization parameter)", 1.0, 10.0, step = 0.01, key = 'C')
                kernel = st.sidebar.radio("Kernel", ("rbf", "linear"), key='kernel')
                gamma = st.sidebar.radio("Gamma (Kernel Coefficient)",("scale", "auto"), key = 'gamma')


                metrics = st.sidebar.multiselect("Which metrics to plot ? ",('Confusion Matrix','ROC Curve','Precision-Recall Curve') )


                if st.sidebar.button("Classify", key = 'classify'):
                    st.subheader("Support Vector Machine (SVM) Results")
                # Modelling initialization, train, evaluation using user provided params

                    model = SVC(C=C, kernel=kernel, gamma = gamma, random_state = 42)
                    model.fit(x_train,y_train_enc)
                    
                    accuracy = model.score(x_test,y_test_enc)
                    y_pred = model.predict(x_test)
                
                    
                    st.write("Accuracy: ", accuracy.round(2))
                    st.write("Precision: ", precision_score(y_test_enc,y_pred).round(2))
                    st.write("Recall: ", recall_score(y_test_enc,y_pred).round(2))
                    plot_metrics(metrics)
                    
                    
                    #print(Counter(y_train_enc))
                    #print(Counter(y_test_enc))
                    
                    #print(Counter(y_pred))
            
            if classifier == 'Logistic Regression': 
                st.sidebar.subheader("Model Huperparameters")
                C = st.sidebar.number_input("C (Regularization parameter)", 0.01, 10.0, step = 0.01, key = 'C_LR')
                max_iter = st.sidebar.slider("Maximum number of iterations", 100, 500, key = 'max_iter')


                metrics = st.sidebar.multiselect("What metrics to plot ? ",('Confusion Matrix','ROC Curve','Precision-Recall Curve') )

                if st.sidebar.button("Classify", key = 'classify'):
                    st.subheader("Logistic Regression Results")
                    model = LogisticRegression(C=C, max_iter=max_iter, random_state = 42)
                    # Below code is similar to each estimator
                    
                    model.fit(x_train,y_train_enc)
                    accuracy = model.score(x_test,y_test_enc)
                    y_pred = model.predict(x_test)
                    st.write("Accuracy: ", accuracy.round(2))
                    st.write("Precision: ", precision_score(y_test_enc,y_pred, labels = class_names).round(2))
                    st.write("Recall: ", recall_score(y_test_enc,y_pred, labels = class_names).round(2))
                    plot_metrics(metrics)
            
            if classifier == 'Random Forest':
                st.sidebar.subheader("Model Huperparameters")
                n_estimators = st.sidebar.number_input("The number of trees in the forest", 100, 5000, step = 10, key = 'n_estimators')
                max_depth = st.sidebar.number_input("The maximum depth of the tree", 1,20,step =1, key= 'max_depth')
                bootstrap = st.sidebar.radio("Bootstrap samples when building trees", ("True", "False"), key = 'bootstrap')
                metrics = st.sidebar.multiselect("What metrics to plot ? ",('Confusion Matrix','ROC Curve','Precision-Recall Curve') )

                if st.sidebar.button("Classify", key = 'classify'):
                    st.subheader("Random Forest Results")
                    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, bootstrap=bootstrap, n_jobs =-1)
                    # Below code is similar to each estimator
                    model.fit(x_train,y_train_enc)
                    accuracy = model.score(x_test,y_test_enc)
                    y_pred = model.predict(x_test)
                    st.write("Accuracy: ", accuracy.round(2))
                    st.write("Precision: ", precision_score(y_test_enc,y_pred, labels = class_names).round(2))
                    st.write("Recall: ", recall_score(y_test_enc,y_pred, labels = class_names).round(2))
                    plot_metrics(metrics)
                    
                    

                

                



if __name__ == '__main__':
    main()


