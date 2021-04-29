import streamlit as st

import base64
from io import BytesIO

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns # For creating plots
import matplotlib.ticker as mtick # For specifying the axes tick format 
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import RidgeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from itertools import cycle, islice
import xlsxwriter



def download_link(object_to_download, download_filename, download_link_text):
    """
    Generates a link to download the given object_to_download.

    object_to_download (str, pd.DataFrame):  The object to be downloaded.
    download_filename (str): filename and extension of file. e.g. mydata.csv, some_txt_output.txt
    download_link_text (str): Text to display for download link.

    Examples:
    download_link(YOUR_DF, 'YOUR_DF.csv', 'Click here to download data!')
    download_link(YOUR_STRING, 'YOUR_STRING.txt', 'Click here to download your text!')

    """
    if isinstance(object_to_download,pd.DataFrame):
        object_to_download = object_to_download.to_csv(index=False)

    # some strings <-> bytes conversions necessary here
    b64 = base64.b64encode(object_to_download.encode()).decode()

    return f'<a href="data:file/txt;base64,{b64}" download="{download_filename}">{download_link_text}</a>'

def main():
    
    def MLmodels(df):

        columns = ["Churn"]
        try:
            for i in columns:
                df[i].replace(to_replace='Yes', value=1, inplace=True)
                df[i].replace(to_replace='No',  value=0, inplace=True)
        except:
            None
        
        df_train = df[df['Churn']>-1]
        df_train = df_train.iloc[:,1:] #removing customer ID
        column_means_train = df_train.mean()
        df_train = df_train.fillna(column_means_train)
        #df_train.dropna(inplace = True)
        df_train = pd.get_dummies(df_train)
        
        df_predict = df[df['Churn'].isna()]
        
        y = df_train['Churn'].values
        X = df_train.drop(columns = ['Churn'])
        features = X.columns.values
        scaler = MinMaxScaler(feature_range = (0,1))
        scaler.fit(X)
        X = pd.DataFrame(scaler.transform(X))
        X.columns = features
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=101)
        
        def gridsearchfunction(grid,model):
            cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
            grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy',error_score=0)
            grid_result = grid_search.fit(X_train, y_train)
        
            print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
            
            means = grid_result.cv_results_['mean_test_score']
            stds = grid_result.cv_results_['std_test_score']
            params = grid_result.cv_results_['params']
            for mean, stdev, param in zip(means, stds, params):
                print("%f (%f) with: %r" % (mean, stdev, param))
            return grid_result
        
        def logisticmodel(X_train,y_train,X_test,y_test):
    
            model = LogisticRegression()
            
            #solvers = ['newton-cg', 'lbfgs', 'liblinear']
            #penalty = ['l2']
            #c_values = [100, 10, 1.0, 0.1, 0.01]
            
            solvers = ['newton-cg']
            penalty = ['l2']
            c_values = [10]
            grid = dict(solver=solvers,penalty=penalty,C=c_values)
            grid_result = gridsearchfunction(grid,model)
        
            model = LogisticRegression(**grid_result.best_params_)
            model.fit(X_train,y_train)
            y_predicted = model.predict(X_test)
            y_proba = model.predict_proba(X_test)
            logaccuracy = round((metrics.accuracy_score(y_test, y_predicted)),3)
            print (f"Test accuracy is {logaccuracy}")
            return  y_predicted,logaccuracy,grid_result.best_params_
        
                
        def ridgemodel(X_train,y_train,X_test,y_test):
            
            model = RidgeClassifier()
            #alpha = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
            alpha = [0.1]
            grid = dict(alpha=alpha)
            cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
            grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy',error_score=0)
            grid_result = gridsearchfunction(grid,model)
        
            model = RidgeClassifier(**grid_result.best_params_)
            model.fit(X_train,y_train)
            y_predicted = model.predict(X_test)
            ridgeaccuracy = round((metrics.accuracy_score(y_test, y_predicted)),3)
            print (f"Test accuracy is {ridgeaccuracy}")
            return  y_predicted,ridgeaccuracy,grid_result.best_params_
        
        
        def KNNmodel(X_train,y_train,X_test,y_test):
            
            model = KNeighborsClassifier()
            #n_neighbors = range(1, 21, 2)
            #weights = ['uniform', 'distance']
            #metric = ['euclidean', 'manhattan', 'minkowski']
        
            n_neighbors = [10]
            weights = ['uniform']
            metric = ['euclidean']
            
            grid = dict(n_neighbors=n_neighbors,weights=weights,metric=metric)
            grid_result = gridsearchfunction(grid,model)
        
            model = KNeighborsClassifier(**grid_result.best_params_)
            model.fit(X_train,y_train)
            y_predicted = model.predict(X_test)
            KNNaccuracy = round((metrics.accuracy_score(y_test, y_predicted)),3)
            print (f"Test accuracy is {KNNaccuracy}")
            return  y_predicted,KNNaccuracy,grid_result.best_params_
        
        def SVMmodel(X_train,y_train,X_test,y_test):              
            model = SVC()
            model.fit(X_train,y_train)
            y_predicted = model.predict(X_test)
            SVMaccuracy = round((metrics.accuracy_score(y_test, y_predicted)),3)
            print (f"Test accuracy is {SVMaccuracy}")
            return  y_predicted,SVMaccuracy#,grid_result.best_params_
    
        def bagging(X_train,y_train,X_test,y_test):
            
            # define models and parameters
            model = BaggingClassifier()
            
            #n_estimators = [10, 100, 1000]
            
            n_estimators = [10]  
            
            grid = dict(n_estimators=n_estimators)
            cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
            grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy',error_score=0)
            grid_result = gridsearchfunction(grid,model)
        
            model = BaggingClassifier(**grid_result.best_params_)
            model.fit(X_train,y_train)
            y_predicted = model.predict(X_test)
            bagaccuracy = round((metrics.accuracy_score(y_test, y_predicted)),3)
            print (f"Test accuracy is {bagaccuracy}")
            return  y_predicted,bagaccuracy,grid_result.best_params_
        
        
        def RF(X_train,y_train,X_test,y_test):
            
            # define models and parameters
            model = RandomForestClassifier()
            #n_estimators = [10, 100, 1000]
            #max_features = ['sqrt', 'log2']
            
            n_estimators = [10]
            max_features = ['sqrt']
        
            grid = dict(n_estimators=n_estimators,max_features=max_features)
            cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
            grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy',error_score=0)
            grid_result = gridsearchfunction(grid,model)
        
            model = RandomForestClassifier(**grid_result.best_params_)
            model.fit(X_train,y_train)
            y_predicted = model.predict(X_test)
            RFaccuracy = round((metrics.accuracy_score(y_test, y_predicted)),3)
            print (f"Test accuracy is {RFaccuracy}")
            return  y_predicted,RFaccuracy,grid_result.best_params_
        
        def SGD(X_train,y_train,X_test,y_test):
            
            model = GradientBoostingClassifier()
            
            #n_estimators = [10, 100]
            #learning_rate = [0.001, 0.01, 0.1]
            #subsample = [0.5, 0.7, 1.0]
            #max_depth = [3, 7, 9]
            
            n_estimators = [10]
            learning_rate = [0.1]
            subsample = [0.5]
            max_depth = [3]
        
            grid = dict(learning_rate=learning_rate, n_estimators=n_estimators, subsample=subsample, max_depth=max_depth)
            cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
            grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy',error_score=0)
            grid_result = gridsearchfunction(grid,model)
        
            model = GradientBoostingClassifier(**grid_result.best_params_)
            model.fit(X_train,y_train)
            y_predicted = model.predict(X_test)
            SGDaccuracy = round((metrics.accuracy_score(y_test, y_predicted)),3)
            print (f"Test accuracy is {SGDaccuracy}")
            return  y_predicted,SGDaccuracy, grid_result.best_params_
        
        ylogpredicted,logaccuracy,logbestparam = logisticmodel(X_train,y_train,X_test,y_test)
        yridgepredicted,ridgeaccuracy,ridgebestparam = ridgemodel(X_train,y_train,X_test,y_test)
        yKNNpredicted,KNNaccuracy,KNNbestparam = KNNmodel(X_train,y_train,X_test,y_test)
        ySVMpredicted,SVMaccuracy = SVMmodel(X_train,y_train,X_test,y_test)
        ybagpredicted,bagaccuracy,bagbestparam = bagging(X_train,y_train,X_test,y_test)
        yRFpredicted,RFaccuracy,RFbestparam = RF(X_train,y_train,X_test,y_test)
        ySGDpredicted,SGDaccuracy,SGDbestparam = SGD(X_train,y_train,X_test,y_test)
        
        data = [['Logistic', round(logaccuracy,3)], 
                ['Ridge', round(ridgeaccuracy,3)], 
                ['KNN', round(KNNaccuracy,3)], 
                ['SVM', round(SVMaccuracy,3)], 
                ['Bagging',round(bagaccuracy,3)], 
                ['RF',round(RFaccuracy,3)],
                ['SGD',round(SGDaccuracy,3)]] 

        df = pd.DataFrame(data, columns = ['Model', 'Accuracy'])
        df.sort_values(by='Accuracy', ascending=False)
        df.reset_index(drop=True)
        
        
        Xestimate = df_predict.iloc[:,1:] #removing customer ID
        Xestimate = Xestimate.drop(columns = ['Churn'])
        
        column_means_estimate = Xestimate.mean()
        Xestimate = Xestimate.fillna(column_means_estimate)
        
        
        #Xestimate = Xestimate.dropna()
        
        df_predict["Churn"] = 2
        column_means_predict = df_predict.mean()
        df_predict = df_predict.fillna(column_means_predict)
        
        #df_predict = df_predict.dropna()
        
        
        Xestimate = pd.get_dummies(Xestimate)
        
        features = Xestimate.columns.values
        scaler = MinMaxScaler(feature_range = (0,1))
        scaler.fit(Xestimate)
        Xestimate = pd.DataFrame(scaler.transform(Xestimate))
        Xestimate.columns = features
        
        if df["Model"][0] == "Logistic":
            print("Selected model is Logistic classifier")
            model = LogisticRegression(**logbestparam)
            model.fit(X_train,y_train)
            y_predicted = model.predict_proba(Xestimate)
        
        elif df["Model"][0] == "Ridge":
            print("Selected model is Ridge classifier")
            model = RidgeClassifier(**ridgebestparam)
            model.fit(X_train,y_train)
            y_predicted = model.predict(Xestimate)
        
        elif df["Model"][0] == "KNN":
            print("Selected model is KNN classifier") 
            model = KNeighborsClassifier(**KNNbestparam)
            model.fit(X_train,y_train)
            y_predicted = model.predict(Xestimate)
        
        elif df["Model"][0] == "SVM":
            print("Selected model is SVM classifier")
            model = SVC()
            model.fit(X_train,y_train)
            y_predicted = model.predict(Xestimate)
        
        elif df["Model"][0] == "Bagging":
            print("Selected model is Bagging classifier")
            model = BaggingClassifier(**bagbestparam)
            model.fit(X_train,y_train)
            y_predicted = model.predict(Xestimate)
            
        elif df["Model"][0] == "RF":
            print("Selected model is RF classifier")
            model = RandomForestClassifier(**RFbestparam)
            model.fit(X_train,y_train)
            y_predicted = model.predict(Xestimate)
        
        elif df["Model"][0] == "SGD":
            print("Selected model is SGD classifier")
            model = GradientBoostingClassifier(**SGDbestparam)
            model.fit(X_train,y_train)
            y_predicted = model.predict(Xestimate)
            
        df_predict["Churn"] = y_predicted
        return df_predict,df
    
    @st.cache(allow_output_mutation=True)
    def load_data(file):
        df = pd.read_csv(file)
        return df
        
    st.sidebar.title("Upload data")    
    uploaded_file = st.sidebar.file_uploader("Choose a file")
    if uploaded_file is not None:
        df = load_data(uploaded_file)
   
    st.header("Homepage.")
    st.write("This web-app is used to analyze Churn probabilities of customers. Please upload your data using the file uploader on the left side.")

    if uploaded_file is None:
        st.write("Please upload the data first.")
        
    if uploaded_file is not None:
        st.subheader("Uploaded data is shown below")
        st.write(df)  
        st.subheader("Feature Engineering")

        columns = ["Churn"]
        try:
            for i in columns:
                df[i].replace(to_replace='Yes', value=1, inplace=True)
                df[i].replace(to_replace='No',  value=0, inplace=True)
        except:
            None
        
        df_train = df[df['Churn']>-1]
        df_train = df_train.iloc[:,1:] #removing customer ID
        column_means_train = df_train.mean()
        df_train = df_train.fillna(column_means_train)
        #df_train.dropna(inplace = True)
        df_train = pd.get_dummies(df_train)

        fig = plt.figure(figsize=(10,5))
        my_colors = list(islice(cycle(['b', 'r', 'g', 'y', 'k']), None, len(df)))
        df_train.corr()['Churn'].sort_values(ascending = False).plot(kind='bar',stacked=True,color=my_colors)
        st.write(fig)

        st.subheader("Model Training")
        df_predict,df_errors = MLmodels(df)
        st.write("7 different classification models were trained and tested for the provided dataset.")
        st.write("Summary of the models and their accuracies are provided below. The best performing model was selected to provide the prediction results.")

        st.write(df_errors)
        
        df_predict = df_predict[['customerID','Churn']]
        df_train = []
        df_errors = []
        df = []
        #st.write("Final dataset with the predictions is shown below.")
        #st.write(df_predict)
        
        #st.write("Calculations completed.")
        tmp_download_link = download_link(df_predict, 'predictionresults.csv', 'Click here to download your data!')
        #st.write("Generating the download link.")
        st.markdown(tmp_download_link, unsafe_allow_html=True)
        #st.write("Thank you for using ChurnModel.")

if __name__ == "__main__":
    main()