import pandas as pd
import numpy as np

# To Preproccesing our data
from sklearn.preprocessing import LabelEncoder

# To fill missing values
from sklearn.impute import SimpleImputer

# To Split our train data
from sklearn.model_selection import train_test_split

# To Visualize Data
import matplotlib.pyplot as plt
import seaborn as sns

# To Train our data
from xgboost import XGBClassifier
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.model_selection import GridSearchCV

# To evaluate end result we have
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import cross_val_score

class Modeler:

    """
    - this class is responsible for modeling
    """

    def __init__(self,df):
        """
        - Initialization of the class
        """
        self.df = df

    def generate_pipeline(self,type_="numeric",x=1):
        """
        purpose:
            - generate_pipelines for the data
        input:
            - string and int
        returns:
            - pipeline
        """
        pipeline = None
        if type_ == "numeric":
            pipeline = Pipeline(steps=[
                ('impute', SimpleImputer(strategy='mean')),
                ('scale', MinMaxScaler())
            ])
        elif type_ == "categorical":
            pipeline = Pipeline(steps=[
            ('impute', SimpleImputer(strategy='most_frequent')),
            ('one-hot', OneHotEncoder(handle_unknown='ignore', sparse=False))
            ])
        else:
            pipeline = np.zeros(x)
        return pipeline

    def generate_transformation(self,pipeline,type_,value,trim=None,key=None):
        """
        purpose:
            - generates transformations for the data
        input:
            - string,int and df
        returns:
            - transformation
        """
        transformation = None
        if type_=="numeric":
            transformation=pipeline.fit_transform(self.df.select_dtypes(include=value))
            if trim:
                transformation=pipeline.fit_transform(pd.DataFrame(self.split_data(key,0.3,trim)).select_dtypes(include=value))
        elif type_ == "categorical":
            transformation=pipeline.fit_transform(self.df.select_dtypes(exclude=value))
            if trim:
                transformation=pipeline.fit_transform(pd.DataFrame(self.split_data(key,0.3,trim)).select_dtypes(exclude=value))
        return transformation

    

    def store_features(self,type_,value):
        """
        purpose:
            - stores features for the data set
        input:
            - string,int,dataframe
        returns:
            - dataframe
        """
        features = [None]
        if type_ == "numeric":
            features = self.df.select_dtypes(include=value).columns.tolist()
        elif type_ == "categorical":
            features = self.df.select_dtypes(exclude=value).columns.tolist()
        return features

    def encoding_data(self):
        """
        - responsible for encoding the columns

        """
        categorical_features = self.store_features("categorical","number")
        to_one_hot_encoding = [col for col in categorical_features if self.df[col].nunique() <= 10 and self.df[col].nunique() > 2]
        # Get Categorical Column names thoose are not in "to_one_hot_encoding"
        to_label_encoding = [col for col in categorical_features if not col in to_one_hot_encoding]
        return to_one_hot_encoding,to_label_encoding

    def hot_encode(self):
        """
        - responsible one hot encoding the columns
        """
        to_one_hot_encoding,_= self.encoding_data()
        one_hot_encoded_columns = pd.get_dummies(self.df[to_one_hot_encoding])
        return one_hot_encoded_columns

    def label_encode(self):
        """
        - responsible for label ecoding the column
        """
        _,to_label_encoding = self.encoding_data()
        label_encoded_columns = []
        # For loop for each columns
        for col in to_label_encoding:
            # We define new label encoder to each new column
            le = LabelEncoder()
            # Encode our data and create new Dataframe of it, 
            # notice that we gave column name in "columns" arguments
            column_dataframe = pd.DataFrame(le.fit_transform(self.df[col]), columns=[col] )
            # and add new DataFrame to "label_encoded_columns" list
            label_encoded_columns.append(column_dataframe)

        # Merge all data frames
        label_encoded_columns = pd.concat(label_encoded_columns, axis=1)
        return label_encoded_columns
    
    def merge_data(self):
        """
        - responsible for bringing all the data together
        """
        # Copy our DataFrame to X variable
        X = self.df.copy()

        # Droping Categorical Columns,
        # "inplace" means replace our data with new one
        # Don't forget to "axis=1"
        categorical_features = self.store_features("categorical","number")
        X.drop(categorical_features, axis=1, inplace=True)

        one_hot_encoded_columns = self.hot_encode()
        label_encoded_columns = self.label_encode()
        # Merge DataFrames
        X = pd.concat([X, one_hot_encoded_columns, label_encoded_columns], axis=1)
        return X

    def get_columns(self):
        """
        - responsible for getting the columns
        """
        y = self.df["yes"]

        # Droping "class" from X
        X = self.merge_data()
        X.drop(["yes"], axis=1, inplace=True)
        return X,y

    def split_data(self):
        """
        - responsible for splitting the data
        """
        X,y =self.get_columns()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
        return X_train, X_test, y_train, y_test

    def model(self,model,**kwargs):
        """
        - model the dataset
        """
        X_train, X_test, y_train, y_test = self.split_data()
        # Define Random Forest Model
        model = model(**kwargs)
        # We fit our model with our train data
        model.fit(X_train, y_train)
        # Then predict results from X_test data
        predicted_data = model.predict(X_test)
        # generate a confusion matrix
        confusion_mat = confusion_matrix(y_test, predicted_data)
        # get accuracy score
        accuracy = accuracy_score(y_test, predicted_data)
        return confusion_mat,accuracy
    
    def hyperparameters(self,parameters):
        """
        -hyperparameters
        """
        X_train, X_test, y_train, y_test = self.split_data()
        
        accuracy = []
        for key in parameters.keys():
            search = GridSearchCV(parameters[key]['model'],parameters[key]['paramas'],cv=5,return_train_score=False,verbose = 2)
            search.fit(X_train,y_train)
            
            accuracy.append({'model':key,'best_score': search.best_score_,'best_parameters':search.best_params_})
         
        return accuracy 
        

if __name__=="__main__":
    df = pd.read_csv("data/data.csv")
    analyzer = Modeler(df)
    numeric_pipeline = analyzer.generate_pipeline("numeric")
    numeric_transformation =  analyzer.generate_transformation(numeric_pipeline,"numeric","number")
    numerical_features = analyzer.store_features("numeric","number")
    categorical_features = analyzer.store_features("categorical","number")
    print(numeric_transformation)