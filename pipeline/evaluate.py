import pandas as pd
import os,sys
print(os.getcwd())
sys.path.append(os.path.abspath(os.path.join('..')))
from scripts.modeling import Modeler
from mlflow import log_metric, log_param, log_metrics,sklearn,start_run
from random import random, randint
# evaluate a logistic regression model using k-fold cross-validation
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

if __name__=='__main__':
    df = pd.read_csv("../data/AdSmartABdata.csv")
    model_=Modeler(df)
    model = model_.get_model()
    X,y =model_.get_columns()
    fold = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    cv = KFold(n_splits=fold, random_state=1, shuffle=True)
    score,min_,max_=model_.evaluate(cv)
    metrics = {"score": score, "min":min_,"max":max_}
    log_metrics(metrics)
    sklearn.log_model(model, "model")

