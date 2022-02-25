import os
import argparse
import numpy as np
import random
import pandas as pd
from sklearn import preprocessing
from sklearn.metrics import roc_auc_score
from sklearn import model_selection
import lightgbm as lgbm
import enum
import math
import glob
from tqdm import tqdm

import warnings
warnings.filterwarnings('ignore')

def optimize(trial,x,y): 
    params = {
        'objective': 'CrossEntropy', 
        'bootstrap_type': 'Bernoulli', 
        'iterations':trial.suggest_int("iterations", 4000, 25000),
        'od_wait':trial.suggest_int('od_wait', 500, 2300),
        'learning_rate' : trial.suggest_uniform('learning_rate',0.01, 1),
        'reg_lambda': trial.suggest_uniform('reg_lambda',1e-5,100),
        'subsample': trial.suggest_uniform('subsample',0,1),
        'random_strength': trial.suggest_uniform('random_strength',10,50),
        'depth': trial.suggest_int('depth',1, 15),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf',1,30),
        'leaf_estimation_iterations': trial.suggest_int('leaf_estimation_iterations',1,15),
        'task_type' : 'GPU',
        'devices' : '0',
        'verbose' : 0,
        'eval_metric':'AUC'
    }
    
    num_rounds=args.n_iters
	
    auc_score = []
        
    kf = model_selection.StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)

    for f, (train_idx, val_idx) in tqdm(enumerate(kf.split(x, y))):
        df_train, df_val = x.iloc[train_idx], x.iloc[val_idx]
        train_target, val_target = y[train_idx], y[val_idx]
        eval_set = [(df_val, val_target)]   
	
        model = CatBoostClassifier(**params)
        model.fit(
		    df_train, 
		    train_target,
		    eval_set=eval_set, 
		    verbose=0, 
		    early_stopping_rounds=100)
        
        predicted = model.predict_proba(df_val)[:, 1]
        auc  = roc_auc_score(val_target, predicted)
        auc_score.append(auc)
      
    return np.mean(auc_score)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--path", type=str, default="../")
    parser.add_argument("--filename", type=str, default="train.csv")
    parser.add_argument("--n_trials", type=float, default=100, required=False)
    parser.add_argument("--es_stop", type=int, default=100, required=False)
    parser.add_argument("--n_iters", type=int, default=200, required=False)
    
    return parser.parse_args()


if __name__=='__main__':
    args = parse_args()
    df = pd.read_csv(os.path.join(args.path, args.filename))
    optimize_func=partial(optimize,x=df, y=df['target'].values)
    study = optuna.create_study(direction='maximize')
    study.optimize(optimize_func, n_trials=args.n_trials)
    trial = study.best_trial
    print('Score: {}'.format(trial.value))
    print("Best hyperparameters: {}".format(trial.params))
