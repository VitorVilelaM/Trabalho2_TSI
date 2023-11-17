import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.neural_network import MLPRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.inspection import permutation_importance


import numpy as np

dataBase = pd.read_csv('./Data.csv')  
dataBase = dataBase.dropna()

featuresTrain = dataBase[dataBase['Date'] < 20220000]
featuresTest = dataBase[dataBase['Date'] > 20220000]

train = featuresTrain.copy()
test = featuresTest.copy()

date = featuresTest['Date'].values
result = featuresTest['B'].values

caminho_arquivo = './src/ML/output.csv'
    
## Random Forest

def rf_holdout_Regressor():
    rf_train_input = train.drop(columns=['Date','B', 'CP', 'CP-1', 'CP-2', 'CP-3', 'CP-4', 'CP-5'])  # Features
    rf_train_output = rf_train_input.pop('R') # Target

    rf_test_input = test.drop(columns=['Date','B', 'CP', 'CP-1', 'CP-2', 'CP-3', 'CP-4', 'CP-5'])  # Features
    rf_test_output = rf_test_input.pop('R') # Target

    model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=2)
    model.fit(rf_train_input, rf_train_output)

    predicts = model.predict(rf_test_input)

    mae = mean_absolute_error(rf_test_output, predicts)

    print(f"MAE RF Regressor: ", mae)

def rf_holdout_Classifier():
    # Random Forest com Holdout
    rf_train_input = train.drop(columns=['Date','R', 'R-1', 'R-2', 'R-3', 'R-4', 'R-5'])  # Features
    rf_train_output = rf_train_input.pop('B') # Target

    date =  test['Date'].values

    rf_test_input = test.drop(columns=['Date','R', 'R-1', 'R-2', 'R-3', 'R-4', 'R-5'])  # Features
    rf_test_output = rf_test_input.pop('B') # Target

    # Criando o meu Baseline
    baseline = []
    for i in range(0, rf_test_output.shape[0]):
        baseline.append(1)

    RandomForest = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=2)
    RandomForest.fit(rf_train_input, rf_train_output.values)

    predict = RandomForest.predict(rf_test_input)

    
    accuracy_baseline = accuracy_score(rf_test_output, baseline)
    accuracy_model = accuracy_score(rf_test_output, predict)
    auc_model = roc_auc_score(rf_test_output, predict)
    precision_model = precision_score(rf_test_output, predict)
    recall_model = recall_score(rf_test_output, predict)

    print("Acurácia do RF:", accuracy_model)
    print("Baseline: ", accuracy_baseline)

def rf_janela_deslizante_Classifier(num_days):
    # Random Forest com Janela Deslizante  
    jd_train_input = train.drop(columns=['Date','R'])  # Features
    jd_train_output = jd_train_input.pop('B') # Target

    jd_test_input = test.drop(columns=['R'])  # Features
    jd_test_output = jd_test_input.pop('B') # Target
    
    limite = jd_test_input.shape[0]

    new_train_input = jd_train_input
    new_train_output = jd_train_output

    date = jd_test_input['Date'].values
    jd_test_input = jd_test_input.drop(columns='Date')

    copy_test_output = jd_test_output.copy()

    predicts = []
    out = []

    while( limite > 0):
        limite = jd_test_input.shape[0]

        if(num_days > limite):
            num_days = limite
       
        new_features = jd_test_input.iloc[0:num_days]
        jd_test_input = jd_test_input.drop(jd_test_input.index[0:num_days])    

        new_target = jd_test_output.iloc[0:num_days]
        jd_test_output = jd_test_output.drop(jd_test_output.index[0:num_days])    
        
        if(limite > 0):
            # Valores do random_state
            # 1 - 22
            # 5 - 5
            # 5 - 1

            RandomForest = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=5)
            RandomForest.fit(new_train_input, new_train_output.values)
            
            predict = RandomForest.predict(new_features)
            predicts.append(predict)
    
        new_train_input = pd.concat([new_train_input, new_features])
        new_train_output = pd.concat([new_train_output, new_target])
        
    predicts = [elemento for sublista in predicts for elemento in sublista]

    for i in range(len(predicts)):
        out.append([date[i], predicts[i]])

    df = pd.DataFrame(out)
    df.to_csv(caminho_arquivo, index=False)

    accuracy_model = accuracy_score(copy_test_output, predicts)
    auc_model = roc_auc_score(copy_test_output, predicts)
    precision_model = precision_score(copy_test_output, predicts)
    recall_model = recall_score(copy_test_output, predicts)

    print("Acurácia do Modelo:", accuracy_model)
    print("AUC:", auc_model)
    print("Recall:", recall_model)
    print("Precision:", precision_model)

## Rede Neural - MLP

def mlp_holdout_Regressor():
    mlp_train_input = train.drop(columns=['Date','B', 'CP', 'CP-1', 'CP-2', 'CP-3', 'CP-4', 'CP-5'])  # Features
    mlp_train_output = mlp_train_input.pop('R') # Target

    mlp_test_input = test.drop(columns=['Date','B', 'CP', 'CP-1', 'CP-2', 'CP-3', 'CP-4', 'CP-5'])  # Features
    mlp_test_output = mlp_test_input.pop('R') # Target

    baseline = [0]
    for i in range(1, mlp_test_output.shape[0]):
        baseline.append(mlp_test_output.values[i - 1])


    mlp = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, learning_rate_init=0.1, activation="tanh",random_state= 42)
    
    # Treine o modelo
    mlp.fit(mlp_train_input, mlp_train_output)    
    predict = mlp.predict(mlp_test_input)

    
    mae = mean_absolute_error(mlp_test_output,predict)
    
    print("MAE MLP Regressor: {:.2f}".format(mae))

def mlp_holdout_Classifier():
    mlp_train_input = train.drop(columns=['Date','R', 'R-1', 'R-2', 'R-3', 'R-4', 'R-5'])  # Features
    mlp_train_output = mlp_train_input.pop('B') # Target

    mlp_test_input = test.drop(columns=['Date','R', 'R-1', 'R-2', 'R-3', 'R-4', 'R-5'])  # Features
    mlp_test_output = mlp_test_input.pop('B') # Target

    mlp = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500,learning_rate_init=0.1, activation='', random_state=42)

    # Treine o modelo
    mlp.fit(mlp_train_input,mlp_train_output)    
    predict = mlp.predict(mlp_test_input)

    accuracy_model = accuracy_score(mlp_test_output, predict)

    print("Acurácia do MLP:", accuracy_model)

def mlp_janela_deslizante_Regressor(num_days):
    
    jd_train_input = train.drop(columns=['Date','B'])  # Features
    jd_train_output = jd_train_input.pop('R') # Target

    jd_test_input = test.drop(columns=['B'])  # Features
    jd_test_output = jd_test_input.pop('R') # Target
    
    limite = jd_test_input.shape[0]

    new_train_input = jd_train_input
    new_train_output = jd_train_output

    date = jd_test_input['Date'].values
    jd_test_input = jd_test_input.drop(columns='Date')

    copy_test_output = jd_test_output.copy()

    predicts = []
    out = []

    while( limite > 0):
        limite = jd_test_input.shape[0]

        if(num_days > limite):
            num_days = limite
       
        new_features = jd_test_input.iloc[0:num_days]
        jd_test_input = jd_test_input.drop(jd_test_input.index[0:num_days])    

        new_target = jd_test_output.iloc[0:num_days]
        jd_test_output = jd_test_output.drop(jd_test_output.index[0:num_days])    
        
        if(limite > 0):
            mlp = MLPRegressor(hidden_layer_sizes=(10, 5),
                    max_iter=5000,
                    learning_rate_init=0.1,
                    solver="lbfgs",
                    activation="logistic",
                    learning_rate="adaptive",
            )
            
            # Treine o modelo
            mlp.fit(new_train_input,new_train_output)    
            predict = mlp.predict(new_features)

            predicts.append(predict)
    
        new_train_input = pd.concat([new_train_input, new_features])
        new_train_output = pd.concat([new_train_output, new_target])
        
    predicts = [elemento for sublista in predicts for elemento in sublista]
    
    for i in range(len(predicts)):
        out.append([date[i], predicts[i]])

    df = pd.DataFrame(out)
    df.to_csv(caminho_arquivo, index=False)

    mae = mean_absolute_error(copy_test_output,predicts)
    mse = mean_squared_error(copy_test_output,predicts)
    rmse = np.sqrt(mse)

    print(f"Valor de MAE: {mae}")
    print(f"Valor de RMSE: {rmse}")

## XGBoost 

def xgboost_holdout_Regressor():
    xgboost_train_input = train.drop(columns=['Date','B', 'CP', 'CP-1', 'CP-2', 'CP-3', 'CP-4', 'CP-5'])  # Features
    xgboost_train_output = xgboost_train_input.pop('R') # Target

    xgboost_test_input = test.drop(columns=['Date','B', 'CP', 'CP-1', 'CP-2', 'CP-3', 'CP-4', 'CP-5'])  # Features
    xgboost_test_output = xgboost_test_input.pop('R') # Target

    model = XGBRegressor()
    model.fit(xgboost_train_input, xgboost_train_output, verbose=True)
    predictions = model.predict(xgboost_test_input)

    print("MAE XGBoost Regressor: {:.2f}".format(mean_absolute_error(predictions, xgboost_test_output)))

def xgboost_holdout_Classifier():
    xgboost_train_input = train.drop(columns=['Date','R', 'R-1', 'R-2', 'R-3', 'R-4', 'R-5'])  # Features
    xgboost_train_output = xgboost_train_input.pop('B') # Target

    xgboost_test_input = test.drop(columns=['Date','R', 'R-1', 'R-2', 'R-3', 'R-4', 'R-5'])  # Features
    xgboost_test_output = xgboost_test_input.pop('B') # Target

    model = XGBClassifier()
    model.fit(xgboost_train_input, xgboost_train_output)
    predictions = model.predict(xgboost_test_input)

    accuracy_model = accuracy_score(xgboost_test_output, predictions)
    print("Acurácia do XGBoost:", accuracy_model)

# KNN

def knn_holdout_Regressor():
    knn_train_input = train.drop(columns=['Date','B', 'CP', 'CP-1', 'CP-2', 'CP-3', 'CP-4', 'CP-5'])  # Features
    knn_train_output = knn_train_input.pop('R') # Target

    knn_test_input = test.drop(columns=['Date','B', 'CP', 'CP-1', 'CP-2', 'CP-3', 'CP-4', 'CP-5'])  # Features
    knn_test_output = knn_test_input.pop('R') # Target

    model = KNeighborsRegressor(n_neighbors=5)

    model.fit(knn_train_input, knn_train_output)
    predicts = model.predict(knn_test_input)

    mae = mean_absolute_error(knn_test_output, predicts)
    print("MAE KNN Regressor: ", mae)

def knn_holdout_Classifier():
    knn_train_input = train.drop(columns=['Date','R', 'R-1', 'R-2', 'R-3', 'R-4', 'R-5'])  # Features
    knn_train_output = knn_train_input.pop('B') # Target

    knn_test_input = test.drop(columns=['Date','R', 'R-1', 'R-2', 'R-3', 'R-4', 'R-5'])  # Features
    knn_test_output = knn_test_input.pop('B') # Target

    model = KNeighborsClassifier()
    model.fit(knn_train_input, knn_train_output)
    predictions = model.predict(knn_test_input)

    accuracy_model = accuracy_score(knn_test_output, predictions)
    print("Acurácia do KNN:", accuracy_model)
  
# Naive Bayes  

def naive_bayes_houdolt_Classifier():
    nb_train_input = train.drop(columns=['Date','R', 'R-1', 'R-2', 'R-3', 'R-4', 'R-5'])  # Features
    nb_train_output = nb_train_input.pop('B') # Target

    nb_test_input = test.drop(columns=['Date','R', 'R-1', 'R-2', 'R-3', 'R-4', 'R-5'])  # Features
    nb_test_output = nb_test_input.pop('B') # Target
    
    model = GaussianNB()
    model.fit(nb_train_input, nb_train_output)
    predicts = model.predict(nb_test_input)
    print(predicts)
    accuracy = accuracy_score(nb_test_output, predicts) 

    print("Acuracia Naive Bayes: ", accuracy)   

def execute_Classifiers():    
    rf_holdout_Classifier()
    mlp_holdout_Classifier()
    knn_holdout_Classifier()
    xgboost_holdout_Classifier()
    naive_bayes_houdolt_Classifier()

def execute_Regressors():
    rf_holdout_Regressor()
    mlp_holdout_Regressor()
    knn_holdout_Regressor()
    xgboost_holdout_Regressor()


execute_Regressors()

