#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso, LassoCV, Ridge, RidgeCV, LinearRegression, lasso_path

hp = pd.read_csv("train.csv", index_col="Id")
print(hp.info())


# Remplacer valeurs manquantes par la moyenne de chaque variable

liste_num = ["LotFrontage","MasVnrArea", "GarageYrBlt" ]

liste_cat = ["Alley","MasVnrType","BsmtQual","BsmtCond", "BsmtExposure", "BsmtFinType1",
            "BsmtFinType2","Electrical","FireplaceQu","GarageType","GarageFinish","GarageQual",
            "GarageCond", "PoolQC", "Fence", 'MiscFeature']

for i in liste_num :
    hp[i].fillna(hp[i].mean(), inplace=True)
    
for j in liste_cat:
    hp[j].fillna(hp[j].mode()[0], inplace=True)


print(hp.isna().sum(),"\n")
print(hp.info())



#Centrer et réduire les variables num de hp
from sklearn.preprocessing import StandardScaler

hp_num = hp.select_dtypes(include =["int64", "float64"])

hp_cat = hp.select_dtypes(include = ["object"])


hp_num[hp_num.columns] = StandardScaler().fit_transform(hp_num)


# transformation des variables catégorielles en variables indicatrices
hp_cat = pd.get_dummies(hp_cat, prefix=hp_cat.columns, prefix_sep="_")

print(hp_cat.info())

# Fusion de hp_num et hp_cat
hp_2 = hp_num.merge(right = hp_cat, on="Id",how="inner").astype("float64")
hp_2.info()



# Définition variables explicatives et cible
target = hp_2["SalePrice"]
feats = hp_2.drop("SalePrice", axis=1)


# Séparation des données en jeu d'entraînement et de test
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(feats, target, test_size = 0.2, random_state=123)



# Création fonction rmse_cv
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, make_scorer

def rmse_cv(model):
    #scorer = make_scorer(mean_squared_error)
    model.fit(X_train, y_train)
    cv = cross_val_score(model, X_train, y_train, scoring= "neg_mean_squared_error", cv=5)
    return abs(cv) #valeur absolue de neg_mse = mse
    


# Création fonction rmse_cv
from sklearn.model_selection import cross_val_score
from sklearn import metrics

def rmse_cv(model):
    model.fit(X_train, y_train)
    cv = cross_val_score(model, X_train, y_train, cv=5)
    y_pred = model.predict(x)
    return print("RMSE cv :",np.sqrt((y_pred - y)**2).mean())
    

# Afficher dans un graphe la RMSEcv appliquée à Ridge pour chaque alpha
alphas = [0.01, 0.05, 0.1, 0.3, 0.8, 1,5,10,15,30,50]

import matplotlib.pyplot as plt
from sklearn.linear_model import RidgeCV
ridge_reg = RidgeCV(alphas=alphas)

RMSE = rmse_cv(ridge_reg)
print("Meilleur alpha :",ridge_reg.alpha_)

plt.figure(figsize=(10,8))
plt.plot(alphas[5:10],abs(RMSE))
plt.xlabel("alpha")
plt.ylabel("RMSE")
plt.title("RMSE en fonction des alpha");

# Pour alpha = 15, RMSE = 0.09 => pas mal



# Regression à partir d'un modèle Ridge et du paramètre alpha déterminé au dessus
from sklearn.linear_model import Ridge

model_1 = Ridge(alpha =15)
model_1.fit(X_train, y_train)

pred_train = model_1.predict(X_train)
pred_test = model_1.predict(X_test)

train_score = model_1.score(X_train, y_train)
test_score = model_1.score(X_test, y_test)

mse_train = mean_squared_error(pred_train, y_train)
mse_test = mean_squared_error(pred_test, y_test)

print("train score:", train_score)
print("test score:", test_score, "\n")
print("mse train:", mse_train)
print("mse test:", mse_test)


"""
Avec modèle Rigde(alpha=15), les scores de train et test sont proches, ainsi que MSE train et test
=> Pas d'overfitting observé et bonne prédiction (score élevé, mse faible)
"""


# Modèle de reg Lasso avec LassoCV
from sklearn.linear_model import LassoCV

alphas = [10,1,0.1, 0.001, 0.0005]
model_lasso = LassoCV(cv=10, alphas=alphas)

model_lasso.fit(X_train, y_train)

pred_train = model_lasso.predict(X_train)
pred_test = model_lasso.predict(X_test)

train_score = model_lasso.score(X_train, y_train)
test_score = model_lasso.score(X_test, y_test)

mse_train = mean_squared_error(pred_train, y_train)
mse_test = mean_squared_error(pred_test, y_test)

print("Meilleur alpha:", model_lasso.alpha_, "\n")
print("train score:", train_score)
print("test score:", test_score, "\n")
print("mse train:", mse_train)
print("mse test:", mse_test)

"""
Avec modèle Lasso(alpha=0.01), meilleures perfs que avec Ridge(alpha=15)
=> Score global meilleur que Ridge mais léger overfitting sur les données d'entraînement
    (train_score > test_score et mse_test > mse_train)
"""



from scipy.stats import probplot
pred_train= model_lasso.predict(X_train)
residus = pred_train - y_train
residus_norm = (residus-residus.mean())/residus.std()
probplot(residus_norm, plot=plt)
plt.show();

# hyp de normalité plausible, les points s'alignent autour de la droite



# Plot LassoCV     
alphas = model_lasso.alphas_
alpha = model_lasso.alpha_

plt.figure(figsize=(10,8))
plt.plot(alphas, model_lasso.mse_path_) # erreur MSE / alpha de chq echtillon
plt.plot(alpha, model_lasso.mse_path_.mean()) # moy score MSE sur tout echnt de alpha
plt.axvline(alpha)
plt.xlabel("alpha")
plt.ylabel("MSE");
print("meilleur alpha :", alpha)



# Nouveau df avec var explicatives et coef de lasso
lasso_coef = model_lasso.coef_
lasso_coef = pd.DataFrame(lasso_coef)
col = pd.DataFrame(feats.columns)

a = pd.concat([col, lasso_coef],axis=1)
a["coef"]= lasso_coef
a["col"] = col
a = a.drop(0, axis=1)
print(a)

# Conserve les coeffs > 0.1 pour élaguer les variables peu significatives et soucis de lecture
a2 = a[a["coef"] >= 0.1]
print(a2)




# Nombre de variables gardées / éliminées par model_lasso
plt.figure(figsize=(10,8))
plt.plot(a2.col, a2.coef)
plt.xticks(rotation = 90)
plt.xlabel("variables explicatives")
plt.ylabel("coefficient de Lasso");

# 19 variables avec un coeff > 0.1 (Choix plus discriminateur des variables)



# Nombre de variables gardées et éliminées par modèle LassoCV
print("Nombre de variables gardées :", len(a2["coef"]))
print("Nombre de variables enlevées :", len(a[a["coef"] < 0.1]))


print(a2.sort_values(by="coef",ascending=False))

"""
Les variables ci-dessous représentent les variables explicatives les plus importantes pour 
l'entraînement du modèle Lasso
"""



# Avec SelectFromModel
from sklearn.feature_selection import SelectFromModel

model_lass = Lasso(alpha=0.001)
sfm = SelectFromModel(model_lass)

sfm_train = sfm.fit_transform(X_train, y_train)
sfm_test = sfm.transform(X_test)

print(feats.columns[sfm.get_support()])


# Nouveau modèle à partir du sfm
model_final = Lasso(alpha=0.001)
model_final.fit(sfm_train, y_train)

train_score = model_final.score(sfm_train, y_train)
test_score = model_final.score(sfm_test, y_test)

print( train_score,"\n",test_score, "\n")

pred_train = model_final.predict(sfm_train)
pred_test = model_final.predict(sfm_test)

print("mse_train :", mean_squared_error(pred_train, y_train))
print("mse_test :", mean_squared_error(pred_test, y_test))

# Leger overfitting sur les données d'entraînement





