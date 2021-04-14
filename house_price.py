# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 16:02:43 2020

@author: Lucas
"""
########################## import ####################
######################################################

import csv
import numpy as np
import pandas as pd
import os
from sklearn import preprocessing
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import SGDRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
import scipy.stats
from datetime import datetime 
from module_function import describe, check_value_error, continu_mod
from module_graph import graph_desc_matplot, graph_desc_seaborn


######################### main #######################
######################################################

##### load data set

os.getcwd()
os.chdir(r"D:\Kaggle\house_price")

train_data = pd.read_csv("train.csv", sep = ",", header = 0 )
test_data = pd.read_csv("test.csv", sep = ",", header = 0 )
price_test = pd.read_csv("sample_submission.csv", sep = ",", header = 0 )
# train_data.fillna(value = "", inplace = True)
train_X = train_data.drop(columns="SalePrice")
train_Y = train_data["SalePrice"]
test_Y = price_test["SalePrice"]
train_Y = train_Y.astype(dtype = float)

test_all = test_data.merge(price_test,left_on="Id",right_on="Id")
all_data = pd.concat((test_all,train_data), axis = 0)
all_data.index = range(0,2919)
###################### description data variables ####################

###### quality data

desc,dim_data = describe(all_data)
desc.missing

# we see that variables "Alley" "PoolQC" "Fence" "MiscFeature" "FireplaceQu"
# have a pourcentage more than 45% of missing value so we xhoose to delete this variables 

drop_var = ["Alley","PoolQC","Fence","MiscFeature","FireplaceQu"]
all_data = all_data.drop(columns= drop_var)
desc = desc.drop(drop_var)


for i in all_data.columns:
    all_data[i].fillna(value = desc["mode"].loc[i], inplace = True)


### aberrant value

datetime.now().year

val_err, index_gar = check_value_error(all_data["GarageYrBlt"], value_lim_min=0, value_lim_max=2021)
all_data["GarageYrBlt"][index_gar]=2007


######################## matplotlib #############################

## some matplotlib graph to describe variables distribution

##### qualitative var


###########################################################################################################################
###########################################################################################################################


"""

graph_desc_matplot(all_data)
graph_desc_seaborn(all_data)

"""

##### bivariate analyse


######################### crosstab 

cross_cond_qual = pd.crosstab(all_data["OverallCond"],all_data["OverallQual"])


#### bivariate correlation

corr = all_data[all_data.SalePrice>1].corr()
top_corr_cols = corr[abs((corr.SalePrice)>=.26)].SalePrice.sort_values(ascending=False).keys()
top_corr = corr.loc[top_corr_cols, top_corr_cols]
plt.figure(figsize=(20, 20))
fig = sns.heatmap(top_corr, cmap=sns.diverging_palette(220, 10, as_cmap=True), annot=True, fmt=".2f")
sns.set(font_scale=0.5)
fig = fig.get_figure()
fig.savefig(os.path.join("graph_desc","bivariate_corr.png"))
plt.show()


### quanti x quanti

scipy.stats.spearmanr(all_data["SalePrice"],all_data["MasVnrArea"])
scipy.stats.spearmanr(all_data["SalePrice"],all_data["YearBuilt"])
scipy.stats.spearmanr(all_data["SalePrice"],all_data["TotalBsmtSF"])
scipy.stats.spearmanr(all_data["SalePrice"],all_data["GrLivArea"])
scipy.stats.spearmanr(all_data["SalePrice"],all_data["1stFlrSF"])
scipy.stats.spearmanr(all_data["SalePrice"],all_data["2ndFlrSF"])
scipy.stats.spearmanr(all_data["SalePrice"],all_data["LowQualFinSF"])

### too much coor 

#TotRmsAbvGrd
#2ndFlrSF
#1stFlrSF
#GarageCars
#GarageYrBlt


### quali x quanti

scipy.stats.chi2_contingency(cross_cond_qual)

### cat binarymod 
sal_centrair = continu_mod(all_data,"SalePrice", "CentralAir","N","Y")
scipy.stats.ttest_ind(sal_centrair["cont_mod0"],sal_centrair["cont_mod1"],equal_var=False)


### cat multimod
sal_heating = continu_mod(all_data,"SalePrice", "Heating","Floor","GasA","GasW","Grav","OthW","Wall")
scipy.stats.f_oneway(sal_heating['cont_mod0'],sal_heating['cont_mod1'],sal_heating['cont_mod2'],sal_heating['cont_mod3'],sal_heating['cont_mod4'],sal_heating['cont_mod5'])

sal_util = continu_mod(all_data,"SalePrice", "Utilities","AllPub","NoSewr","NoSeWa","ELO")
scipy.stats.f_oneway(sal_util['cont_mod0'],sal_util['cont_mod2'])

sal_lots = continu_mod(all_data,"SalePrice", "LotShape","Reg","IR1","IR2","IR3")
scipy.stats.f_oneway(sal_lots['cont_mod0'],sal_lots['cont_mod1'],sal_lots['cont_mod2'],sal_lots['cont_mod3'],sal_lots['cont_mod4'],sal_heating['cont_mod5'])

sal_roof = continu_mod(all_data,"SalePrice", "RoofStyle","Flat","Gable","Gambrel","Hip","Mansard","Shed")
scipy.stats.f_oneway(sal_roof['cont_mod0'],sal_roof['cont_mod1'],sal_roof['cont_mod2'],sal_roof['cont_mod3'],sal_roof['cont_mod4'],sal_roof['cont_mod5'])

sal_mastype = continu_mod(all_data,"SalePrice", "MasVnrType","BrkCmn","BrkFace","CBlock","None","Stone")
scipy.stats.f_oneway(sal_mastype['cont_mod0'],sal_mastype['cont_mod1'],sal_mastype['cont_mod3'],sal_mastype['cont_mod4'])

sal_exqual = continu_mod(all_data,"SalePrice", "ExterQual","Ex","Gd","Ta","Fa","Po")
scipy.stats.f_oneway(sal_exqual['cont_mod0'],sal_exqual['cont_mod1'],sal_exqual['cont_mod3'])

sal_garquali = continu_mod(all_data,"SalePrice", "GarageQual","Ex","Gd","Ta","Fa","Po","Na")
scipy.stats.f_oneway(sal_garquali['cont_mod0'],sal_garquali['cont_mod1'],sal_garquali['cont_mod3'],sal_heating['cont_mod4'],sal_heating['cont_mod5'])

## delete some variables too much coorelate 

all_data_nocorr = all_data.drop(columns=["TotRmsAbvGrd","2ndFlrSF","1stFlrSF","GarageCars","GarageYrBlt"])

## make test and train data

data_test_ = all_data.iloc[:1459]
data_train_ = all_data.iloc[1459:]

data_train_x = data_train_.drop(["SalePrice","Id"], axis=1)
data_train_y = data_train_["SalePrice"]
data_test_x = data_test_.drop(["SalePrice","Id"], axis=1)
data_test_y = data_test_["SalePrice"]

#####  modèle for predictin
########### just continue variables for some predict modèle

train_data_x = data_train_[[ "LotArea" ,"YearBuilt", "MasVnrArea", "TotalBsmtSF", "GrLivArea","GarageArea" ,"OverallQual", "EnclosedPorch", "OpenPorchSF", "OverallCond"
                           ]]

test_data_x = data_test_[[ "LotArea" ,"YearBuilt", "MasVnrArea", "TotalBsmtSF", "GrLivArea","GarageArea" ,"OverallQual", "EnclosedPorch", "OpenPorchSF", "OverallCond"
                           ]]

#########
#### Scale 

sc = StandardScaler()
X_train = sc.fit_transform(train_data_x)
X_test = sc.fit_transform(test_data_x)
Y_train =  np.reshape(sc.fit_transform(pd.DataFrame(data_train_y)),1460)
Y_test = np.reshape(sc.fit_transform(pd.DataFrame(data_test_y)),1459)



######################## MLPRegressor

reg_mlp = MLPRegressor(random_state=1, max_iter=500, hidden_layer_sizes =200, learning_rate = "constant")
reg_mlp.fit(X_train, Y_train)
reg_mlp.score(X_train, Y_train)
reg_mlp.score(X_test, Y_test)
pred_mlp_train = reg_mlp.predict(X_train)
comp_mlp_train_pred_true = pd.DataFrame(np.transpose((pred_mlp_train, Y_train)))
comp_mlp_train_pred_true.columns = ["pred", "true"]
fig = sns.scatterplot(comp_mlp_train_pred_true["pred"],comp_mlp_train_pred_true["true"])
fig = fig.get_figure()
fig.savefig(os.path.join("graph_desc","res_reg-mlp_train.png"))
plt.plot(np.linspace(-2,7), np.linspace(-2,7))
plt.show()


pred_mlp = reg_mlp.predict(X_test)
comp_mlp_pred_true = pd.DataFrame(np.transpose((pred_mlp, Y_test)))
comp_mlp_pred_true.columns = ["pred", "true"]
fig = sns.scatterplot(comp_mlp_pred_true["pred"],comp_mlp_pred_true["true"])
fig = fig.get_figure()
fig.savefig(os.path.join("graph_desc","res_reg-mlp_test.png"))
plt.plot(np.linspace(-2,7), np.linspace(-2,7))
plt.show()

#################  Gradient boosting regressor (for more large dataset)

reg_grad= GradientBoostingRegressor()
reg_grad.fit(X_train, Y_train)
reg_grad.score(X_train, Y_train)
reg_grad.score(X_test, Y_test)

pred_grad_train = reg_grad.predict(X_train)
comp_grad_train_pred_true = pd.DataFrame(np.transpose((pred_grad_train, Y_train)))
comp_grad_train_pred_true.columns = ["pred", "true"]
fig = sns.scatterplot(comp_grad_train_pred_true["pred"],comp_grad_train_pred_true["true"])
fig = fig.get_figure()
fig.savefig(os.path.join("graph_desc","res_Grad-boost-reg_train.png"))
plt.plot(np.linspace(-2,7), np.linspace(-2,7))
plt.show()

pred_grad = reg_grad.predict(X_test)
comp_grad_pred_true = pd.DataFrame(np.transpose((pred_grad, Y_test)))
comp_grad_pred_true.columns = ["pred", "true"]
fig = sns.scatterplot(comp_grad_pred_true["pred"],comp_grad_pred_true["true"])
fig = fig.get_figure()
fig.savefig(os.path.join("graph_desc","res_Grad-boost-reg_test.png"))
plt.plot(np.linspace(-2,7), np.linspace(-2,7))
plt.show()


########################## knn regressor ############
res = []


for i in range(1,20):
    
    neigh = KNeighborsRegressor(n_neighbors=i)
    neigh.fit(X_train, Y_train)
    ne_score_train = neigh.score(X_train, Y_train)
    ne_score_test = neigh.score(X_test, Y_test)
    res.append([i,ne_score_train,ne_score_test])

neigh = KNeighborsRegressor(n_neighbors=2)
neigh.fit(X_train, Y_train)
pred_neigh_train = neigh.predict(X_train)
comp_neigh_train_pred_true = pd.DataFrame(np.transpose((pred_neigh_train, Y_train)))
comp_neigh_train_pred_true.columns = ["pred", "true"]
fig = sns.scatterplot(comp_neigh_train_pred_true["pred"],comp_neigh_train_pred_true["true"])
fig = fig.get_figure()
fig.savefig(os.path.join("graph_desc","res_knn-reg_train.png"))
plt.plot(np.linspace(-2,7), np.linspace(-2,7))
plt.show()


pred_neigh_test = neigh.predict(X_test)
comp_neigh_test_pred_true = pd.DataFrame(np.transpose((pred_neigh_test, Y_test)))
comp_neigh_test_pred_true.columns = ["pred", "true"]
fig = sns.scatterplot(comp_neigh_test_pred_true["pred"],comp_neigh_test_pred_true["true"])
fig = fig.get_figure()
fig.savefig(os.path.join("graph_desc","res_knn-reg_test.png"))
plt.plot(np.linspace(-2,7), np.linspace(-2,7))
plt.show()


std_y_train = np.std(data_train_y)
mean_y_train = np.mean(data_train_y)
sub_unscale = [(i *std_y_train)+mean_y_train for i in pred_neigh_test]
sub = pd.DataFrame(np.transpose((test_all['Id'],sub_unscale)))
sub.columns = ["Id","submission"]
sub.to_csv(os.path.join("house_price_result","submission_house_price.csv"),columns=sub.columns, index = False, sep = ",")

########################  test bullshit solution

### I try the labelencoder but it's not a god solution for me because transform 
## a categorial variable in continue variable isn't good. the différence between 
## 1 and 3 isn't the same that between "a" and "c" for exemple.
"""
from sklearn.preprocessing import LabelEncoder
catagory_cols = ('MSZoning','Street','LotShape','LandContour','Utilities','LotConfig','LandSlope','Neighborhood','Condition1','Condition2','BldgType', 'HouseStyle', 'RoofStyle','RoofMatl','Exterior1st','Exterior2nd','ExterCond','Foundation','Heating','HeatingQC','CentralAir','KitchenQual','Functional','PavedDrive','SaleType','SaleCondition', "GarageType", "GarageFinish", "GarageQual", "GarageCond", "BsmtFinType2", "BsmtCond", "BsmtQual", "BsmtExposure", "MasVnrType", "Electrical", "BsmtFinType1", "ExterQual")
for c in catagory_cols:
  le = LabelEncoder()
  data_train_x[c]= le.fit_transform(data_train_x[c].values)
  data_test_x[c]= le.fit_transform(data_test_x[c].values)


sc = StandardScaler()
X_train = sc.fit_transform(data_train_x)
X_test = sc.transform(data_test_x)

from sklearn.decomposition import PCA
pca = PCA(n_components = 10)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)


from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 200, random_state = 0)
regressor.fit(X_train, data_train_y)
Y_pred = regressor.predict(X_test)

regressor.score(X_train, data_train_y)

### poor result

regressor.score(X_test, data_test_y)
df_matrix = pd.DataFrame(np.transpose((Y_pred, data_test_y)))
df_matrix.columns = ["pred", "true"]
sns.scatterplot(df_matrix["pred"],df_matrix["true"])

########################################################

sc = StandardScaler()
X_train = sc.fit_transform(data_train_x)
X_test = sc.transform(data_test_x)


pca = PCA(n_components = 4)
X_train_ = pca.fit_transform(X_train)
X_test_ = pca.transform(X_test)

from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 200, random_state = 0)
regressor.fit(X_train_, data_train_y)
Y_pred = regressor.predict(X_test_)

regressor.score(X_train_, data_train_y)

regressor.score(X_test_, data_test_y)
df_matrix = pd.DataFrame(np.transpose((Y_pred, data_test_y)))
df_matrix.columns = ["pred", "true"]
ax = sns.scatterplot(df_matrix["pred"],df_matrix["true"])
ax.plot(50000,50000, color='r')
plt.plot(np.linspace(50000,600000), np.linspace(50000,600000))
plt.xlim(50000,600000)
plt.ylim(50000,600000)
plt.show()

"""