import numpy as np
import pandas as pd
import os
from sklearn import preprocessing
import matplotlib.pyplot as plt
import seaborn as sns



def graph_desc_matplot():

    ###qualité 
    fig, axes = plt.subplots()
    freq_overallqual = pd.value_counts(all_data["OverallQual"])
    plt.bar(Counter(all_data["OverallQual"]).keys(),freq_overallqual, orientation = "vertical", color = '#e377c2', edgecolor = '#2ca02c')
    plt.title("Distribution of OverallQual")
    plt.xlabel("OverallQual")
    fig.savefig(os.path.join("graph_desc","qualite.png"))

    ### condition général
    fig, axes = plt.subplots()
    freq_overallcond = pd.value_counts(all_data["OverallCond"])
    plt.bar(Counter(all_data["OverallCond"]).keys(),freq_overallcond, orientation = "vertical", color = 'blue', edgecolor = '#2ca02c')
    plt.title("Distribution of OverallCond")
    plt.xlabel("OverallCond")
    fig.savefig(os.path.join("graph_desc","condition_general.png"))

    ## type de toit
    fig, axes = plt.subplots()
    freq_RoofStyle = pd.value_counts(all_data["RoofStyle"])
    plt.bar(Counter(all_data["RoofStyle"]).keys(),freq_RoofStyle, orientation = "vertical", color = 'blue', edgecolor = '#2ca02c')
    plt.title("Distribution of RoofStyle")
    plt.xlabel("RoofStyle")
    fig.savefig(os.path.join("graph_desc","type_toit.png"))

    ## type de zone
    fig, axes = plt.subplots()
    freq_MSZoning = pd.value_counts(all_data["MSZoning"])
    plt.bar(Counter(all_data["MSZoning"]).keys(),freq_MSZoning, orientation = "vertical", color = 'blue', edgecolor = '#2ca02c')
    plt.title("Distribution of MSZoning")
    plt.xlabel("MSZoning")
    fig.savefig(os.path.join("graph_desc","type_zone.png"))

    ## type de rue 
    fig, axes = plt.subplots()
    freq_Street = pd.Series(pd.value_counts(all_data["Street"]))/sum(pd.value_counts(train_data["Street"]))
    plt.pie(freq_Street, labels = pd.value_counts(all_data["Street"]).index, autopct = '%1.1f%%', colors = ['#1f77b4', '#e377c2'])
    plt.title("Distribution of Street")
    fig.savefig(os.path.join("graph_desc","type_rue.png"))

    ### forme du terrain
    fig, axes = plt.subplots()
    freq_LotShape = pd.Series(pd.value_counts(all_data["LotShape"]))/sum(pd.value_counts(train_data["LotShape"]))
    plt.pie(freq_LotShape, labels = pd.value_counts(all_data["LotShape"]).index, autopct = '%1.1f%%', colors = ['#1f77b4', '#e377c2', "red","green"])
    plt.title("Distribution of LotShape")
    fig.savefig(os.path.join("graph_desc","forme_terrain.png"))


    ###### Electrical
    fig, axes = plt.subplots()
    freq_Electrical = pd.value_counts(all_data["Electrical"])
    plt.bar(Counter(all_data["Electrical"]).keys(),freq_Electrical, orientation = "vertical", color = 'blue', edgecolor = '#2ca02c')
    plt.title("Distribution of Electrical")
    plt.xlabel("Electrical")
    fig.savefig(os.path.join("graph_desc","elec.png"))

    ###### Foundation
    fig, axes = plt.subplots()
    freq_Foundation = pd.value_counts(all_data["Foundation"])
    plt.bar(Counter(all_data["Foundation"]).keys(),freq_Foundation, orientation = "vertical", color = 'blue', edgecolor = '#2ca02c')
    plt.title("Distribution of Foundation")
    plt.xlabel("Foundation")
    fig.savefig(os.path.join("graph_desc","foundation.png"))

    ##### Heating
    fig, axes = plt.subplots()
    freq_Heating = pd.Series(pd.value_counts(all_data["Heating"]))/sum(pd.value_counts(train_data["Heating"]))
    plt.pie(freq_Heating, labels = pd.value_counts(all_data["Heating"]).index, autopct = '%1.1f%%', colors = ['#1f77b4', '#e377c2', "red","green", "lightblue"])
    plt.title("Distribution of Heating")
    fig.savefig(os.path.join("graph_desc","heating.png"))

    #######   HeatingQC
    fig, axes = plt.subplots()
    freq_HeatingQC = pd.Series(pd.value_counts(all_data["HeatingQC"]))/sum(pd.value_counts(train_data["HeatingQC"]))
    plt.pie(freq_HeatingQC, labels = pd.value_counts(all_data["HeatingQC"]).index, autopct = '%1.1f%%', colors = ['#1f77b4', '#e377c2', "red","green", "lightblue"])
    plt.title("Distribution of HeatingQC")
    fig.savefig(os.path.join("graph_desc","heating_quali.png"))

    ###  GarageType
    fig, axes = plt.subplots()
    freq_GarageType = pd.value_counts(all_data["GarageType"])
    plt.bar(Counter(all_data["GarageType"]).keys(),freq_GarageType, orientation = "vertical", color = 'blue', edgecolor = '#2ca02c')
    plt.title("Distribution of GarageType")
    plt.xlabel("GarageType")
    fig.savefig(os.path.join("graph_desc","GarageType.png"))

    ### air centralisé
    fig, axes = plt.subplots()
    freq_CentralAir = pd.Series(pd.value_counts(all_data["CentralAir"]))/sum(pd.value_counts(train_data["CentralAir"]))
    plt.pie(freq_CentralAir, labels = pd.value_counts(all_data["CentralAir"]).index, autopct = '%1.1f%%', colors = ['#1f77b4', '#e377c2', "red","green", "lightblue"])
    plt.title("Distribution of CentralAir")
    fig.savefig(os.path.join("graph_desc","central_air.png"))

    ###### quantitative var 

    ### YearBuilt
    fig, axes = plt.subplots()
    plt.hist(all_data["YearBuilt"], density = False, cumulative = False, align = "mid")
    plt.xlabel("YearBuilt")
    plt.title("Distribution of YearBuilt")
    fig.savefig(os.path.join("graph_desc","annee_constr.png"))

    #### MasVnrArea: 
    fig, axes = plt.subplots()
    plt.hist(all_data["MasVnrArea"], density = True, cumulative = False, align = "mid")
    plt.xlabel("MasVnrArea")
    plt.title("Distribution of MasVnrArea")
    fig.savefig(os.path.join("graph_desc","taille_revetement.png"))

    ### TotalBsmtSF
    fig, axes = plt.subplots()
    plt.hist(all_data["TotalBsmtSF"], density = False, cumulative = False, align = "mid")
    plt.xlabel("TotalBsmtSF")
    plt.title("Distribution of TotalBsmtSF")
    fig.savefig(os.path.join("graph_desc","superficie_sous-sol.png"))

    #### GrLivArea
    fig, axes = plt.subplots()
    plt.boxplot([all_data["GrLivArea"]], autorange = True, labels=["GrLivArea"])
    plt.title("Distribution of GrLivArea")
    fig.savefig(os.path.join("graph_desc","superficie_dessus.png"))

    ### TotRmsAbvGrd
    fig, axes = plt.subplots()
    plt.boxplot([all_data["TotRmsAbvGrd"]], autorange = True, labels=["TotRmsAbvGrd"])
    plt.title("Distribution of TotRmsAbvGrd")
    fig.savefig(os.path.join("graph_desc","nb_piece_hors_sous-sol.png"))

    ### LotArea 
    fig, axes = plt.subplots()
    plt.boxplot([all_data["LotArea"]], autorange = True, labels=["LotArea"])
    plt.title("Distribution of LotArea")
    fig.savefig(os.path.join("graph_desc","taille_terrain.png"))

    ### SalePrice
    fig, axes = plt.subplots()
    box_saleprice = plt.boxplot([all_data["SalePrice"]], autorange = True, labels=["SalePrice"])
    plt.title("Distribution of SalePrice")
    [item.get_ydata() for item in box_saleprice['whiskers']]
    fig.savefig(os.path.join("graph_desc","prix.png"))

######################## seaborn #############################

## some seaborn graph to describe variables distribution



## boxplot

def graph_desc_seaborn():
    fig, axes = plt.subplots()
    box = all_data.boxplot(column = ["SalePrice"] ,by= "HeatingQC") ## pb title
    box.savefig(os.path.join("graph_desc","test.png"))
    ######

    box = sns.boxplot(x="HeatingQC", y="SalePrice", data=all_data).set_title("Distribution of SalePrice by Heating Quality")
    sns.set_context(font_scale = 3)
    fig = box.get_figure()
    fig.savefig(os.path.join("graph_desc","plot_HeatingQC_SalePrice.png"))


    sns.set_style("whitegrid")
    plot = sns.lmplot(x="GrLivArea", y="SalePrice", data=all_data,col="CentralAir",  hue="CentralAir" , height=2, scatter_kws={ "alpha": 1})
    plot.savefig(os.path.join("graph_desc","plot_GrLivArea_CentralAir.png"))
    ### smooth kernel density histo

    sns.set_style(style="white")
    g = sns.JointGrid(data=all_data, x="GrLivArea", y="SalePrice", space=0)
    g.plot_joint(sns.kdeplot, fill=True,
                 thresh=0,levels=100)
    g.plot_marginals(sns.distplot, color="#03051A")
    g.savefig(os.path.join("graph_desc","plot_GrLivArea_Saleprice.png"))

    ### hist x scatter

    plot = sns.jointplot(data=all_data, x="GrLivArea", y="SalePrice")
    plot.savefig(os.path.join("graph_desc","plot_GrLivArea_CentralAir.png"))
    #sns.jointplot(data=all_data, x="GrLivArea", y="SalePrice", hue="CentralAir") ## hue don't work

    #### paireplot

    scatt = pd.DataFrame(np.transpose((all_data["1stFlrSF"],all_data["2ndFlrSF"],all_data["LowQualFinSF"], all_data["GrLivArea"], all_data["CentralAir"])))

    plot = sns.pairplot(scatt, kind="scatter")
    plot.savefig(os.path.join("graph_desc","plot_box_scatt_1stF_2stF_GrLivArea_LowQualFinSF.png"))
    plot = sns.pairplot(scatt, kind="scatter",hue = 4)
    plot.savefig(os.path.join("graph_desc","plot_scatt_1stF_2stF_GrLivArea_LowQualFinSF_central-air.png"))

    ####┘ heatmap  

    cross_t = pd.crosstab(all_data["OverallCond"],all_data["OverallQual"])
    f, ax = plt.subplots(figsize=(9, 6))
    plot = sns.heatmap(cross_t, annot=True, fmt="d", linewidths=.5, ax=ax)
    fig = plot.get_figure()
    fig.savefig(os.path.join("graph_desc","plot_condi_quali.png"))




###########################################################################################################################
###########################################################################################################################

