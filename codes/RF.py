from sklearn.ensemble import RandomForestRegressor
import os
import pandas as pd
import pickle

base_dir = "./AgePrediction/"
os.chdir("./AgePrediction/")

    
TrainData = pd.read_csv("./data/processed/CD4T_updateseuratdata_agegene_dftrain_matrix.txt", sep = "\t")


golden_standard = TestData.iloc[:, 0:2]

   
### model training ### 
output = TrainData["age"]
expr_data_input = TrainData.iloc[:, 2:]

treeEstimator = RandomForestRegressor(n_estimators=500, n_jobs = 1)
treeEstimator.fit(expr_data_input,output)
pickle.dump(treeEstimator, open("./data/processed/CD4T_updateseuratdata_agegene_RFmodel.sav", 'wb'))



### prediction ###
base_dir = "./AgePrediction/"
cell_type = "MONO"
treeEstimator = pickle.load(open(base_dir + "/data/processed/" + cell_type + "_updateseuratdata_agegene_RFmodel.sav", 'rb'))
TestData = pd.read_csv(base_dir + '/data/processed/' + cell_type + '_updateseuratdata_agegene_validation_matrix.txt', sep = "\t")
TestData = pd.read_csv(base_dir + '/data/processed/GOUT_' + cell_type + '_lasso_validation_matrix.txt', sep = "\t")
golden_standard = pd.read_csv(base_dir + '/data/processed/GOUT_' + cell_type + '_RealAge.txt', sep = "\t")
golden_standard = TestData.iloc[:, 0:2]

PredictRes = treeEstimator.predict(TestData.iloc[:, 1:])
golden_standard['Prediction'] = PredictRes
golden_standard.to_csv("/home/wli/test.txt", index = False, sep = "\t")


### feature importance ###

idx = (-treeEstimator.feature_importances_).argsort()
feature_df = treeEstimator.feature_importances_[idx]
feature_df = pd.DataFrame(feature_df)
feature_df.index = TestData.iloc[:, 0:].columns[idx]

feature_df['ensembl'] = feature_df.index
feature_df.to_csv("./prediction.txt", sep = "\t")
