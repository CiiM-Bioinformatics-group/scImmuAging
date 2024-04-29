import mira
import anndata
import optuna
import os
import math
import pickle
import scanpy as sc
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import ShuffleSplit
#mira.utils.pretty_sderr()


os.chdir("~/AgePrediction/")

base_dir = "~/AgePrediction/"


cell_type = "CD4T"
filename = base_dir + "/data/processed/" + cell_type + "_newdata_allgene_rawcounts.h5ad"
data = anndata.read_h5ad(filename)


# sc.pp.filter_genes(data, min_cells=15)
data_raw = data

sc.pp.normalize_total(data, target_sum=1e4)
sc.pp.log1p(data)


sc.pp.highly_variable_genes(data, min_disp = 0.2)
data.var['exog'] = data.var.highly_variable.copy()

data.var['endog'] = data.var.exog & (data.var.dispersions_norm > 0.7)
data.layers['counts'] = data_raw.raw.X.copy()
data.layers['counts'].max()



model = mira.topics.ExpressionTopicModel(
    endogenous_key='endog',
    exogenous_key='exog',
    counts_layer='counts'
)

model.get_learning_rate_bounds(data) 




tuner = mira.topics.TopicModelTuner(model,
    save_name = base_dir + cell_type + '_allgene_3-fold',
    seed = 0,
    cv = ShuffleSplit(n_splits=3, train_size=0.8), 
    batch_sizes = [64, 128] 
)

tuner.train_test_split(data)
tuner.tune(data, n_workers=1)



best_model = tuner.select_best_model(data, record_umaps=False)
filename = base_dir + '/data/processed/' + cell_type + '_newdata_allgene_topic_model.pth'
best_model.save(filename)


best_model = mira.topic_model.ExpressionTopicModel.load(filename)

best_model.predict(data)


### select top genes from topics ###

top_gene_len = []
for i in range(len(best_model.topic_cols)):
        top_gene_len.append(best_model.get_top_genes(i, min_genes = 0).shape[0])


topic_gene = []
for i in range(len(best_model.topic_cols)):
        # print(best_model.get_top_genes(i, min_genes = top_gene_len[i]))
        topic_gene.extend(best_model.get_top_genes(i, top_n = top_gene_len[i]).tolist())
        # print(len(best_model.get_top_genes(i, top_n = top_gene_len[i])))

topic_gene = set(topic_gene)

topic_gene_df = pd.DataFrame(topic_gene, columns=['gene'])

filename = base_dir + "/data/processed/" + cell_type + "_newdata_topicmodel_gene.txt"
topic_gene_df.to_csv(filename, sep = "\t", index = False)


topic_gene = pd.DataFrame(index = range(len(topic_gene)), columns = range(best_model.num_topics))
for i in range(len(best_model.topic_cols)):
        # print(best_model.get_top_genes(i, min_genes = top_gene_len[i]))
        topic_gene.iloc[0:top_gene_len[i], i] = best_model.get_top_genes(i, top_n = top_gene_len[i])


topic_gene.to_csv(base_dir + "/data/processed/" + cell_type + "_newdata_topicmodel_genelist.txt", sep = "\t", index = False)

