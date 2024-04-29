library(readr)
library(dplyr)
library(Seurat)
library(tidyverse)
library(glmnet)
library(viridis)
library(ggpubr)
library(ggplot2)
library(biomaRt)
library(SeuratData)
library(SeuratDisk)
library(infotheo)

setwd("/home/wli/")

preprocessing <- function(seurat_object)
{
  DefaultAssay(seurat_object) = "RNA"
  meta_data = seurat_object@meta.data
  
  if(!length(which(colnames(meta_data) == "donor_id")))
    stop("error: please add donor_id in your metadata!")
  
  if(is.null(which(colnames(meta_data) == "age")))
    stop("error: please add age in your metadata!")
  
  meta_data <- meta_data[, c("donor_id", "age")]
  input_mtx <- t(as.matrix(seurat_object@assays$RNA@data))
  combined_input <- as_tibble(cbind(meta_data, input_mtx))
  return(combined_input)
  
}

## cite: PMID: 37118510 ##
pseudocell <- function(input, size=15, n=100, replace="dynamic") {
  pseudocells <- c()
  if (replace == "dynamic") {
    if (nrow(input) <= size) {replace <- TRUE} else {replace <- FALSE}
  }
  for (i in c(1:n)) {
    batch <- input[sample(1:nrow(input), size = size, replace = replace), ]
    pseudocells <- rbind(pseudocells, colMeans(batch))
  }
  colnames(pseudocells) <- colnames(input)
  return(as_tibble(pseudocells))
}
#########################


# MONO, NK, B, CD8T, CD4T
cell_type = "CD8T"

#### split dataset ###

train_donor = data.frame(read_tsv("./training_donor_id.txt"))$train_donor
v_donor = data.frame(read_tsv("./validation_donor_id.txt"))$ID
sc_train <- subset(scdata, donor_id %in% train_donor)

sc_train[['ProcessedData']] = NULL


#### process the training data ###
train_set = preprocessing(sc_train) %>% group_by(donor_id, age) %>% nest()
train_set <- train_set %>% mutate(pseudocell_all = map(data, pseudocell))
train_set$data <- NULL

saveRDS(train_set, 
        paste0("./", cell_type, "_newdata_allgene_train.RDS"))   ### 

train_set <- unnest(train_set, pseudocell_all)
write_tsv(train_set, 
          paste0("./", cell_type, "_newdata_allgene_train_matrix.txt"))


### calculate MI and PEARSON with rank product ###

# train_set1 = readRDS(paste0("./", cell_type, "_newdata_allgene_train.RDS"))
# train_set1 <- unnest(train_set1, pseudocell_all)
# 
# 
# 
# train_set = train_set1[, c(1,2, which(colnames(train_set1) %in% gene_var$gene[1:300]))]
### rank product for MI and PEARSON ###
mi = c()
for(i in 1:100)
{
  idx = seq(i, dim(train_set)[1], by = 100)
  data.temp = discretize(train_set[idx, ])
  
  mi.temp = apply(data.temp[, 3:ncol(data.temp)], 2, 
                  FUN = function(x) mutinformation(data.temp[, 2], x, method="emp"))
  mi = cbind(mi, mi.temp)
}

write_tsv(as.data.frame(mi), paste0("./", cell_type, "_mi_train_periteration.txt"))

pers = c()

for(i in 1:100)
{
  idx = seq(i, dim(train_set)[1], by = 100)
  data.temp = train_set[idx, -c(1,2)]
  
  cor.temp = apply(data.temp, 2, 
                   FUN = function(x) cor.test(as.vector(unlist(train_set[idx, 2])), 
                                              as.vector(unlist(x)))$estimate)
  # pval.temp = apply(data.temp, 2, 
  #                   FUN = function(x) cor.test(as.vector(unlist(train_set[idx, 2])), 
  #                                              as.vector(unlist(x)))$p.value)
  # 
  
  pers = cbind(pers, cor.temp)
}

colnames(pers) = paste0("cor", seq(1:100))
write_tsv(as.data.frame(pers) %>% mutate(gene = rownames(pers)), paste0("./", cell_type, "_pearson_train_periteration.txt"))


cell_type
mi = data.frame(read_tsv(paste0("./", cell_type, "_mi_train_periteration.txt")))
pears = data.frame(read_tsv(paste0("./", cell_type, "_pearson_train_periteration.txt")))

rownames(pears) = pears[,"gene"]
pears = pears[,-ncol(pears)]

rownames(mi) = rownames(pears)
colnames(mi) = paste0("mi", seq(1:100))


mi = -abs(mi)
pears = -abs(pears)

mi_order_index = as.data.frame(apply(mi, 2, FUN = function(x) rank(x)))
pears_order_index = as.data.frame(apply(pears, 2, FUN = function(x) rank(x)))

mi_order_index$rp = as.data.frame(unlist(apply(mi_order_index, 1, FUN = function(x) prod(x^(1/100)))))
pears_order_index$rp = as.data.frame(unlist(apply(pears_order_index, 1, FUN = function(x) prod(x^(1/100)))))

mi_rp = c()
pears_rp = c()
for(i in 1:1000)
{
  
  mi_temp = mi[sample(1:nrow(mi)), ]
  mi_order_temp = as.data.frame(apply(mi_temp, 2, FUN = function(x) rank(x)))
  mi_rp = rbind(mi_rp, 
                unlist(apply(mi_order_temp, 1, FUN = function(x) prod(x^(1/100)))))
  pears_temp = pears[sample(1:nrow(pears)), ]
  pears_order_temp = as.data.frame(apply(pears_temp, 2, FUN = function(x) rank(x)))
  pears_rp = rbind(pears_rp,
                   unlist(apply(pears_order_temp, 1, FUN = function(x) prod(x^(1/100)))))
  
  
}

pears_rp = rbind(pears_rp, t(pears_order_index$rp))
mi_rp = rbind(mi_rp, t(mi_order_index$rp))

counts = c()
for(i in 1:ncol(pears_rp))
{
  counts = c(counts, length(which(pears_rp[-nrow(pears_rp), i] < pears_rp[nrow(pears_rp), i])))
}

pears_counts = counts/1000

counts = c()
for(i in 1:ncol(mi_rp))
{
  counts = c(counts, length(which(mi_rp[-nrow(mi_rp), i] < mi_rp[nrow(mi_rp), i])))
}

mi_counts = counts/1000
pears_genes = colnames(pears_rp)[which(pears_counts < 0.05)]    
mi_genes = colnames(pears_rp)[which(mi_counts < 0.05)]

all_genes = unique(c(pears_genes, mi_genes))

# topic_genes = read_tsv(paste0("./", cell_type, "_topicmodel_gene.txt"))
# 
# length(intersect(topic_genes$gene, all_genes))


### model_training ###

feature_genes = read_tsv(paste0("./", cell_type, "_rankproduct_gene.txt"))$all_genes

## pearson: _age_gene.txt
## mira: _mira_gene.txt
## rp: rankproduct_gene.txt

selected = feature_genes
train_set = readRDS(paste0("./", cell_type, "_newdata_allgene_train.RDS"))
train_set <- unnest(train_set, pseudocell_all)

input_mtx = train_set[, selected]
write_tsv(data.frame(input_mtx), paste0("./", cell_type, "_newdata_rpgene_train_matrix.txt"))
write_tsv(train_set[,c(1,2)], paste0("./", cell_type, "_newdata_train_RealAge.txt"))


y_age = data.frame(read_tsv(paste0("./", cell_type, "_newdata_train_RealAge.txt")))

age_df = as.matrix(table(y_age$age))
age_df[,1] = age_df[,1]/100


idx = match(y_age$age, rownames(age_df))
sample.weight = 1/age_df[idx,1]


model <- cv.glmnet(x = as.matrix(input_mtx),
                   y = as.matrix(train_set[,2]),
                   type.measure="mae",
                   standardize=F,
                   relax=F,
                   weights = sample.weight,
                   nfolds = 5)
saveRDS(model, 
        paste0("./", cell_type, "_newdata_rplasso_model.RDS"))


### validation dataset ###

### 20% internal validation ###

sc_validation <- subset(scdata, donor_id %in% v_donor)
sc_validation = subset(sc_validation, features = selected)

internal_valid = preprocessing(sc_validation) %>% group_by(donor_id, age) %>% nest()
internal_valid <- internal_valid %>% mutate(pseudocell_all = map(data, pseudocell))
internal_valid$data <- NULL
internal_valid <- unnest(internal_valid, pseudocell_all)


donor_id = internal_valid[,1]
age = internal_valid[,2]

final_mtx = as.matrix(internal_valid[, -c(1,2)])

golden_standard = data.frame(donor_id = internal_valid[, 1],
                             age = internal_valid[, 2])

testPredictions <- predict(model, newx = final_mtx, s="lambda.min")
test_df = data.frame(donor_id = internal_valid[, 1],
                     age = internal_valid[, 2],
                     Prediction = testPredictions[,1])

test_df = data.frame(donor_id = donor_id,
                     age = age,
                     Prediction = testPredictions[,1])

head(test_df)
write_tsv(test_df, "./prediction.txt")


