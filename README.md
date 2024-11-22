# scImmuAging
### scImmuAging: the R implementation of the cell-type-specific transcriptome aging clocks for human PBMC. 

In this study, we established a robust cell-type-specific aging clock, covering monocytes, CD4+ T, CD8+ T, NK and B cells, based on single-cell transcriptomic 
profiles from 1081 PBMC samples from European healthy adults. Our research sheds light on understanding biological age alterations in response to vaccinations 
and diseases, revealing the most relevant cell type and subset of genes that play dual roles in both aging and immune responses to various stimuli.

We describe the scImmuAging in the following paper: Cell-Type-Specific Aging Clocks Unveil Inter-Individual Heterogeneity in Immune Aging and Rejuvenation during Infection and Vaccination

## Introduction to scImmuAging
We developed cell-type-specific transcriptome aging clocks for human PBMCs using scRNA-seq datasets from five studies, encompassing 1081 healthy individuals of 
European ancestry aged 18 to 97 years. Focusing on the five most prevalent cell types - CD4+ T cells, CD8+ T cells, monocytes, NK cells, and B cells- we build 
independent aging clocks for each using machine learning (LASSO and random forest) and deep learning (PointNet) methods, assessing performance through various 
metrics.

![Workflow of scImmuAging](https://github.com/wenchaoli1007/HPscAC/blob/main/data/workflow.png)

## Operating system
MacOS and Linux

## Software requirements
R 4.2.1, python 3.9.8

## Package requirements
R: Seurat 4.0, FUMA 1.5.2, ggplot2 3.4.1, readr 2.1.3, tidyverse 1.3.2, glmnet 4.1.4, ggpubr 0.4.0, biomaRt 2.52.0, infotheo 1.2.0.1, purrr 0.3.4, ggridges 0.5.4, ComplexHeatmap 2.12.1, RcisTarget 1.16.0, GENIE3 1.18.0

python: pandas 1.4.4, numpy 1.19.5,  tensorflow 2.5.3, keras 2.5.0, scipy 1.9.3, sklearn 0.23.2, scanpy 1.9.13, mira 1.0.4.

## Installation of scImmuAging
install.packages("devtools")

devtools::install_github("CiiM-Bioinformatics-group/scImmuAging")

Installation will only take few seconds.

## Extract model features and coefficients 
    model_set = readRDS(system.file("data", "all_model.RDS", package = "scImmuAging"))

    feature_set = readRDS(system.file("data", "all_model_inputfeatures.RDS", package = "scImmuAging"))

    feature_set1 = list()

    for(i in c("CD4T", "CD8T", "MONO", "NK", "B"))

    {

        temp_df = coef(model_set[[i]])
  
        temp_feature = rownames(temp_df)[which(temp_df[,1] != 0)]
  
        feature_set1[[i]] = temp_feature
  
    }

It will only take few seconds.

## Citation
Cell-Type-Specific Aging Clocks Unveil Inter-Individual Heterogeneity in Immune Aging and Rejuvenation during Infection and Vaccination


