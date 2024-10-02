library(Seurat)
library(dplyr)
library(readr)
library(scImmuAging)
library(glmnet)
library(purrr)

setwd("~/HPscAC/data/")

model_set = readRDS(system.file("data", "all_model.RDS", package = "scImmuAging"))
feature_set = readRDS(system.file("data", "all_model_inputfeatures.RDS", package = "scImmuAging"))
input = readRDS(system.file("data", "demo_bcg_cd8t.RDS", package = "scImmuAging"))

cell_type = "CD8T"
model = model_set[[cell_type]]
marker_gene = feature_set[[cell_type]]

## preprocess the data
preprocessed_df = PreProcess(input, cell_type, model, marker_gene)


### predict the age

predict_res = AgingClockCalculator(preprocessed_df, model, marker_gene)

age_per_donor = Age_Donor(predict_res)

### visualization

# density plot #
predict_res = data.frame(read_tsv(system.file("data", "predict_res.txt", package = "HPscAC")))
predict_res = predict_res %>% filter(vaccine == "BMR")
predict_res_plot = predict_res[, c("donor_id", "age", "Prediction", "condition")]
  
ggplot(predict_res, aes(x = Prediction, fill = condition)) +
  geom_density(alpha=0.5)+
  theme_classic() +
  facet_wrap(~group, ncol = 4) +
  theme(text = element_text(size = 16)) +
  geom_vline(aes(xintercept = md, color = condition), size = 0.8) +
  xlab("Predicted Age") +
  xlim(0, 100) +
  scale_fill_manual(values = c("#b0d992", "#f4c28f", "#d2352c")) +
  scale_color_manual(values = c("#b0d992", "#f4c28f", "#d2352c"))
