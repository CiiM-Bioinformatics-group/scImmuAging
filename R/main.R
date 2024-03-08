#' Pre-Processing of input dataset
#'
#' input is a seurat object. meta.data should have "donor_id" and "age" columns.
#' @param seurat_object seurat object with metadata "donor_id", "age"
#' @return Return pre-processed input
#' @export

PreProcess = function(input, cell_type, model, marker_gene)
{
  new.seurat = subset(input, features = marker_gene)
  preprocessed_df = preprocessing(new.seurat) %>% group_by(donor_id, age) %>% tidyr::nest()
  preprocessed_df <- preprocessed_df %>% mutate(pseudocells = map(data, pseudocell))
  preprocessed_df$data <- NULL
  preprocessed_df <- tidyr::unnest(preprocessed_df, pseudocells)
  
  return(preprocessed_df)
}


#' Pre-Processing of input dataset
#'
#' input is a seurat object. meta.data should have "donor_id" and "age" columns.
#' @param seurat_object seurat object with metadata "donor_id", "age"
#' @return Return pre-processed input
#'
#' @export

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



#' Pre-Processing of input dataset
#'
#' input could be interested gene list. Each row is one gene and columns are "gene" and "value".
#' "value" could be log2FC or specific value.
#' colnames of other metadata should be "subtype".
#' @param input the output from preprocessing() function.
#' @param size how many cells would be randomly selected to generate pseudocells
#' @param n how many times to repeat random selection
#' @return Return ranked data.frame based on "value"
#'
#' @export

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

#' predict age for each individual
#'
#' @param preprocessed_df the output from PreProcess() function.
#' @param model the cell type aging clock model
#' @param marker_gene selected marker genes from the corresponding model
#' @return Return dataframe of predicted age
#'
#' @export

AgingClockCalculator <- function(preprocessed_df, model, marker_gene) {
  
  test_mtx = as.matrix(preprocessed_df[,-c(1:2)])
  train_genes = marker_gene
  
  shared.genes = intersect(colnames(test_mtx), train_genes)
  test_mtx = test_mtx[, shared.genes]
  miss_mtx = matrix(nrow = nrow(test_mtx), ncol = (length(train_genes) - ncol(test_mtx)), 0)
  colnames(miss_mtx) = setdiff(train_genes, colnames(test_mtx))
  
  final_mtx = cbind(test_mtx, miss_mtx)
  
  idx = match(train_genes, colnames(final_mtx))
  final_mtx = final_mtx[,idx]
  
  predict_df = pre_fun(model, marker_gene, final_mtx, preprocessed_df)
  
  return(predict_df)
  
}


#' age prediction
#'
#' @param final_mtx input to the model
#' @param model the cell type aging clock model
#' @param selected selected marker genes from the corresponding model
#' @param preprocessed_df the output from PreProcess() function.
#' @return Return dataframe of predicted age
#'
#' @export


pre_fun = function(model, selected, final_mtx, preprocessed_df)
{
  testPredictions <- predict(model, newx = as.matrix(final_mtx), s="lambda.min")
  test_df = data.frame(donor_id = preprocessed_df[, 1],
                       age = preprocessed_df[, 2],
                       Prediction = testPredictions[,1])
  
  return(test_df)
  
}


#' predicted age for each individual
#'
#' @param predict_res the output from AgingClockCalculator() function
#' @return Return predicted age for each individual
#'
#' @export

Age_Donor = function(predict_res)
{
  donor_df = predict_res %>% group_by(donor_id) %>% mutate(predicted = round(mean(Prediction))) %>% dplyr::select(-Prediction) %>% distinct()
  
  return(donor_df)
}





