#################################################################
# CPM impute data at connectomes and phenotypes level
# Input:
# all_edges: edges of complete subjects
# all_behav: phenotype of complete subjects
# sub_ic: id of subject with missing connectome
# seed: random number for cross-validation data split
# method_com: 1) kNN: impute missing values using k nearest neighbor (across subjects)
#             2) mean: impute missing values using mean values
#             3) const: impute missing values using a constant value
# method_behav: 1) mean: impute missing values using mean values
#               2) missForest: impute missing values using missForest
#               3) imputePCA: impute missing values using a regularized iterative PCA
#               4) pmm: impute missing values using predictive mean matching in MICE  
# thresh: feature selection threshold, p-value
# lambda: ridge regression parameters 
#
# Output: 
# Rcv for regression. 

rcpm_com_behav <- function(all_edges, all_behav, sub_ic, method_com, method_behav,
                        threshold = 0.01, alpha = 0, 
                        lambda = 0.5*10^seq(-4, 5, length.out = 20), nfold = 10,
                        seed = 123){
  require(glmnet)
  require(cvTools)
  source("impute_missing.R")
  source("impute_edges.R")
  source("nrmse.R")
  num_sub_total = dim(all_edges)[1]
  num_edges = dim(all_edges)[2]
  
  coef_total = matrix(nrow = num_edges, 
                      ncol = nfold) # store all the coefficients
  coef0_total = c()
  lambda_total = c() # store all the lambda
  r_pearson_fold = c()
  y.predict <- c()
  mse_fold <- c()
  r_2_fold <- c()
  
  #Perform 10 fold cross validation
  behav <- all_behav
  missing_behav <- rowSums(is.na(behav)) + sub_ic
  label_missing <- (missing_behav > 0)
  Index <- 1:nrow(all_edges)
  missIndex <- Index[label_missing]
  compIndex <- Index[!label_missing]
  
  edge_comp  <- all_edges[compIndex,]
  behav_comp <- behav[compIndex,]
  edge_miss  <- all_edges[missIndex,]
  behav_miss <- behav[missIndex,]
  
  set.seed(seed)
  folds <- cvFolds(length(compIndex), K = nfold)
  
  for(i_fold in 1:nfold){
    testIndexes <- folds$subsets[folds$which == i_fold]
    test_mat <- edge_comp[testIndexes, ]
    test_behav <- behav_comp[testIndexes, ]
    train_mat <- rbind(edge_comp[-testIndexes, ], edge_miss)
    train_behav <- rbind(behav_comp[-testIndexes,], behav_miss)

    # data imputation for missing edges
    train_mat <- impute_edges(train_mat, method_com)

    # data imputation for the missing behavioral measures
    train_behav <- impute_missing(train_behav, method_behav)
    train.pca <- prcomp(train_behav, center = TRUE, 
                               scale. = TRUE, rank. = 1)
    train_behav <- train.pca$x
    test_behav <- predict(train.pca, test_behav)
    
    # Feature Selection using complete data
    n_comp <- nrow(edge_comp[-testIndexes, ])
    cor_pvalue <- apply(train_mat[1:n_comp,], 2, function(x){
      cc <- cor.test(x, train_behav[1:n_comp], method = "pearson")
      cc$p.value
    })
    edges_1 <- which(cor_pvalue <= threshold)
    #cat("\n CV:", i_fold, "/number of selected edges:", length(edges_1), "\n")
    
    #Use the train data to decide best lambda for ridge regression
    model.cv <- cv.glmnet(x = train_mat[, edges_1], y = train_behav, 
                          family = "gaussian", alpha = alpha, nfolds = 10)
    idxLambda1SE = model.cv$lambda.1se
    coef_total[edges_1, i_fold] <- as.vector(model.cv$glmnet.fit$
                                beta[, which(model.cv$lambda == idxLambda1SE)])
    coef0_total[i_fold] <- model.cv$glmnet.fit$
                           a0[which(model.cv$lambda == idxLambda1SE)]
    lambda_total[i_fold] <- idxLambda1SE
    y.predict[testIndexes] <- test_mat[, edges_1] %*% 
                       matrix(coef_total[edges_1, i_fold]) + coef0_total[i_fold]
    r_pearson_fold[i_fold] <- cor(y.predict[testIndexes], test_behav)
    mse_fold[i_fold] <- mean((y.predict[testIndexes] - test_behav)^2)
    r_2_fold[i_fold] <- 1 - mse_fold[i_fold]/var(test_behav)
  }
  #browser()
  rm(all_edges)
  r_2_fold[r_2_fold < 0] = 0
  r_2 <- mean(sqrt(r_2_fold))
  cat("seed:", seed, "/r:", r_2, "\n")
  return(r_2)
}
