###########################################################################
# CPM predict the 1st PC of the phenotypes
# Input:
# all_edges: edges of complete subjects
# all_behav: phenotype of complete subjects
# seed: random number for cross-validation data split 
# thresh: feature selection threshold, p-value
# lambda: ridge regression parameters 
#
# Output: 
# Rcv for regression. 

rcpm_pc <- function(all_edges, all_behav, 
                     threshold = 0.01, alpha = 0,
                     lambda = 0.5*10^seq(-2, 10, length.out = 100), nfold = 10,
                     seed = 123){
  require(glmnet)
  require(cvTools)
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
  ## perform ridge regression
  set.seed(seed)
  folds <- cvFolds(nrow(all_edges), K = nfold)
  
  #Perform 10 fold cross validation
  for(i_fold in 1:nfold){
    #Segement your data by fold using the which() function 
    testIndexes <- folds$subsets[folds$which == i_fold]
    test_mat <- all_edges[testIndexes, ]
    test_behav <- all_behav[testIndexes, ]
    train_mat <- all_edges[-testIndexes, ]
    train_behav <- all_behav[-testIndexes, ]

    #calculate the pc as the behavioral measure
    train.pca <- prcomp(train_behav, center = TRUE, scale. = TRUE, rank. = 1)
    train_behav <- train.pca$x
    test_behav <- predict(train.pca, test_behav)
    
    # Feature Selection
    cor_pvalue <- apply(train_mat, 2, function(x){
      cc <- cor.test(x, train_behav, method = "pearson")
      cc$p.value
    })
    
    edges_1 <- which(cor_pvalue <= threshold)
    cat("\n CV:", i_fold, "/number of selected edges:", length(edges_1), "\n")
    
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
  r_2 = mean(sqrt(r_2_fold))
}
