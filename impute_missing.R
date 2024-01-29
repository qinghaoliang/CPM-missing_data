#################################################################
############## Phenotypes Imputation methods implementation #####
#################################################################
# method: 1) mean: impute missing values using mean values
#         2) missForest: impute missing values using missForest
#         3) imputePCA: impute missing values using a regularized iterative PCA
#         4) pmm: impute missing values using predictive mean matching in MICE
impute_missing <- function(data, method){
    if (method == "mean"){
        source("fillmissing.R")
        M <- colMeans(data, na.rm = TRUE)
        data <- fillmissing(data, M)
    }

    if (method == "missForest" ){
        library(missForest)
        data_forest <- missForest(data, maxiter = 10, ntree = 15)
        data <- data_forest$ximp
    }

    if (method == "imputePCA"){
        library(missMDA)
        data_pca <- imputePCA(data, ncp = 4)
        data <- data_pca$completeObs
    }

    if (method == "pmm"){
        library(mice)
        library(plyr)
        data_imp <- mice(data, method = "pmm", m = 10, printFlag = FALSE)
        data_comp <- complete(data_imp, 'all')
        data <- aaply(laply(data_comp, as.matrix), c(2,3), mean)
        colnames(data) <- NULL
    }

    data
}
