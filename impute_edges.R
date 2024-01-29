#################################################################
############## Edges Imputation methods implementation ##########
#################################################################
# method: 1) kNN: impute missing values using k nearest neighbor (across subjects)
#         2) mean: impute missing values using mean values
#         3) const: impute missing values using a constant value 

impute_edges <- function(data, method){
    if (method == "const"){
        source("fillmissing.R")
        c <- mean(data, na.rm = TRUE)
        M <- rep(c, ncol(data))
        data <- fillmissing(data, M)
    }

    if (method == "mean" ){
        source("fillmissing.R")
        M <- colMeans(data, na.rm = TRUE)
        data <- fillmissing(data, M)
    }

    if (method == "kNN"){
        library(VIM)
        data_kNN <- kNN(all_behavs_part, numFun=mean, useImputedDist=FALSE, imp_var = FALSE)
        data <- data.matrix(data_kNN)
    }

    data
}
