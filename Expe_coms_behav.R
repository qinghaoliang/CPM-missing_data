########################################################################
##### Experiment CPM with missing connectomes and Phenotypes ###########
########################################################################
# load the data set
# all_edges_part: edges data of connectomes, should be rendered in shape (n,m*(m-1)/2*k)
# m is the number of nodes, n is the number of subject, k the number of tasks
# all_behavs_part: phenotypes, (n, k)
# sub_ic: id of subjects with missing connectome

args <- commandArgs(trailingOnly = TRUE)
source("rcpm_com_behav.R")
load("./data/HBN/sub_ic.RData")
load("./data/HBN/all_edges_part.RData")
load("./data/HBN/all_behavs_part.RData")
seed <- as.numeric(args[1])

cat("seed:", seed, "\n")
method_com <- c("mean", "task")
method_behav <- c("missForest", "imputePCA", "pmm")
eval <- c("Rcv")
result <- data.frame(matrix(data = NA, nrow = 1, ncol = 3))
rownames(result) <- method_com
colnames(result) <- method_behav

for (j in 1:3){
  result[1,j] <- rcpm_com_behav(all_edges_part, all_behavs_part, sub_ic, method_com[1], method_behav[j], seed=seed)
}

filename <- paste("../results/HBN_com_behav_",seed,".RData", sep = "")
save(result, file = filename)
