library(TSclust)

dataset_name <- toString(commandArgs(TRUE)[1])

cluster1 <- readRDS(paste("./", dataset_name, "/cluster_result_1", ".Rds", sep=""))
cluster2 <- readRDS(paste("./", dataset_name, "/cluster_result_2", ".Rds", sep=""))
score <- cluster.evaluation(cluster1, cluster2)

cat(score,"\n", file="./measure.txt", sep=" ")

