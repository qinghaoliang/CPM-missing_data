# fill in the missing entries with values
fillmissing <- function(xmis, value){
  for (i in 1:ncol(xmis)){
    xmis[is.na(xmis[,i]), i] <- value[i] 
  }
  xmis
}