nrmse <- function(ximp, xtrue){
  sqrt(mean((ximp-xtrue)^2)/var(xtrue))
}