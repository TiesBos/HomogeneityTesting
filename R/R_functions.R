# The R Equivalents of the Rcpp Function:

#' @title: Generating panel data for heterogeneity testing
#' @description: Generate panel data for heterogeneity testing
#' @param N: Number of individuals
#' @param tt: Number of time periods
#' @param beta: Coefficients
#' @param sigma: Standard deviation
#' @return: A list of data
#' @export
generate_data <- function(N, tt, beta, sigma){
  return(generate_panel_data(N, tt, beta, sigma))
}

#' @title: Rcpp version for glm probit
#' @description: Rcpp version for glm probit
#' @param X: Design matrix
#' @param Y: Response variable
#' @param max_iter: Maximum number of iterations
#' @param tol: Tolerance
#' @return: A list of coefficients
#' @export
probit_glm <- function(X, Y, max_iter = 1000, tol = 1e-6){
  return(probit_mle(X, Y, max_iter, tol))
}

#' @title: Rcpp version for the heterogenous probit model
#' @description: Rcpp version for the heterogenous probit model
#' @param df: Data frame
#' @return: A list of coefficients
#' @export
hetero_probit <- function(df){
  return(binary_individual_slopes(df))
}


#' @title: Rcpp version for the heterogenous probit model
#' @description: Rcpp version for the heterogenous probit model
#' @param df: Data frame
#' @return: A list of coefficients
#' @export
quantile_function <- function(x, q){
  return(quantile_func(x, q))
}








