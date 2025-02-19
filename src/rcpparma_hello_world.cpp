#include <RcppParallel.h>
#define ARMA_WARN_LEVEL 0
#include <RcppArmadillo.h>
#include <Rcpp.h>
#include <random>
// #include <cstdlib>  // For std::ldiv, std::lldiv, etc.
// #include <cerrno>   // For errno
// #include <cstddef>  // For size_t
//#include <nlopt.hpp>  // For optimization

// [[Rcpp::depends(RcppArmadillo, RcppParallel)]]

using namespace Rcpp;
using namespace arma;
using namespace RcppParallel;

// =============== The DGP Function ============================================
// Function to create data.frames:
// [[Rcpp::export]]
arma::mat generate_panel_data(int N, int tt, double beta, double sigma) {
  int total_obs = N * tt;
  arma::mat data(total_obs, 4);
  
  // Generate panel data
  for (int i = 0; i < N; i++) {
    for (int t = 0; t < tt; t++) {
      int index = i * tt + t; // Current row index
      
      // Generate X from standard normal distribution
      data(index, 1) = R::rnorm(0, 1);
      
      // Generate error term from normal distribution
      double error = R::rnorm(0, sigma);
      
      // Compute latent Y*
      double Y_star = beta * data(index, 1) + error;
      
      // Generate probit Y (binary outcome)
      data(index, 0) = (Y_star > 0) ? 1.0 : 0.0;
      
      // Assign ID and time
      data(index, 2) = i + 1;
      data(index, 3) = t + 1;
    }
  }
  
  return data;
}

// =============== The Probit Estimation Functions =============================
// Log-likelihood and gradient for probit model
double probit_log_likelihood(const arma::vec& beta, const arma::mat& X, const arma::vec& Y) {
  arma::vec linear_predictor = X * beta;
  arma::vec Phi = 0.5 * (1.0 + arma::erf(linear_predictor / std::sqrt(2.0)));
  Phi = arma::clamp(Phi, 1e-10, 1 - 1e-10); // Avoid log(0)
  return arma::sum(Y % arma::log(Phi) + (1 - Y) % arma::log(1 - Phi));
}

arma::vec probit_log_likelihood_gradient(const arma::vec& beta, const arma::mat& X, const arma::vec& Y) {
  arma::vec linear_predictor = X * beta;
  arma::vec pdf = arma::exp(-0.5 * arma::square(linear_predictor)) / std::sqrt(2.0 * M_PI);
  arma::vec Phi = 0.5 * (1.0 + arma::erf(linear_predictor / std::sqrt(2.0)));
  Phi = arma::clamp(Phi, 1e-10, 1 - 1e-10); // Avoid division by zero
  arma::vec score = (Y - Phi) / (Phi % (1 - Phi));
  return X.t() * (pdf % score);
}

arma::mat probit_log_likelihood_hessian(const arma::vec& beta, const arma::mat& X, const arma::vec& Y) {
  arma::vec linear_predictor = X * beta;
  arma::vec pdf = arma::exp(-0.5 * arma::square(linear_predictor)) / std::sqrt(2.0 * M_PI);
  arma::vec Phi = 0.5 * (1.0 + arma::erf(linear_predictor / std::sqrt(2.0)));
  Phi = arma::clamp(Phi, 1e-10, 1 - 1e-10); // Avoid division by zero
  arma::vec W = (pdf % pdf) / (Phi % (1 - Phi)); // Diagonal of weight matrix
  return -X.t() * arma::diagmat(W) * X;
}

// Newton-Raphson method for probit MLE
// [[Rcpp::export]]
arma::vec probit_mle(const arma::mat& X, const arma::vec& Y, int max_iter = 100000, double tol = 1e-6) {
  int p = X.n_cols;
  arma::vec beta = arma::zeros(p); // Initial guess
  
  for (int iter = 0; iter < max_iter; ++iter) {
    arma::vec gradient = probit_log_likelihood_gradient(beta, X, Y);
    arma::mat hessian = probit_log_likelihood_hessian(beta, X, Y);
    
    arma::vec step = arma::solve(hessian, gradient, arma::solve_opts::fast); // Newton step
    beta -= step;
    
    // Convergence check
    if (arma::norm(step, 2) < tol) {
      break;
    }
  }
  
  double log_likelihood = probit_log_likelihood(beta, X, Y);
  
  arma::vec output(1 + beta.n_elem);
  output[0] = log_likelihood;
  output.subvec(1, beta.n_elem) = beta;
  
  return(output);
}


// // Probit log-likelihood function
// double probit_log_likelihood(const std::vector<double> &beta, std::vector<double> &grad, void* params) {
//   // Extract parameters (X and Y)
//   arma::mat* X = static_cast<arma::mat*>(static_cast<void**>(params)[0]);
//   arma::vec* Y = static_cast<arma::vec*>(static_cast<void**>(params)[1]);
//   
//   arma::vec beta_arma = arma::vec(beta); // Convert std::vector to arma::vec
//   
//   // Compute linear predictor
//   arma::vec linear_predictor = (*X) * beta_arma;
//   
//   // Compute the cumulative normal distribution (Probit model)
//   arma::vec Phi = 0.5 * (1.0 + arma::erf(linear_predictor / std::sqrt(2.0)));
//   Phi = arma::clamp(Phi, 1e-10, 1 - 1e-10); // Avoid log(0)
//   
//   // Compute the log-likelihood
//   double log_likelihood = -arma::sum((*Y) % arma::log(Phi) + (1 - (*Y)) % arma::log(1 - Phi));
//   
//   // Compute gradient if required
//   if (!grad.empty()) {
//     arma::vec pdf = arma::exp(-0.5 * arma::square(linear_predictor)) / std::sqrt(2.0 * M_PI);
//     arma::vec score = ((*Y - Phi) / (Phi % (1 - Phi))) % pdf;
//     arma::vec gradient = -(*X).t() * score;
//     
//     // Fill std::vector grad
//     for (size_t i = 0; i < gradient.n_elem; ++i) {
//       grad[i] = gradient[i];
//     }
//   }
//   
//   return log_likelihood;
// }
// 
// //[[Rcpp::export]]
// arma::vec probit_mle(const arma::mat& X, const arma::vec& Y, int max_iter = 10000, double tol = 1e-6) {
//   size_t n_params = X.n_cols;
//   std::vector<double> beta(n_params, 0.0);
//   void* params[] = {const_cast<arma::mat*>(&X), const_cast<arma::vec*>(&Y)};
//   
//   nlopt::opt opt(nlopt::LD_LBFGS, n_params);
//   opt.set_min_objective(probit_log_likelihood, static_cast<void*>(params));
//   opt.set_xtol_rel(tol);
//   opt.set_maxeval(max_iter);
//   
//   double minf;
//   nlopt::result result = opt.optimize(beta, minf);
//   
//   arma::vec beta_final = arma::vec(beta);
//   arma::vec output(1 + beta_final.n_elem);
//   output[0] = -minf;
//   output.subvec(1, beta_final.n_elem) = beta_final;
//   
//   return output;
// }

// [[Rcpp::export]]
arma::vec binary_individual_slopes(const arma::mat& data, int max_iter = 1000, double tol = 1e-6) {
  // Extract necessary columns
  arma::vec Y = data.col(0);
  arma::vec X = data.col(1);
  arma::vec ID = data.col(2);
  
  int N = arma::vec(arma::unique(ID)).n_elem;
  
  // Create ID dummies using model.matrix equivalent in Rcpp
  arma::mat X_ID = arma::zeros<arma::mat>(ID.n_elem, N);
  for (int i = 0; i < ID.n_elem; i++) {
    X_ID(i, ID(i)-1) = X(i);
  }

  // Remove the first column of X_ID to avoid multicollinearity
  arma::mat X_ID_new = X_ID.cols(1, X_ID.n_cols - 1);

  // Create interaction between X and ID dummies
  arma::mat full_X = join_rows(X, X_ID_new);

  // Run MLE using probit_log_likelihood and its gradient (you can adjust optimizer here)
  arma::vec mle_result = probit_mle(full_X, Y, max_iter, tol);
  
  // Return result
  return arma::vec(mle_result);
}

// =============== The Bootstrap Functions =====================================

// [[Rcpp::export]]
arma::mat param_bootstrap_data(const arma::mat& data, const arma::vec& beta) {
  // Extract the data (X matrix from the full model)
  arma::vec X = data.col(1);
  int N = arma::vec(arma::unique(data.col(2))).n_elem;
  int tt = arma::vec(arma::unique(data.col(3))).n_elem;
  
  // Calculate the predicted probabilities using the fitted model
  arma::vec probs_Y = 0.5 * (1.0 + arma::erf((X * beta) / std::sqrt(2.0)));  // Probit model probabilities
  
  // Random number generation for the bootstrap process
  std::random_device rd;
  std::mt19937 gen(rd());
  
  // Simulate the bootstrap Y_star values (binary outcomes)
  arma::vec Y_star(N * tt);
  for (int i = 0; i < N * tt; ++i) {
    // Create a binomial distribution with 1 trial and probability Y_prob(i)
    std::binomial_distribution<> binom(1, probs_Y(i));
    
    // Simulate the bootstrap response (0 or 1) based on the given probability
    Y_star(i) = binom(gen);  // Bootstrap response based on probability Y_prob(i)
  }
  
  // Prepare bootstrapped data
  arma::mat boot_data = data;
  boot_data.col(0) = Y_star;
  
  return boot_data;
}

// [[Rcpp::export]]
double quantile_func(arma::vec vec, double prob){
  // Sort the vector
  std::sort(vec.begin(), vec.end());
  
  // Calculate the index of the quantile
  int n = vec.size();
  double index = prob * (n - 1);
  
  // If the index is an integer, return that element
  if (index == std::floor(index)) {
    return vec[index];
  }
  
  // If the index is not an integer, interpolate between the two closest elements
  int lower_index = std::floor(index);
  int upper_index = lower_index + 1;
  
  double lower_value = vec[lower_index];
  double upper_value = vec[upper_index];
  
  return lower_value + (upper_value - lower_value) * (index - lower_index);
}

struct BootWorker : public Worker {
  const arma::mat& data;
  const arma::vec& beta_star_vec;
  arma::vec& boot_stats;
  
  BootWorker(const arma::mat& data, const arma::vec& beta_star_vec, arma::vec& boot_stats)
    : data(data), beta_star_vec(beta_star_vec), boot_stats(boot_stats) {}
  
  void operator()(std::size_t begin, std::size_t end) {
    for (std::size_t i = begin; i < end; i++) {
      try {
        // Generate bootstrap sample
        arma::mat boot_sample = param_bootstrap_data(data, beta_star_vec);
        
        // Run full and null model within each thread (ensuring thread safety)
        arma::vec fe_model = binary_individual_slopes(boot_sample);
        
        arma::vec X = boot_sample.col(1);
        arma::vec Y = boot_sample.col(0);
        arma::vec fe_model_null = probit_mle(X, Y);
        
        // Compute test statistic
        double LR_stat = 2 * (fe_model(0) - fe_model_null(0));
        
        // Store result safely
        if(LR_stat < 0){
          boot_stats[i] = NA_REAL;
        } else{
          boot_stats[i] = LR_stat;
        }
      } catch (const std::exception& e) {
        boot_stats[i] = NA_REAL;  // Assign NA if an error occurs
      }
    }
  }
};

// [[Rcpp::export]]
arma::vec boot_function(const arma::mat& data, int B, const arma::vec& null_model) {
  arma::vec boot_stats(B);  // Output vector
  arma::vec beta_star_vec = arma::vec(1).fill(null_model(1));  // Extract beta estimates
  
  // Use parallel processing
  BootWorker worker(data, beta_star_vec, boot_stats);
  parallelFor(0, B, worker, 1);  // Set grainsize to 1 for better load balancing
  
  return boot_stats;
}

// [[Rcpp::export]]
Rcpp::List bootstrap_procedure(const arma::mat& data, int B, int max_iter = 1000, double tol = 1e-6) {
  // Calculate N:
  int N = arma::vec(arma::unique(data.col(2))).n_elem;
  
  // Fit the null model
  arma::vec X = data.col(1);
  arma::vec Y = data.col(0);
  arma::vec null_model = probit_mle(X, Y, max_iter, tol);
  arma::vec full_model = binary_individual_slopes(data, max_iter, tol);
  double LLR_stat = 2 * (full_model(0) - null_model(0));
  
  if(LLR_stat < 0){
    return Rcpp::List::create(
        Rcpp::Named("LLR_stat") = LLR_stat,
        Rcpp::Named("normalized_LLR_stat") = NA_REAL,
        Rcpp::Named("chi_squared_reject") = NA_LOGICAL,
        Rcpp::Named("q_5_reject") = NA_LOGICAL,
        Rcpp::Named("q_95_reject") = NA_LOGICAL,
        Rcpp::Named("q_025_975_reject") = NA_LOGICAL,
        Rcpp::Named("boot_reject") = NA_LOGICAL
    );
  } else{
  
    // Run the bootstrap procedure
    arma::vec boot_stats_unfiltered = boot_function(data, B, null_model);
    Rcpp::Rcout << "boot_stats_unfiltered size: " << boot_stats_unfiltered.n_elem << std::endl;
    arma::uvec finite_indices = find_finite(boot_stats_unfiltered);
    Rcpp::Rcout << "Number of finite elements: " << finite_indices.n_elem << std::endl;
    arma::vec boot_stats = boot_stats_unfiltered.elem(find_finite(boot_stats_unfiltered));
    
    // check if boot_stats is empty:
    if(boot_stats.n_elem == 0){
      return Rcpp::List::create(
        Rcpp::Named("LLR_stat") = LLR_stat,
        Rcpp::Named("normalized_LLR_stat") = NA_REAL,
        Rcpp::Named("chi_squared_reject") = NA_LOGICAL,
        Rcpp::Named("q_5_reject") = NA_LOGICAL,
        Rcpp::Named("q_95_reject") = NA_LOGICAL,
        Rcpp::Named("q_025_975_reject") = NA_LOGICAL,
        Rcpp::Named("boot_reject") = NA_LOGICAL
      );
    }
    
    double mean_boot_stats = arma::mean(boot_stats);
    double sd_boot_stats = arma::stddev(boot_stats);
    double normalized_boot_stats = (LLR_stat - mean_boot_stats) / sd_boot_stats;
    
    // Standard Normal quantile:
    double quantile_005 = R::qnorm(0.05, 0.0, 1.0, 1, 0);
    double quantile_095 = R::qnorm(0.95, 0.0, 1.0, 1, 0);
    double quantile_025 = R::qnorm(0.025, 0.0, 1.0, 1, 0);
    double quantile_975 = R::qnorm(0.975, 0.0, 1.0, 1, 0);
    
    // chi-squared quantile
    double chi_squared_005 = R::qchisq(0.95,N-1, true, false);
    
    // bootstrap quantile:
    double boot_quantile = quantile_func(boot_stats, 0.95);
    
    // Return the results
    return Rcpp::List::create(
      Rcpp::Named("LLR_stat") = LLR_stat,
      Rcpp::Named("normalized_LLR_stat") = normalized_boot_stats,
      Rcpp::Named("chi_squared_reject") = LLR_stat > chi_squared_005,
      Rcpp::Named("q_5_reject") = normalized_boot_stats < quantile_005,
      Rcpp::Named("q_95_reject") = normalized_boot_stats > quantile_095,
      Rcpp::Named("q_025_975_reject") = normalized_boot_stats < quantile_025 || normalized_boot_stats > quantile_975,
      Rcpp::Named("boot_reject") = LLR_stat > boot_quantile
);
  }
}

// [[Rcpp::export]]
Rcpp::List simulation_procedure(int N, int tt, int no_sim, int B, int max_iter = 1000, double tol = 1e-6) {
  
  Rcpp::Environment base = Rcpp::Environment::base_env();
  Rcpp::Function suppressWarnings = base["suppressWarnings"];
  
  Rcpp::List sim_results(no_sim);
  // Store the results:
  for(int i = 0; i < no_sim; i++) {
    try {
      // Generate panel data
      arma::mat data = generate_panel_data(N, tt, 1.0, 1.0);
      std::ofstream null_stream("/dev/null");
      std::streambuf* old_cerr = std::cerr.rdbuf(null_stream.rdbuf());
      // Run the bootstrap procedure
      //suppressWarnings(Rcpp::wrap(your_expression));
      Rcpp::List results = suppressWarnings(Rcpp::wrap(bootstrap_procedure(data, B, max_iter, tol)));
      // Store results
      sim_results[i] = results;
      // Update print statement:
      std::cerr.rdbuf(old_cerr);
      Rcpp::Rcout << "Simulation " << i + 1 << " completed." << std::endl;
    } catch (const std::exception& e) {
      Rcpp::Rcout << "Simulation " << i + 1 << " failed." << std::endl;
      sim_results[i] = NA_REAL;
    }
    
  }
  
  return(sim_results);
}












// #include <RcppParallel.h>
// #include <RcppArmadillo.h>
// #include <Rcpp.h>
// #include <random>
// 
// // [[Rcpp::depends(RcppArmadillo, RcppParallel)]]
// 
// 
// using namespace Rcpp;
// using namespace arma;
// using namespace RcppParallel;
// 
// // =============== The DGP Function ============================================
// // Function to create data.frames:
// // [[Rcpp::export]]
// Rcpp::DataFrame generate_panel_data(int N, int tt, double beta, double sigma) {
//   int total_obs = N * tt;
//   arma::vec X(total_obs);
//   arma::vec Y(total_obs);
//   arma::vec ID(total_obs);
//   arma::vec time(total_obs);
//   
//   // Generate panel data
//   for (int i = 0; i < N; i++) {
//     for (int t = 0; t < tt; t++) {
//       int index = i * tt + t; // Current row index
//       
//       // Generate X from standard normal distribution
//       X[index] = R::rnorm(0, 1);
//       
//       // Generate error term from normal distribution
//       double error = R::rnorm(0, sigma);
//       
//       // Compute latent Y*
//       double Y_star = beta * X[index] + error;
//       
//       // Generate probit Y (binary outcome)
//       Y[index] = (Y_star > 0) ? 1.0 : 0.0;
//       
//       // Assign ID and time
//       ID[index] = i + 1;
//       time[index] = t + 1;
//     }
//   }
//   
//   // Return as a DataFrame
//   return Rcpp::DataFrame::create(
//     Rcpp::Named("Y") = Y,
//     Rcpp::Named("X") = X,
//     Rcpp::Named("ID") = ID,
//     Rcpp::Named("time") = time
//   );
// }
// 
// // =============== The Probit Estimation Functions =============================
// 
// // Log-likelihood and gradient for probit model
// double probit_log_likelihood(const arma::vec& beta, const arma::mat& X, const arma::vec& Y) {
//   arma::vec linear_predictor = X * beta;
//   arma::vec Phi = 0.5 * (1.0 + arma::erf(linear_predictor / std::sqrt(2.0)));
//   Phi = arma::clamp(Phi, 1e-10, 1 - 1e-10); // Avoid log(0)
//   return arma::sum(Y % arma::log(Phi) + (1 - Y) % arma::log(1 - Phi));
// }
// 
// arma::vec probit_log_likelihood_gradient(const arma::vec& beta, const arma::mat& X, const arma::vec& Y) {
//   arma::vec linear_predictor = X * beta;
//   arma::vec pdf = arma::exp(-0.5 * arma::square(linear_predictor)) / std::sqrt(2.0 * M_PI);
//   arma::vec Phi = 0.5 * (1.0 + arma::erf(linear_predictor / std::sqrt(2.0)));
//   Phi = arma::clamp(Phi, 1e-10, 1 - 1e-10); // Avoid division by zero
//   arma::vec score = (Y - Phi) / (Phi % (1 - Phi));
//   return X.t() * (pdf % score);
// }
// 
// arma::mat probit_log_likelihood_hessian(const arma::vec& beta, const arma::mat& X, const arma::vec& Y) {
//   arma::vec linear_predictor = X * beta;
//   arma::vec pdf = arma::exp(-0.5 * arma::square(linear_predictor)) / std::sqrt(2.0 * M_PI);
//   arma::vec Phi = 0.5 * (1.0 + arma::erf(linear_predictor / std::sqrt(2.0)));
//   Phi = arma::clamp(Phi, 1e-10, 1 - 1e-10); // Avoid division by zero
//   arma::vec W = (pdf % pdf) / (Phi % (1 - Phi)); // Diagonal of weight matrix
//   return -X.t() * arma::diagmat(W) * X;
// }
// 
// // Newton-Raphson method for probit MLE
// // [[Rcpp::export]]
// Rcpp::List probit_mle(const arma::mat& X, const arma::vec& Y,  int max_iter = 100000, double tol = 1e-6) {
//   int p = X.n_cols;
//   arma::vec beta = arma::zeros(p); // Initial guess
//   
//   for (int iter = 0; iter < max_iter; ++iter) {
//     arma::vec gradient = probit_log_likelihood_gradient(beta, X, Y);
//     arma::mat hessian = probit_log_likelihood_hessian(beta, X, Y);
//     
//     arma::vec step = arma::solve(hessian, gradient, arma::solve_opts::fast); // Newton step
//     beta -= step;
//     
//     // Convergence check
//     if (arma::norm(step, 2) < tol) {
//       break;
//     }
//   }
//   
//   double log_likelihood = probit_log_likelihood(beta, X, Y);
//   
//   return Rcpp::List::create(
//     Rcpp::Named("estimate") = beta,
//     Rcpp::Named("log_likelihood") = log_likelihood
//   );
// }
// 
// 
// 
// 
// 
// // // Probit log-likelihood function
// // // [[Rcpp::export]]
// // double probit_log_likelihood(const std::vector<double> &beta, std::vector<double> &grad, void* params) {
// //   // Extract parameters (X and Y)
// //   arma::mat* X = static_cast<arma::mat*>(static_cast<void**>(params)[0]);
// //   arma::vec* Y = static_cast<arma::vec*>(static_cast<void**>(params)[1]);
// //   
// //   arma::vec beta_arma = arma::vec(beta); // Convert std::vector to arma::vec
// //   
// //   // Compute linear predictor
// //   arma::vec linear_predictor = (*X) * beta_arma;
// //   
// //   // Compute the cumulative normal distribution (Probit model)
// //   arma::vec Phi = 0.5 * (1.0 + arma::erf(linear_predictor / std::sqrt(2.0)));
// //   Phi = arma::clamp(Phi, 1e-10, 1 - 1e-10); // Avoid log(0)
// //   
// //   // Compute the log-likelihood
// //   double log_likelihood = -arma::sum((*Y) % arma::log(Phi) + (1 - (*Y)) % arma::log(1 - Phi));
// //   
// //   // Compute gradient if required
// //   if (!grad.empty()) {
// //     arma::vec pdf = arma::exp(-0.5 * arma::square(linear_predictor)) / std::sqrt(2.0 * M_PI);
// //     arma::vec score = ((*Y - Phi) / (Phi % (1 - Phi))) % pdf;
// //     arma::vec gradient = -(*X).t() * score;
// //     
// //     // Fill std::vector grad
// //     for (size_t i = 0; i < gradient.n_elem; ++i) {
// //       grad[i] = gradient[i];
// //     }
// //   }
// //   
// //   return log_likelihood;
// // }
// // 
// // // [[Rcpp::export]]
// // Rcpp::List probit_mle(const arma::mat& X, const arma::vec& Y, int max_iter = 10000, double tol = 1e-6) {
// //   // Initialize optimization parameters
// //   size_t n_params = X.n_cols;
// //   std::vector<double> beta(n_params, 0.0); // Initial guess (zero vector)
// //   
// //   // Parameter container
// //   void* params[] = {const_cast<arma::mat*>(&X), const_cast<arma::vec*>(&Y)};
// //   
// //   // Configure nlopt optimizer
// //   nlopt::opt opt(nlopt::LD_LBFGS, n_params); // Use L-BFGS algorithm
// //   opt.set_min_objective(probit_log_likelihood, static_cast<void*>(params)); // Set objective function
// //   opt.set_xtol_rel(tol); // Convergence tolerance
// //   opt.set_maxeval(max_iter); // Maximum iterations
// //   
// //   // Result container
// //   double minf;
// //   nlopt::result result = opt.optimize(beta, minf);
// //   
// //   // Convert results to arma::vec
// //   arma::vec beta_final = arma::vec(beta);
// //   
// //   // Return result
// //   return Rcpp::List::create(
// //     Rcpp::Named("estimate") = beta_final,
// //     Rcpp::Named("log_likelihood") = -minf,
// //     Rcpp::Named("iterations") = opt.get_numevals(),
// //     Rcpp::Named("status") = result == nlopt::SUCCESS ? "Success" : "Failure" // Return status as a string
// //   );
// // }
// 
// // [[Rcpp::export]]
// List binary_individual_slopes(const DataFrame& df, int max_iter = 1000, double tol = 1e-6) {
//   
//   // Extract necessary columns
//   NumericVector Y_temp = df["Y"];
//   NumericVector X_temp = df["X"];
//   NumericVector ID_temp = df["ID"];
//   int N = unique(ID_temp).size();
//   // Convert from NumericVector to arma::vec and NumericMatrix to arma::mat
//   arma::vec Y(Y_temp.begin(), Y_temp.size(), false); // Copying to arma::vec
//   arma::vec X(X_temp.begin(), X_temp.size(), false); // Copying to arma::mat
//   arma::vec ID(ID_temp.begin(), ID_temp.size(), false); // Copying to arma::vec
//   
//   // Create ID dummies using model.matrix equivalent in Rcpp
//   // create a matrix with only zero's:
//   // It needs to have the same number of rows as the ID vector and the number of columns as the number of unique ID's
//   arma::mat X_ID = arma::zeros<arma::mat>(ID.n_elem, N);
//   for (int i = 0; i < ID.n_elem; i++) {
//     X_ID(i, ID(i)-1) = X(i);
//   }
//   
//   // Remove the first column of X_ID to avoid multicollinearity
//   arma::mat X_ID_new = X_ID.cols(1, X_ID.n_cols - 1);
//   
//   // Create interaction between X and ID dummies
//   // Combine data (X, ID dummies, and X_ID)
//   arma::mat full_X = join_rows(X, X_ID_new);
//   
//   // Run MLE using probit_log_likelihood and its gradient (you can adjust optimizer here)
//   Rcpp::List mle_result = probit_mle(full_X, Y, max_iter, tol);
//   
//   // Return result
//   return List::create(
//     Named("estimate") = mle_result["estimate"],
//                                   Named("log_likelihood") = mle_result["log_likelihood"],
//                                                                       Named("full_X") = full_X,
//                                                                       Named("Y") = Y
//   );
// }
// 
// // =============== The Bootstrap Functions =====================================
// 
// // [[Rcpp::export]]
// DataFrame param_bootstrap_data(DataFrame df, arma::vec beta) {
//   // Convert input DataFrame to Armadillo types
//   NumericVector ID_temp = df["ID"];
//   NumericVector time = df["time"];
//   int N = unique(ID_temp).size();  // Number of unique IDs
//   int tt = unique(time).size(); // Number of unique time points
//   
//   // Extract the data (X matrix from the full model)
//   arma::vec X = Rcpp::as<arma::vec>(df["X"]);
//   
//   // Calculate the predicted probabilities using the fitted model
//   arma::vec probs_Y = 0.5 * (1.0 + erf((X * beta) / std::sqrt(2.0)));  // Probit model probabilities
//   
//   // Random number generation for the bootstrap process
//   std::random_device rd;
//   std::mt19937 gen(rd());
//   
//   // Simulate the bootstrap Y_star values (binary outcomes)
//   arma::vec Y_star(N * tt);
//   for (int i = 0; i < N * tt; ++i) {
//     // Create a binomial distribution with 1 trial and probability Y_prob(i)
//     std::binomial_distribution<> binom(1, probs_Y(i));
//     
//     // Simulate the bootstrap response (0 or 1) based on the given probability
//     Y_star(i) = binom(gen);  // Bootstrap response based on probability Y_prob(i)
//   }
//   
//   // Prepare bootstrapped DataFrame
//   df["Y"] = Y_star;
//   
//   return df;
// }
// 
// // struct BootWorker : public Worker {
// //   const DataFrame df;
// //   const arma::vec beta_star_vec;
// //   RVector<double> boot_stats;
// // 
// //   BootWorker(const DataFrame df, const arma::vec& beta_star_vec, NumericVector boot_stats)
// //     : df(df), beta_star_vec(beta_star_vec), boot_stats(boot_stats) {}
// // 
// //   void operator()(std::size_t begin, std::size_t end) {
// //     for (std::size_t i = begin; i < end; i++) {
// //       try {
// //         DataFrame boot_sample = param_bootstrap_data(df, beta_star_vec);
// //         List fe_model = binary_individual_slopes(boot_sample);
// // 
// //         NumericVector X_temp = boot_sample["X"];
// //         arma::mat X(X_temp.begin(), X_temp.size(), 1, false);
// //         NumericVector Y_temp = boot_sample["Y"];
// //         arma::vec Y(Y_temp.begin(), Y_temp.size(), false);
// //         List fe_model_null = probit_mle(X, Y);
// // 
// //         double LR_stat = 2 * (as<double>(fe_model["log_likelihood"]) - as<double>(fe_model_null["log_likelihood"]));
// //         boot_stats[i] = LR_stat;
// //       } catch (const std::exception& e) {
// //         boot_stats[i] = NA_REAL;
// //       }
// //     }
// //   }
// // };
// // 
// // // [[Rcpp::export]]
// // NumericVector boot_function(DataFrame df, int B, List null_model) {
// //   NumericVector boot_stats(B);
// //   arma::vec beta_star_vec = as<arma::vec>(null_model["estimate"]);
// // 
// //   BootWorker worker(df, beta_star_vec, boot_stats);
// //   parallelFor(0, B, worker);
// // 
// //   return boot_stats;
// // }
// 
// struct BootWorker : public Worker {
//   const DataFrame df;
//   const arma::vec beta_star_vec;
//   RVector<double> boot_stats;
//   
//   BootWorker(const DataFrame df, const arma::vec& beta_star_vec, NumericVector boot_stats)
//     : df(df), beta_star_vec(beta_star_vec), boot_stats(boot_stats) {}
//   
//   void operator()(std::size_t begin, std::size_t end) {
//     for (std::size_t i = begin; i < end; i++) {
//       try {
//         // Generate bootstrap sample
//         DataFrame boot_sample = param_bootstrap_data(df, beta_star_vec);
//         
//         // Run full and null model within each thread (ensuring thread safety)
//         List fe_model = binary_individual_slopes(boot_sample);
//         
//         NumericVector X_temp = boot_sample["X"];
//         arma::mat X(X_temp.begin(), X_temp.size(), 1, false);
//         NumericVector Y_temp = boot_sample["Y"];
//         arma::vec Y(Y_temp.begin(), Y_temp.size(), false);
//         List fe_model_null = probit_mle(X, Y);
//         
//         // Compute test statistic
//         double LR_stat = 2 * (as<double>(fe_model["log_likelihood"]) - as<double>(fe_model_null["log_likelihood"]));
//         
//         // Store result safely
//         boot_stats[i] = LR_stat;
//       } catch (const std::exception& e) {
//         boot_stats[i] = NA_REAL;  // Assign NA if an error occurs
//       }
//     }
//   }
// };
// 
// // [[Rcpp::export]]
// NumericVector boot_function(DataFrame df, int B, List null_model) {
//   NumericVector boot_stats(B);  // Output vector
//   arma::vec beta_star_vec = as<arma::vec>(null_model["estimate"]);  // Extract beta estimates
//   
//   for(int i = 0; i < B; i++) {
//     // Generate bootstrap sample
//     DataFrame boot_sample = param_bootstrap_data(df, beta_star_vec);
//     
//     // Run full and null model
//     List fe_model = binary_individual_slopes(boot_sample);
//     
//     NumericVector X_temp = boot_sample["X"];
//     arma::mat X(X_temp.begin(), X_temp.size(), 1, false);
//     NumericVector Y_temp = boot_sample["Y"];
//     arma::vec Y(Y_temp.begin(), Y_temp.size(), false);
//     List fe_model_null = probit_mle(X, Y);
//     
//     // Compute test statistic
//     double LR_stat = 2 * (as<double>(fe_model["log_likelihood"]) - as<double>(fe_model_null["log_likelihood"]));
//     
//     if(LR_stat < 0){
//       LR_stat = NA_REAL;
//     }
//     
//     // Store result
//     boot_stats[i] = LR_stat;
//   }
//   
//   return boot_stats;
// }
// 
// 
// // Calculate the quantile of the bootstrap statistics
// //[[Rcpp::export]]
// double quantile_func(arma::vec vec, double prob){
//   // Sort the vector
//   std::sort(vec.begin(), vec.end());
//   
//   // Calculate the index of the quantile
//   int n = vec.size();
//   double index = prob * (n - 1);
//   
//   // If the index is an integer, return that element
//   if (index == std::floor(index)) {
//     return vec[index];
//   }
//   
//   // If the index is not an integer, interpolate between the two closest elements
//   int lower_index = std::floor(index);
//   int upper_index = lower_index + 1;
//   
//   double lower_value = vec[lower_index];
//   double upper_value = vec[upper_index];
//   
//   return lower_value + (upper_value - lower_value) * (index - lower_index);
// }
// 
// 
// 
// // [[Rcpp::export]]
// List bootstrap_procedure(DataFrame df, int B, int max_iter = 1000, double tol = 1e-6) {
//   // Calculate N:
//   NumericVector ID_temp = df["ID"];
//   int N = unique(ID_temp).size();
//   
//   
//   // Fit the null model
//   NumericVector X_temp = df["X"];
//   arma::mat X(X_temp.begin(), X_temp.size(),1, false);
//   NumericVector Y_temp = df["Y"];
//   arma::vec Y(Y_temp.begin(), Y_temp.size(), false);
//   List null_model = probit_mle(X, Y, max_iter, tol);
//   List full_model = binary_individual_slopes(df, max_iter, tol);
//   double LLR_stat = 2 * (as<double>(full_model["log_likelihood"]) - as<double>(null_model["log_likelihood"]));
//   if(LLR_stat<0){
//     LLR_stat = NA_REAL;
//   }
//   // Run the bootstrap procedure
//   NumericVector boot_stats = boot_function(df, B, null_model);
//   boot_stats = boot_stats[!is_na(boot_stats)];
//   double mean_boot_stats = mean(boot_stats);
//   double sd_boot_stats = sd(boot_stats);
//   double normalized_boot_stats = (LLR_stat - mean_boot_stats) / sd_boot_stats;
//   
//   // Standard Normal quantile:
//   double quantile_005 = R::qnorm(0.05, 0.0, 1.0, 1, 0);
//   double quantile_095 = R::qnorm(0.95, 0.0, 1.0, 1, 0);
//   double quantile_025 = R::qnorm(0.025, 0.0, 1.0, 1, 0);
//   double quantile_975 = R::qnorm(0.975, 0.0, 1.0, 1, 0);
//   
//   // chi-squared quantile
//   double chi_squared_005 = R::qchisq(0.95,N-1, true, false);
//   
//   // bootstrap quantile:
//   double boot_quantile = quantile_func(boot_stats, 0.95);
//   
//   // Return the results
//   return List::create(
//     Named("LLR_stat") = LLR_stat,
//     Named("boot_stats") = boot_stats,
//     Named("boot_reject") = LLR_stat > boot_quantile,
//     Named("normalized_LLR_stat") = normalized_boot_stats,
//     Named("chi_squared_reject") = LLR_stat > chi_squared_005,
//     Named("q_5_reject") = normalized_boot_stats < quantile_005,
//     Named("q_95_reject") = normalized_boot_stats > quantile_095,
//     Named("q_025_975_reject") = normalized_boot_stats < quantile_025 || normalized_boot_stats > quantile_975
//   );
// }
// 
// // [[Rcpp::export]]
// List simulation_procedure(int N, int tt, int no_sim, int B, int max_iter = 1000, double tol = 1e-6) {
//   // Initialize the results
//   arma::vec LLR_stats(no_sim);
//   arma::vec normalized_LL_stats(no_sim);
//   arma::vec chi_squared_reject(no_sim);
//   arma::vec q_5_reject(no_sim);
//   arma::vec q_95_reject(no_sim);
//   arma::vec q_025_975_reject(no_sim);
//   
//   // Run the simulation procedure
//   for (int i = 0; i < no_sim; i++) {
//     // Generate panel data
//     DataFrame df = generate_panel_data(N, tt, 1.0, 1.0);
//     
//     // Run the bootstrap procedure
//     List results = bootstrap_procedure(df, B, max_iter, tol);
//     
//     // Store the results
//     LLR_stats(i) = as<double>(results["LLR_stat"]);
//     normalized_LL_stats(i) = as<double>(results["normalized_LLR_stat"]);
//     chi_squared_reject(i) = as<bool>(results["chi_squared_reject"]);
//     q_5_reject(i) = as<bool>(results["q_5_reject"]);
//     q_95_reject(i) = as<bool>(results["q_95_reject"]);
//     q_025_975_reject(i) = as<bool>(results["q_025_975_reject"]);
//   }
//   
//   // Calculate the quantiles
//   double quantile_025 = quantile_func(normalized_LL_stats, 0.025);
//   double quantile_975 = quantile_func(normalized_LL_stats, 0.975);
//   
//   // Return the results
//   return List::create(
//     Named("LLR_stats") = LLR_stats,
//     Named("normalized_LL_stats") = normalized_LL_stats,
//     Named("quantile_025") = quantile_025,
//     Named("quantile_975") = quantile_975
//   );
// }
// 
// 
// 
// 
// 
// 
// 
// 
