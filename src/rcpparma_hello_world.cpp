#include <RcppParallel.h>
#include <RcppArmadillo.h>
#include <Rcpp.h>
#include <nlopt.hpp>
#include <random>

// [[Rcpp::depends(RcppArmadillo, RcppParallel)]]


using namespace Rcpp;
using namespace arma;
using namespace RcppParallel;

// =============== The DGP Function ============================================
// Function to create data.frames:
// [[Rcpp::export]]
Rcpp::DataFrame generate_panel_data(int N, int tt, double beta, double sigma) {
  int total_obs = N * tt;
  arma::vec X(total_obs);
  arma::vec Y(total_obs);
  arma::vec ID(total_obs);
  arma::vec time(total_obs);
  
  // Generate panel data
  for (int i = 0; i < N; i++) {
    for (int t = 0; t < tt; t++) {
      int index = i * tt + t; // Current row index
      
      // Generate X from standard normal distribution
      X[index] = R::rnorm(0, 1);
      
      // Generate error term from normal distribution
      double error = R::rnorm(0, sigma);
      
      // Compute latent Y*
      double Y_star = beta * X[index] + error;
      
      // Generate probit Y (binary outcome)
      Y[index] = (Y_star > 0) ? 1.0 : 0.0;
      
      // Assign ID and time
      ID[index] = i + 1;
      time[index] = t + 1;
    }
  }
  
  // Return as a DataFrame
  return Rcpp::DataFrame::create(
    Rcpp::Named("Y") = Y,
    Rcpp::Named("X") = X,
    Rcpp::Named("ID") = ID,
    Rcpp::Named("time") = time
  );
}

// =============== The Probit Estimation Functions =============================
// Probit log-likelihood function
// [[Rcpp::export]]
double probit_log_likelihood(const std::vector<double> &beta, std::vector<double> &grad, void* params) {
  // Extract parameters (X and Y)
  arma::mat* X = static_cast<arma::mat*>(static_cast<void**>(params)[0]);
  arma::vec* Y = static_cast<arma::vec*>(static_cast<void**>(params)[1]);
  
  arma::vec beta_arma = arma::vec(beta); // Convert std::vector to arma::vec
  
  // Compute linear predictor
  arma::vec linear_predictor = (*X) * beta_arma;
  
  // Compute the cumulative normal distribution (Probit model)
  arma::vec Phi = 0.5 * (1.0 + arma::erf(linear_predictor / std::sqrt(2.0)));
  Phi = arma::clamp(Phi, 1e-10, 1 - 1e-10); // Avoid log(0)
  
  // Compute the log-likelihood
  double log_likelihood = -arma::sum((*Y) % arma::log(Phi) + (1 - (*Y)) % arma::log(1 - Phi));
  
  // Compute gradient if required
  if (!grad.empty()) {
    arma::vec pdf = arma::exp(-0.5 * arma::square(linear_predictor)) / std::sqrt(2.0 * M_PI);
    arma::vec score = ((*Y - Phi) / (Phi % (1 - Phi))) % pdf;
    arma::vec gradient = -(*X).t() * score;
    
    // Fill std::vector grad
    for (size_t i = 0; i < gradient.n_elem; ++i) {
      grad[i] = gradient[i];
    }
  }
  
  return log_likelihood;
}

// [[Rcpp::export]]
Rcpp::List probit_mle(const arma::mat& X, const arma::vec& Y, int max_iter = 10000, double tol = 1e-6) {
  // Initialize optimization parameters
  size_t n_params = X.n_cols;
  std::vector<double> beta(n_params, 0.0); // Initial guess (zero vector)
  
  // Parameter container
  void* params[] = {const_cast<arma::mat*>(&X), const_cast<arma::vec*>(&Y)};
  
  // Configure nlopt optimizer
  nlopt::opt opt(nlopt::LD_LBFGS, n_params); // Use L-BFGS algorithm
  opt.set_min_objective(probit_log_likelihood, static_cast<void*>(params)); // Set objective function
  opt.set_xtol_rel(tol); // Convergence tolerance
  opt.set_maxeval(max_iter); // Maximum iterations
  
  // Result container
  double minf;
  nlopt::result result = opt.optimize(beta, minf);
  
  // Convert results to arma::vec
  arma::vec beta_final = arma::vec(beta);
  
  // Return result
  return Rcpp::List::create(
    Rcpp::Named("estimate") = beta_final,
    Rcpp::Named("log_likelihood") = -minf,
    Rcpp::Named("iterations") = opt.get_numevals(),
    Rcpp::Named("status") = result == nlopt::SUCCESS ? "Success" : "Failure" // Return status as a string
  );
}

// [[Rcpp::export]]
List binary_individual_slopes(const DataFrame& df, int max_iter = 1000, double tol = 1e-6) {
  
  // Extract necessary columns
  NumericVector Y_temp = df["Y"];
  NumericVector X_temp = df["X"];
  NumericVector ID_temp = df["ID"];
  int N = unique(ID_temp).size();
  // Convert from NumericVector to arma::vec and NumericMatrix to arma::mat
  arma::vec Y(Y_temp.begin(), Y_temp.size(), false); // Copying to arma::vec
  arma::vec X(X_temp.begin(), X_temp.size(), false); // Copying to arma::mat
  arma::vec ID(ID_temp.begin(), ID_temp.size(), false); // Copying to arma::vec
  
  // Create ID dummies using model.matrix equivalent in Rcpp
  // create a matrix with only zero's:
  // It needs to have the same number of rows as the ID vector and the number of columns as the number of unique ID's
  arma::mat X_ID = arma::zeros<arma::mat>(ID.n_elem, N);
  for (int i = 0; i < ID.n_elem; i++) {
    X_ID(i, ID(i)-1) = X(i);
  }
  
  // Remove the first column of X_ID to avoid multicollinearity
  arma::mat X_ID_new = X_ID.cols(1, X_ID.n_cols - 1);
  
  // Create interaction between X and ID dummies
  // Combine data (X, ID dummies, and X_ID)
  arma::mat full_X = join_rows(X, X_ID_new);
  
  // Run MLE using probit_log_likelihood and its gradient (you can adjust optimizer here)
  Rcpp::List mle_result = probit_mle(full_X, Y, max_iter, tol);
  
  // Return result
  return List::create(
    Named("estimate") = mle_result["estimate"],
                                  Named("log_likelihood") = mle_result["log_likelihood"],
                                                                      Named("full_X") = full_X,
                                                                      Named("Y") = Y
  );
}

// =============== The Bootstrap Functions =====================================

// [[Rcpp::export]]
DataFrame param_bootstrap_data(DataFrame df, arma::vec beta) {
  // Convert input DataFrame to Armadillo types
  NumericVector ID_temp = df["ID"];
  NumericVector time = df["time"];
  int N = unique(ID_temp).size();  // Number of unique IDs
  int tt = unique(time).size(); // Number of unique time points
  
  // Extract the data (X matrix from the full model)
  arma::vec X = Rcpp::as<arma::vec>(df["X"]);
  
  // Calculate the predicted probabilities using the fitted model
  arma::vec probs_Y = 0.5 * (1.0 + erf((X * beta) / std::sqrt(2.0)));  // Probit model probabilities
  
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
  
  // Prepare bootstrapped DataFrame
  df["Y"] = Y_star;
  
  return df;
}

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
//         DataFrame boot_sample = param_bootstrap_data(df, beta_star_vec);
//         List fe_model = binary_individual_slopes(boot_sample);
// 
//         NumericVector X_temp = boot_sample["X"];
//         arma::mat X(X_temp.begin(), X_temp.size(), 1, false);
//         NumericVector Y_temp = boot_sample["Y"];
//         arma::vec Y(Y_temp.begin(), Y_temp.size(), false);
//         List fe_model_null = probit_mle(X, Y);
// 
//         double LR_stat = 2 * (as<double>(fe_model["log_likelihood"]) - as<double>(fe_model_null["log_likelihood"]));
//         boot_stats[i] = LR_stat;
//       } catch (const std::exception& e) {
//         boot_stats[i] = NA_REAL;
//       }
//     }
//   }
// };
// 
// // [[Rcpp::export]]
// NumericVector boot_function(DataFrame df, int B, List null_model) {
//   NumericVector boot_stats(B);
//   arma::vec beta_star_vec = as<arma::vec>(null_model["estimate"]);
// 
//   BootWorker worker(df, beta_star_vec, boot_stats);
//   parallelFor(0, B, worker);
// 
//   return boot_stats;
// }

struct BootWorker : public Worker {
  const DataFrame df;
  const arma::vec beta_star_vec;
  RVector<double> boot_stats;
  
  BootWorker(const DataFrame df, const arma::vec& beta_star_vec, NumericVector boot_stats)
    : df(df), beta_star_vec(beta_star_vec), boot_stats(boot_stats) {}
  
  void operator()(std::size_t begin, std::size_t end) {
    for (std::size_t i = begin; i < end; i++) {
      try {
        // Generate bootstrap sample
        DataFrame boot_sample = param_bootstrap_data(df, beta_star_vec);
        
        // Run full and null model within each thread (ensuring thread safety)
        List fe_model = binary_individual_slopes(boot_sample);
        
        NumericVector X_temp = boot_sample["X"];
        arma::mat X(X_temp.begin(), X_temp.size(), 1, false);
        NumericVector Y_temp = boot_sample["Y"];
        arma::vec Y(Y_temp.begin(), Y_temp.size(), false);
        List fe_model_null = probit_mle(X, Y);
        
        // Compute test statistic
        double LR_stat = 2 * (as<double>(fe_model["log_likelihood"]) - as<double>(fe_model_null["log_likelihood"]));
        
        // Store result safely
        boot_stats[i] = LR_stat;
      } catch (const std::exception& e) {
        boot_stats[i] = NA_REAL;  // Assign NA if an error occurs
      }
    }
  }
};

// [[Rcpp::export]]
NumericVector boot_function(DataFrame df, int B, List null_model) {
  NumericVector boot_stats(B);  // Output vector
  arma::vec beta_star_vec = as<arma::vec>(null_model["estimate"]);  // Extract beta estimates
  
  // Use parallel processing
  BootWorker worker(df, beta_star_vec, boot_stats);
  parallelFor(0, B, worker, 1);  // Set grainsize to 1 for better load balancing
  
  return boot_stats;
}


// Calculate the quantile of the bootstrap statistics
//[[Rcpp::export]]
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



// [[Rcpp::export]]
List bootstrap_procedure(DataFrame df, int B, int max_iter = 1000, double tol = 1e-6) {
  // Calculate N:
  NumericVector ID_temp = df["ID"];
  int N = unique(ID_temp).size();
  
  
  // Fit the null model
  Rcpp::Rcout << "test " << std::endl;
  NumericVector X_temp = df["X"];
  arma::mat X(X_temp.begin(), X_temp.size(),1, false);
  NumericVector Y_temp = df["Y"];
  arma::vec Y(Y_temp.begin(), Y_temp.size(), false);
  List null_model = probit_mle(X, Y, max_iter, tol);
  Rcpp::Rcout << "test " << std::endl;
  List full_model = binary_individual_slopes(df, max_iter, tol);
  Rcpp::Rcout << "test " << std::endl;
  double LLR_stat = 2 * (as<double>(full_model["log_likelihood"]) - as<double>(null_model["log_likelihood"]));
  
  // Run the bootstrap procedure
  NumericVector boot_stats = boot_function(df, B, null_model);
  double mean_boot_stats = mean(boot_stats);
  double sd_boot_stats = sd(boot_stats);
  double normalized_boot_stats = (LLR_stat - mean_boot_stats) / sd_boot_stats;
  
  // Standard Normal quantile:
  double quantile_005 = R::qnorm(0.05, 0.0, 1.0, 1, 0);
  double quantile_095 = R::qnorm(0.95, 0.0, 1.0, 1, 0);
  double quantile_025 = R::qnorm(0.025, 0.0, 1.0, 1, 0);
  double quantile_975 = R::qnorm(0.975, 0.0, 1.0, 1, 0);
  
  // chi-squared quantile
  double chi_squared_005 = R::qchisq(0.95,N-1, true, false);
  
  // Return the results
  return List::create(
    Named("LLR_stat") = LLR_stat,
    Named("normalized_LLR_stat") = normalized_boot_stats,
    Named("chi_squared_reject") = LLR_stat > chi_squared_005,
    Named("q_5_reject") = normalized_boot_stats < quantile_005,
    Named("q_95_reject") = normalized_boot_stats > quantile_095,
    Named("q_025_975_reject") = normalized_boot_stats < quantile_025 || normalized_boot_stats > quantile_975
  );
}

// [[Rcpp::export]]
List simulation_procedure(int N, int tt, int no_sim, int B, int max_iter = 1000, double tol = 1e-6) {
  // Initialize the results
  arma::vec LLR_stats(no_sim);
  arma::vec normalized_LL_stats(no_sim);
  arma::vec chi_squared_reject(no_sim);
  arma::vec q_5_reject(no_sim);
  arma::vec q_95_reject(no_sim);
  arma::vec q_025_975_reject(no_sim);
  
  // Run the simulation procedure
  for (int i = 0; i < no_sim; i++) {
    // Generate panel data
    DataFrame df = generate_panel_data(N, tt, 1.0, 1.0);
    
    // Run the bootstrap procedure
    List results = bootstrap_procedure(df, B, max_iter, tol);
    
    // Store the results
    LLR_stats(i) = as<double>(results["LLR_stat"]);
    normalized_LL_stats(i) = as<double>(results["normalized_LLR_stat"]);
    chi_squared_reject(i) = as<bool>(results["chi_squared_reject"]);
    q_5_reject(i) = as<bool>(results["q_5_reject"]);
    q_95_reject(i) = as<bool>(results["q_95_reject"]);
    q_025_975_reject(i) = as<bool>(results["q_025_975_reject"]);
  }
  
  // Calculate the quantiles
  double quantile_025 = quantile_func(normalized_LL_stats, 0.025);
  double quantile_975 = quantile_func(normalized_LL_stats, 0.975);
  
  // Return the results
  return List::create(
    Named("LLR_stats") = LLR_stats,
    Named("normalized_LL_stats") = normalized_LL_stats,
    Named("quantile_025") = quantile_025,
    Named("quantile_975") = quantile_975
  );
}








