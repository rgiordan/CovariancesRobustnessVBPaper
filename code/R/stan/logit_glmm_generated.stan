data {
  // Data
  int <lower=0> NG;  // number of groups
  int <lower=0> N;  // total number of observations
  int <lower=0> K;  // dimensionality of parameter vector which is jointly distributed
  int <lower=0, upper=1> y[N];       // outcome variable of interest
  vector[K] x[N];  // Covariates
  int y_group[N];  // y_group is zero-indexed group indicators

  // Hyperparameters:
  matrix[K,K] beta_prior_info;
  vector[K] beta_prior_mean;
  real mu_prior_mean;
  real mu_prior_info; // <lower=0>
  real tau_prior_alpha; // <lower=0>
  real tau_prior_beta; // <lower=0>
}
parameters {
  vector[K] beta;      // Global regressors.
  real mu;             // The mean of the random effect.
  real <lower=0> tau;  // The information of the random effect.
  vector[NG] u;        // The actual random effects.
}
transformed parameters {
  real log_tau;
  log_tau = log(tau);
}
model {
  // priors
  tau ~ gamma(tau_prior_alpha, tau_prior_beta);
  beta ~ multi_normal_prec(beta_prior_mean, beta_prior_info);
  mu ~ normal(mu_prior_mean, 1.0 / sqrt(mu_prior_info));

  // The model
  for (g in 1:NG) {
    u[g] ~ normal(mu, 1 / tau);
  }

  for (n in 1:N) {
    // y_group is zero-indexed, but stan is one-indexed
    y[n] ~ bernoulli(inv_logit(x[n]' * beta + u[y_group[n] + 1]));
  }
}

