# Load data
data <- read.table("data-39.txt", header = FALSE)
x <- data$V1  # First column as x
y <- data$V2  # Second column as y

# Convert x and y to matrices for matrix operations
x_matrix <- matrix(x, ncol = 1)
y_matrix <- matrix(y, ncol = 1)

# Initialize parameters (alpha0, alpha1, beta0, beta1) as a matrix
params <- matrix(c(10, 2.1, 10, 1), ncol = 1)  # Initial guesses for alpha0, alpha1, beta0, beta1

# Define learning rate and convergence tolerance
learning_rate <- 1e-2
tolerance <- 1e-6
max_iter <- 1000

# Function to compute RSS based on current parameters
compute_rss <- function(params) {
  alpha0 <- params[1]
  alpha1 <- params[2]
  beta0 <- params[3]
  beta1 <- params[4]
  
  # Predicted values
  y_pred <- (alpha0 + alpha1 * x_matrix) / (beta0 + beta1 * x_matrix)
  
  # Calculate RSS
  rss <- sum((y_matrix - y_pred)^2)
  return(list(rss = rss, y_pred = y_pred))
}

# Iteratively update parameters to minimize RSS
for (i in 1:max_iter) {
  # Current RSS and predicted values
  result <- compute_rss(params)
  rss_current <- result$rss
  y_pred <- result$y_pred
  
  # Finite difference gradient approximation
  gradient <- matrix(0, nrow = 4, ncol = 1)
  for (j in 1:4) {
    # Perturb parameter j slightly
    params_perturbed <- params
    params_perturbed[j] <- params[j] + 1e-5
    
    # Compute perturbed RSS
    rss_perturbed <- compute_rss(params_perturbed)$rss
    
    # Approximate gradient
    gradient[j] <- (rss_perturbed - rss_current) / 1e-5
  }
  
  # Update parameters
  params <- params - learning_rate * gradient
  
  # Check convergence
  if (sqrt(sum(gradient^2)) < tolerance) {
    break
  }
}

# Display estimated parameters and final RSS
print("Estimated Parameters (alpha0, alpha1, beta0, beta1):")
print(params)

# Calculate and print final RSS
final_rss <- compute_rss(params)$rss
cat("Final Residual Sum of Squares (RSS):", final_rss, "\n")

# Calculate residuals and residual variance
residuals <- y - y_pred
residual_variance <- sum(residuals^2) / (length(y) - length(params))
print("Residual Variance:")
print(residual_variance)

# Calculate Jacobian matrix for parameter variance
jacobian <- matrix(0, nrow = length(y), ncol = length(params))
alpha0 <- params[1]
alpha1 <- params[2]
beta0 <- params[3]
beta1 <- params[4]

for (i in 1:length(y)) {
  jacobian[i, 1] <- 1 / (beta0 + beta1 * x[i])  # partial derivative w.r.t. alpha0
  jacobian[i, 2] <- x[i] / (beta0 + beta1 * x[i])  # partial derivative w.r.t. alpha1
  jacobian[i, 3] <- -(alpha0 + alpha1 * x[i]) / (beta0 + beta1 * x[i])^2  # partial derivative w.r.t. beta0
  jacobian[i, 4] <- -(alpha0 * x[i] + alpha1 * x[i]^2) / (beta0 + beta1 * x[i])^2  # partial derivative w.r.t. beta1
}

# Regularization parameter
lambda <- 1e-5

# Calculate parameter variance-covariance matrix with regularization
fisher_info_matrix <- t(jacobian) %*% jacobian
param_var_cov_matrix <- residual_variance * solve(fisher_info_matrix + lambda * diag(4))

# Extract variances of each parameter from the diagonal of the variance-covariance matrix
param_variances <- diag(param_var_cov_matrix)
print("Variances of Parameters (alpha0, alpha1, beta0, beta1):")
print(param_variances)

# Calculate confidence intervals (assuming normality of estimates)
confidence_level <- 0.95
z_value <- qnorm(1 - (1 - confidence_level) / 2)

confidence_intervals <- data.frame(
  Parameter = c("alpha0", "alpha1", "beta0", "beta1"),
  Estimate = params,
  Lower = params - z_value * sqrt(param_variances),
  Upper = params + z_value * sqrt(param_variances)
)

print("Confidence Intervals for Parameters:")
print(confidence_intervals)

# Plot the data and fitted curve
plot(x, y, main = "Nonlinear Regression Fit (Model 2)", xlab = "x", ylab = "y", pch = 19, col = "red")
lines(sort(x), y_pred[order(x)], col = "blue", lwd = 2)

# Add legend
# legend("topright", legend = c("Data", "Fitted Curve"), col = c("red", "blue"), pch = c(19, NA), lty = c(NA, 1))
# Plot residuals
plot(x, residuals, main = "Residuals of Nonlinear Regression Model", xlab = "x", ylab = "Residuals", pch = 19, col = "darkgreen")
abline(h = 0, col = "red", lwd = 2)  # Horizontal line at zero for reference

# Apply Anderson-Darling test on the 'x' column
library(nortest)
result <- ad.test(residuals)
print(result) 