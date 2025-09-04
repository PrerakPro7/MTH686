# Load the data
# install.packages("nortest")

data <- read.table("data-39.txt", header=FALSE, col.names=c("x", "y"))
n <- dim(data)[1]

# Initialize parameters
alpha0 <- 1
alpha1 <- 1
alpha2 <- 1
beta1 <- 1
beta2 <- 1
params <- c(alpha0, alpha1, alpha2, beta1, beta2)

# Define the model function
model <- function(x, params) {
  params[1] + params[2] * exp(params[4] * x) + params[3] * exp(params[5] * x)
}

# Define the Jacobian function to calculate partial derivatives
jacobian <- function(x, params) {
  alpha0 <- params[1]
  alpha1 <- params[2]
  alpha2 <- params[3]
  beta1 <- params[4]
  beta2 <- params[5]
  
  # Compute partial derivatives for each parameter
  d_alpha0 <- rep(1, length(x))
  d_alpha1 <- exp(beta1 * x)
  d_alpha2 <- exp(beta2 * x)
  d_beta1 <- alpha1 * x * exp(beta1 * x)
  d_beta2 <- alpha2 * x * exp(beta2 * x)
  
  # Combine into a Jacobian matrix
  J <- cbind(d_alpha0, d_alpha1, d_alpha2, d_beta1, d_beta2)
  return(J)
}

# Set parameters for the iterative process
tolerance <- 1e-6
max_iter <- 100
lambda <- 1e-4  # Regularization parameter for Levenberg-Marquardt
iter <- 1
converged <- FALSE

# Iterative optimization
while (iter <= max_iter && !converged) {
  # Calculate the predictions and residuals
  predictions <- model(data$x, params)
  residuals <- data$y - predictions
  
  # Calculate the Jacobian matrix
  J <- jacobian(data$x, params)
  
  # Regularized update step using Levenberg-Marquardt (J^T * J + lambda * I)^{-1} * J^T * residuals
  H <- t(J) %*% J + lambda * diag(ncol(J))  # Regularized Hessian approximation
  delta <- solve(H) %*% t(J) %*% residuals
  
  # Update parameters
  params <- params + delta
  
  # Check for convergence
  if (sum(delta^2) < tolerance) {
    converged <- TRUE
  }
  
  # Increment iteration counter
  iter <- iter + 1
}

# Calculating RSS
predictions <- model(data$x, params)
rss <- sum((data$y - predictions)^2)

# Output the final parameters
cat("Optimized parameters:\n")
print(params)
cat("Residual Sum of Squares (RSS):", rss, "\n")

# Calculate residuals and residual variance
residual_variance <- rss / (n - length(params))
cat("Residual Variance:\n", residual_variance, "\n")

# Plot the data and the fitted curve
plot(data$x, data$y, main="Non-linear Regression Fit using Osborne Algorithm with Regularization", xlab="x", ylab="y", pch=19, col="red")
lines(data$x, model(data$x, params), col="blue", lwd=2)

# Residual plot
plot(data$x, residuals, 
     main="Residual Plot",
     xlab="x", ylab="Residuals",
     pch=19, col="purple")
abline(h=0, col="red", lwd=2)

# Calculate 95% Confidence Intervals using Fisher Information Matrix
fisher_information_matrix <- solve(H)
standard_errors <- sqrt(diag(fisher_information_matrix))

alpha <- 0.05  # For 95% confidence level
z_score <- qnorm(1 - alpha / 2)

# Confidence intervals
confidence_intervals <- data.frame(
  Parameter = c("alpha0", "alpha1", "alpha2", "beta1", "beta2"),
  Estimate = params,
  Lower_Bound = params - z_score * standard_errors,
  Upper_Bound = params + z_score * standard_errors
)

cat("95% Confidence Intervals for Parameters:\n")
print(confidence_intervals)

# library(nortest)

# Apply Anderson-Darling test on the 'x' column
library(nortest)
result <- ad.test(residuals)
print(result)  # Print the result explicitly

