# Load data
data <- read.table("data-39.txt", header = FALSE)
x <- data$V1  # First column as x
y <- data$V2  # Second column as y

# Create the design matrix with powers of x
X <- cbind(1, x, x^2, x^3, x^4)

# Convert y to a column vector
y_matrix <- matrix(y, ncol = 1)

# Solve for the coefficients using matrix algebra
beta <- solve(t(X) %*% X) %*% t(X) %*% y_matrix

# Calculate the predicted values
y_pred <- X %*% beta

# Calculate the Residual Sum of Squares (RSS)
rss <- sum((y_matrix - y_pred)^2)

# Display the coefficients and RSS
print("Coefficients (beta):")
print(beta)

cat("Residual Sum of Squares (RSS):", rss, "\n")

# Calculate residuals and residual variance
residuals <- y - y_pred
residual_variance <- sum(residuals^2) / (length(y) - length(params))
print("Residual Variance:")
print(residual_variance)

# Plot the data and fitted curve
plot(x, y, main = "Polynomial Regression Fit(Model_3)", xlab = "x", ylab = "y", pch = 19, col = "red")
lines(sort(x), y_pred[order(x)], col = "blue", lwd = 2)
# legend("topright", legend = c("Data", "Fitted Curve"), col = c("blue", "red"), pch = c(19, NA), lty = c(NA, 1))
# Calculate predictions using the final optimized parameters
predictions <- model(data$x, params)

# Check if lengths of data$y and predictions match
# if (length(data$y) != length(predictions)) {
#   stop("Error: Lengths of observed and predicted values do not match.")
# }
# Calculate the Fisher information matrix
fisher_info <- t(X) %*% X

# Calculate the variance-covariance matrix of the coefficients
cov_matrix <- residual_variance * solve(fisher_info)

# Set confidence level (e.g., 95%)
confidence_level <- 0.95
z_value <- qnorm((1 + confidence_level) / 2)

# Calculate confidence intervals for each coefficient
ci_lower <- beta - z_value * sqrt(diag(cov_matrix))
ci_upper <- beta + z_value * sqrt(diag(cov_matrix))

# Display the confidence intervals
print("Confidence Intervals for each coefficient (beta):")
for (i in 1:length(beta)) {
  cat("Coefficient beta", i - 1, ": [", ci_lower[i], ", ", ci_upper[i], "]\n")
}
# Plot residuals
plot(x, residuals, main = "Residuals of Nonlinear Regression Model", xlab = "x", ylab = "Residuals", pch = 19, col = "darkgreen")
abline(h = 0, col = "red", lwd = 2)  # Horizontal line at zero for reference

# Apply Anderson-Darling test on the 'x' column
library(nortest)
result <- ad.test(residuals)
print(result) 