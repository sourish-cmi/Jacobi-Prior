library(MASS)       # for mvrnorm
library(glmnet)     # for lasso/ridge/elastic net
library(caret)      # for cross-validation utilities
library(ggplot2)    # for visualisation

No.of.Datasets=50
RNGkind(sample.kind = "Rounding")
set.seed(883)
# Parameters
beta <- c(3, 1.5, 0, 0, 2, 0, 0, 0)
sigma <- 3
p <- length(beta)
rho <- 0.5
n <- 100

# Covariance matrix
Sigma <- matrix(NA, p, p)
for (i in 1:p){
  for (j in 1:p){
    Sigma[i,j] <- sigma * rho^(abs(i-j))
  }
}
mu <- rep(0, p)

# Simulate predictors
X <- mvrnorm(n=n, mu=mu, Sigma=Sigma)

# Linear predictor and binary response
eta <- X %*% beta
#prob <- 1/(1 + exp(-eta))
prob <- exp(eta)/(1 + exp(eta))
y <- rbinom(n=n, size=1, prob=prob)

# Split into train/test for evaluation
train_index <- createDataPartition(y, p=0.8, list=FALSE)
X_train <- X[train_index, ]
y_train <- y[train_index]
X_test <- X[-train_index, ]
y_test <- y[-train_index]

# Helper: rmse on test set
rmse <- function(y_true, y_prob) {
  return(sqrt(mean((y_true-y_prob)^2)))
}



# Grid of (a,b) for Jacobi prior
ab_grid <- expand.grid(a=seq(0.01,0.5,by=0.001)
                       , b=seq(0.01,0.5,by=0.001))
results_jacobi <- data.frame(a=ab_grid$a, b=ab_grid$b
                             , rmse=NA)

# Placeholder Jacobi estimator (to be implemented)
jacobi_estimator <- function(X, y, a, b) {
  m =1 
  eta = log((y+a)/(b+m-y))
  beta_hat <- solve(t(X) %*% X) %*% t(X) %*% eta
  return(as.vector(beta_hat))
}

# Evaluate Jacobi for each (a,b)
 
# Visualise
library(ggplot2)
library(viridis)

results_jacobi_rounded <- results_jacobi
for(i in 1:nrow(results_jacobi)){
  ai = results_jacobi_rounded[i,'a']
  bi = results_jacobi_rounded[i,'b']
  beta_hat = jacobi_estimator(X_train,y_train,a=ai,b=bi)
  z_tst = X_test%*%beta_hat
  y_pred = 1/(1+exp(-z_tst))
  results_jacobi_rounded[i,'rmse']=rmse(y_true=y_test
                                        ,y_prob = y_pred)
}

library(ggplot2)
library(viridis)
library(scales)
library(scico)

results_jacobi_rounded$a <- round(results_jacobi_rounded$a, 2)
results_jacobi_rounded$b <- round(results_jacobi_rounded$b, 2)

ggplot(results_jacobi_rounded, aes(x = a, y = b, fill = rmse)) +
  geom_tile(color = "white", linewidth = 0.3) +
  scale_fill_scico(palette = "vik", direction = -1, name = "RMSE") +
  labs(
    title = "",
    subtitle = "RMSE over grid of (a, b)",
    x = expression(italic("a")),
    y = expression(italic("b"))
  ) +
  theme_bw(base_size = 14) +
  theme(
    panel.grid = element_blank(),
    plot.title = element_text(face = "bold", size = 16),
    plot.subtitle = element_text(size = 12),
    axis.title = element_text(face = "bold"),
    legend.position = "right",
    axis.text.x = element_text(angle = 90, vjust = 0.5, hjust = 1)
  )