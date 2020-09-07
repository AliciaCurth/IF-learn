library(grf)

pseudo_CATE <- function(y, w, p, mu_0, mu_1){
  # EIF transformation for CATE
  w_1 <- w/p
  w_0 <- (1-w)/(1-p)
  res <- (w_1-w_0)*y + ((1-w_1)*mu_1-(1-w_0)*mu_0)
  return(res)
}

get_te_predictions <- function(X, y, w, p, X_test, num_trees=2000, seedy=42){
  # function gets treatment effect predictions for:
  # (1) Plug-in learner based on GRF RF
  # (2) Causal forest with known propensity
  # (3) Causal forest with estimated propensity
  # (4) IF-learner with known propensity based on GRF RF
  # (5) IF-learner with unknown propensity based on GRF RF
  
  # fit models for centering as described in Athey et al (2019)
  y.forest <- regression_forest(X, y, seed=seedy, num.trees = num_trees)
  w.forest <- regression_forest(X, w, seed=seedy, num.trees=num_trees)
  
  # use oob-prediction capabilities of grf to get predictions
  y.hat = predict(y.forest)$predictions
  w.hat = predict(w.forest)$predictions
  
  # causal forests -------------------------------------------------------------------------
  # with known propensities and estimated propensities
  cf_knownp <- causal_forest(X, y, w, Y.hat = y.hat, W.hat = p, 
                             num.trees = num_trees, seed=seedy)
  cf_unknownp <- causal_forest(X, y, w, Y.hat = y.hat, W.hat = w.hat, 
                               num.trees = num_trees, seed=seedy)
  
  # make predictions for test data
  pred_knownp <- predict(cf_knownp, X_test)$predictions
  pred_unknownp <- predict(cf_unknownp, X_test)$predictions
  
  # IF-learner -------------------------------------------------------------------
  # make oob predictions for potential outcomes
  n <- nrow(X)
  y0.hat <- rep(NA, n)
  y1.hat <- rep(NA, n)
  
  #y0
  y0.forest <- regression_forest(X[w==0, ], y[w==0], seed=seedy, 
                                 num.trees = num_trees)
  y0.hat[w==0] <- predict(y0.forest)$predictions #oob
  y0.hat[w==1] <- predict(y0.forest, X[w==1,])$predictions
  
  #y1
  y1.forest <- regression_forest(X[w==1, ], y[w==1], seed=seedy, 
                                 num.trees = num_trees)
  y1.hat[w==1] <- predict(y1.forest)$predictions #oob
  y1.hat[w==0] <- predict(y1.forest, X[w==0, ])$predictions
  
  # make pseudo outcome
  pseudo_knownp <- pseudo_CATE(y,w, p, y0.hat, y1.hat)
  pseudo_unknownp <- pseudo_CATE(y,w, w.hat, y0.hat, y1.hat)
  
  # fit second stage regression
  if_knownp <- regression_forest(X, pseudo_knownp, 
                                 seed=seedy, num.trees = num_trees)
  if_unknownp <- regression_forest(X, pseudo_unknownp, 
                                   seed=seedy, num.trees = num_trees)
  
  pred_if_knownp <- predict(if_knownp, X_test)$predictions
  pred_if_unknownp <- predict(if_unknownp, X_test)$predictions
  
  # Adapted IF-learner -------------------------------------------------------
  # make other pseudooutcome -- simplified according to Rubin & vdLaan (2008)
  pseudo_simple <- (w - (1-w))*(y-y.hat)
  s.weights.known <- w * (1/p) + (1-w) * (1/(1-p))
  s.weights.unknown <- w * (1/w.hat) + (1-w) * (1/(1-w.hat))
  
  # fit second stage regression
  if_simple_knownp <- regression_forest(X, pseudo_simple, 
                                 seed=seedy, num.trees = num_trees, 
                                 sample.weights = s.weights.known)
  if_simple_unknownp <- regression_forest(X, pseudo_simple, 
                                   seed=seedy, num.trees = num_trees, 
                                   sample.weights = s.weights.unknown)
  
  pred_if_simple_knownp <- predict(if_simple_knownp, X_test)$predictions
  pred_if_simple_unknownp <- predict(if_simple_unknownp, X_test)$predictions
  
  # weighted version of the former
  y.forest.w <- regression_forest(X, y, sample.weights=s.weights.known, 
                                         seed=seedy, num.trees = num_trees)
  y.hat.w <- predict(y.forest.w)$predictions
  pseudo_simple_knownp_w  <- (w - (1-w))*(y-y.hat.w)
  if_simple_knownp_w <- regression_forest(X, pseudo_simple_knownp_w, 
                                          seed=seedy, num.trees = num_trees, sample.weights = s.weights.known)
  pred_if_simple_knownp_w <- predict(if_simple_knownp_w, X_test)$predictions
  
  # plug-in estimator ---------------------------------------------------
  pred_plugin <- predict(y1.forest, X_test)$predictions - predict(y0.forest, X_test)$predictions
  
  res = data.frame(plugin = pred_plugin, 
                   cf_p = pred_knownp, 
                   cf_np = pred_if_unknownp,
                   if_p = pred_if_knownp, 
                   if_np = pred_if_unknownp,
                   if_s_p = pred_if_simple_knownp,
                   if_s_np = pred_if_simple_unknownp,
                   if_s_p_w = pred_if_simple_knownp_w)
  return(res)
}

