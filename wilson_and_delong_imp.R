setwd("C:/Users/075be/Downloads/thesis")

cl <- makeCluster(max(1, detectCores() - 1))
registerDoParallel(cl)
set.seed(42)

# ── 1. Load & Prepare Data 
setwd("C:/Users/075be/Downloads/thesis")

df <- read.csv("gait_features_rich.csv", stringsAsFactors = FALSE)
View(df)
df <- df[, !(colnames(df) %in% "Subject")]
df$Class <- factor(ifelse(df$Class == "ASD", "ASD", "NonASD"),
                   levels = c("NonASD", "ASD"))
feat_cols <- setdiff(colnames(df), "Class")

cat("Dimensions:", dim(df), "\n")
cat("Class distribution:\n"); print(table(df$Class)); cat("\n")

# ── 2. Remove NZV & Highly Correlated Features
nzv_idx <- nearZeroVar(df[, feat_cols])
if (length(nzv_idx) > 0) {
  df        <- df[, !(colnames(df) %in% feat_cols[nzv_idx])]
  feat_cols <- setdiff(colnames(df), "Class")
  cat("Removed", length(nzv_idx), "near-zero-variance features.\n")
}

cor_mat  <- cor(df[, feat_cols], use = "pairwise.complete.obs")
high_cor <- findCorrelation(cor_mat, cutoff = 0.95, verbose = FALSE)
if (length(high_cor) > 0) {
  df        <- df[, -which(colnames(df) %in% feat_cols[high_cor])]
  feat_cols <- setdiff(colnames(df), "Class")
  cat("Removed", length(high_cor), "highly correlated features.\n")
}
cat("Features remaining:", length(feat_cols), "\n\n")

################################

test_res <- do.call(rbind, lapply(feat_cols, function(col) {
  sw_p <- tryCatch(shapiro.test(df[[col]])$p.value, error = function(e) NA)
  
  t_res <- t.test(df[[col]] ~ df$Class, var.equal = FALSE)   # Welch
  w_res <- wilcox.test(df[[col]] ~ df$Class, exact = FALSE)
  
  chosen_test <- if (!is.na(sw_p) && sw_p > 0.05) "Welch_t" else "Wilcoxon"
  chosen_p <- if (!is.na(sw_p) && sw_p > 0.05) t_res$p.value else w_res$p.value
  
  data.frame(
    Feature = col,
    Shapiro_p = round(sw_p, 4),
    Chosen_Test = chosen_test,
    Chosen_p = round(chosen_p, 4),
    Welch_p = round(t_res$p.value, 4),
    Wilcoxon_p = round(w_res$p.value, 4),
    Sig = ifelse(chosen_p < 0.05, "***", "ns"),
    stringsAsFactors = FALSE
  )
}))

print(test_res)
#################################

# ── 3. Group Difference Tests ─────────────────────────────────────────────────
test_res <- do.call(rbind, lapply(feat_cols, function(col) {
  sw_p <- tryCatch(shapiro.test(df[[col]])$p.value, error = function(e) NA)
  if (!is.na(sw_p) && sw_p > 0.05) {
    r <- t.test(df[[col]] ~ df$Class)
    data.frame(Feature = col, Test = "t-test",
               p_value = round(r$p.value, 4),
               Sig = ifelse(r$p.value < 0.05, "***", "ns"),
               stringsAsFactors = FALSE)
  } else {
    r <- wilcox.test(df[[col]] ~ df$Class, exact = FALSE)
    data.frame(Feature = col, Test = "Wilcoxon",
               p_value = round(r$p.value, 4),
               Sig = ifelse(r$p.value < 0.05, "***", "ns"),
               stringsAsFactors = FALSE)
  }
}))

sig_feats <- test_res$Feature[test_res$Sig == "***"]
cat(sprintf("Significant features (p<0.05): %d / %d\n\n",
            length(sig_feats), length(feat_cols)))
print(test_res[test_res$Sig == "***", ])


# ── 5. Boruta Feature Selection 
set.seed(42)
boruta_res <- tryCatch(
  Boruta(Class ~ ., data = df, doTrace = 0, maxRuns = 200),
  error = function(e) NULL
)
selected <- if (!is.null(boruta_res)) {
  getSelectedAttributes(TentativeRoughFix(boruta_res))
} else character(0)

if (length(selected) == 0) {
  cat("Boruta found no features.\n")
  selected <- if (length(sig_feats) > 0) sig_feats else feat_cols
}
cat("Selected features (", length(selected), "):",
    paste(selected, collapse = ", "), "\n\n")
plot(boruta_res,las=2,cex.axis=0.7)
df_sel    <- df[, c(selected, "Class")]
sel_feats <- selected


# ── 6. Train / Test Split
split_idx  <- createDataPartition(df_sel$Class, p = 0.70, list = FALSE)
train_raw  <- df_sel[ split_idx, ]
test_raw   <- df_sel[-split_idx, ]
cat(sprintf("Train: %d  |  Test: %d\n\n", nrow(train_raw), nrow(test_raw)))

# ── 7. Preprocessing + SMOTE (train only) 
pre_proc     <- preProcess(train_raw[, sel_feats], method = c("center", "scale"))
train_scaled <- cbind(predict(pre_proc, train_raw[, sel_feats]),
                      Class = train_raw$Class)
test_scaled  <- cbind(predict(pre_proc, test_raw[, sel_feats]),
                      Class = test_raw$Class)

train_balanced <- recipe(Class ~ ., data = train_scaled) |>
  themis::step_smote(Class, over_ratio = 1, seed = 42) |>
  prep() |> bake(new_data = NULL)

cat("After SMOTE:"); print(table(train_balanced$Class)); cat("\n")
n_train <- nrow(train_balanced)

# ── 8. CV Control 
# Use 5-fold (3 repeated) to avoid empty-class folds with small SMOTE set
ctrl <- trainControl(
  method          = "repeatedcv",
  number          = 5,
  repeats         = 3,
  classProbs      = TRUE,
  summaryFunction = twoClassSummary,
  savePredictions = "final",
  allowParallel   = TRUE
)

# ── 8.5 Wilson CI Helper 
wilson_ci <- function(x, n, conf.level = 0.95) {
  if (is.na(x) || is.na(n) || n == 0) {
    return(c(lower = NA, upper = NA))
  }
  
  z <- qnorm(1 - (1 - conf.level) / 2)
  p <- x / n
  
  denom  <- 1 + (z^2 / n)
  center <- (p + z^2 / (2 * n)) / denom
  half   <- (z * sqrt((p * (1 - p) / n) + (z^2 / (4 * n^2)))) / denom
  
  c(lower = center - half, upper = center + half)
}

# ── 9. Evaluation Helper 
eval_model <- function(model, test_data, name) {
  probs <- predict(model, newdata = test_data, type = "prob")[, "ASD"]
  preds <- predict(model, newdata = test_data)
  cm    <- confusionMatrix(preds, test_data$Class, positive = "ASD")
  
  roc_o <- roc(response = test_data$Class,
               predictor = probs,
               levels = c("NonASD", "ASD"),
               direction = "<",
               quiet = TRUE)
  auc_v  <- as.numeric(auc(roc_o))
  auc_ci <- as.numeric(ci.auc(roc_o, method = "delong"))
  
  # confusion matrix cells
  tp <- as.numeric(cm$table["ASD", "ASD"])
  fn <- as.numeric(cm$table["NonASD", "ASD"])
  fp <- as.numeric(cm$table["ASD", "NonASD"])
  tn <- as.numeric(cm$table["NonASD", "NonASD"])
  
  # denominators
  n_total <- tp + tn + fp + fn
  n_pos   <- tp + fn
  n_neg   <- tn + fp
  
  # Wilson CIs
  acc_ci  <- wilson_ci(tp + tn, n_total)
  sens_ci <- wilson_ci(tp, n_pos)
  spec_ci <- wilson_ci(tn, n_neg)
  
  cat(sprintf(
    "\n══════════════════════════════════════════\n  %s\n══════════════════════════════════════════\n",
    name))
  print(cm$table)
  cat(sprintf(
    paste0(
      "  Accuracy   : %.4f (95%% CI: %.4f–%.4f)\n",
      "  Sensitivity: %.4f (95%% CI: %.4f–%.4f)\n",
      "  Specificity: %.4f (95%% CI: %.4f–%.4f)\n",
      "  Precision  : %.4f\n",
      "  F1 Score   : %.4f\n",
      "  AUC        : %.4f (95%% CI: %.4f–%.4f)\n"
    ),
    cm$overall["Accuracy"], acc_ci["lower"], acc_ci["upper"],
    cm$byClass["Sensitivity"], sens_ci["lower"], sens_ci["upper"],
    cm$byClass["Specificity"], spec_ci["lower"], spec_ci["upper"],
    cm$byClass["Precision"],
    cm$byClass["F1"],
    auc_v, auc_ci[1], auc_ci[3]
  ))
  
  list(
    name        = name,
    model       = model,
    cm          = cm,
    roc         = roc_o,
    auc         = auc_v,
    auc_lower   = auc_ci[1],
    auc_upper   = auc_ci[3],
    accuracy    = as.numeric(cm$overall["Accuracy"]),
    acc_lower   = acc_ci["lower"],
    acc_upper   = acc_ci["upper"],
    sensitivity = as.numeric(cm$byClass["Sensitivity"]),
    sens_lower  = sens_ci["lower"],
    sens_upper  = sens_ci["upper"],
    specificity = as.numeric(cm$byClass["Specificity"]),
    spec_lower  = spec_ci["lower"],
    spec_upper  = spec_ci["upper"],
    precision   = as.numeric(cm$byClass["Precision"]),
    f1          = as.numeric(cm$byClass["F1"])
  )
}

results_list <- list()   # collect results safely

# ── 10. Logistic Regression (Elastic Net)
cat("\n>>> Logistic Regression (Elastic Net) ...\n")
lr_grid <- expand.grid(
  alpha  = c(0, 0.25, 0.5, 0.75, 1),
  lambda = 10^seq(-4, 1, length = 40)
)
lr_model <- train(Class ~ ., data = train_balanced,
                  method = "glmnet", trControl = ctrl,
                  tuneGrid = lr_grid, metric = "ROC")
cat("Best LR — alpha:", lr_model$bestTune$alpha,
    "  lambda:", round(lr_model$bestTune$lambda, 5), "\n")
results_list[["LR"]] <- eval_model(lr_model, test_scaled,
                                   "Logistic Regression (Elastic Net)")

# ── 11. Random Forest 
cat("\n>>> Random Forest ...\n")
rf_grid <- data.frame(mtry = unique(c(2, 3, 4,
                                      round(sqrt(length(sel_feats))),
                                      round(length(sel_feats) / 2))))
rf_model <- train(Class ~ ., data = train_balanced,
                  method = "rf", trControl = ctrl,
                  tuneGrid = rf_grid, ntree = 1000,
                  importance = TRUE, metric = "ROC")
cat("Best RF mtry:", rf_model$bestTune$mtry, "\n")
results_list[["RF"]] <- eval_model(rf_model, test_scaled, "Random Forest")

png("plot_rf_importance.png", width = 800, height = 600, res = 100)
varImpPlot(rf_model$finalModel,
           main = "Random Forest – Variable Importance", pch = 19)
dev.off()



# ── 12. SVM (RBF) 
cat("\n>>> SVM (RBF) ...\n")
svm_grid <- expand.grid(
  C     = c(0.01, 0.1, 0.5, 1, 5, 10, 50, 100),
  sigma = c(0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1)
)
svm_model <- train(Class ~ ., data = train_balanced,
                   method = "svmRadial", trControl = ctrl,
                   tuneGrid = svm_grid, metric = "ROC")

cat("Best SVM — C:", svm_model$bestTune$C,
    "  sigma:", svm_model$bestTune$sigma, "\n")
results_list[["SVM"]] <- eval_model(svm_model, test_scaled, "SVM (RBF Kernel)")


stopCluster(cl); registerDoSEQ()

# ── 15. Comparison Table
comparison <- do.call(rbind, lapply(results_list, function(r) {
  data.frame(
    Model       = r$name,
    Accuracy    = round(r$accuracy,    4),
    Acc_CI      = sprintf("(%.4f, %.4f)", r$acc_lower, r$acc_upper),
    Sensitivity = round(r$sensitivity, 4),
    Sens_CI     = sprintf("(%.4f, %.4f)", r$sens_lower, r$sens_upper),
    Specificity = round(r$specificity, 4),
    Spec_CI     = sprintf("(%.4f, %.4f)", r$spec_lower, r$spec_upper),
    Precision   = round(r$precision,   4),
    F1          = round(r$f1,          4),
    AUC         = round(r$auc,         4),
    AUC_CI      = sprintf("(%.4f, %.4f)", r$auc_lower, r$auc_upper),
    stringsAsFactors = FALSE
  )
}))
rownames(comparison) <- NULL

cat("\n\n══════════════════════════════════════════════════════\n")
cat("              MODEL COMPARISON SUMMARY\n")
cat("══════════════════════════════════════════════════════\n")
print(comparison)

best_idx <- which.max(sapply(results_list, function(x) x$auc))
best <- results_list[[best_idx]]
cat(sprintf("\n★  Best Model: %s  |  AUC=%.4f  Acc=%.4f  F1=%.4f\n\n",
            best$name, best$auc, best$accuracy, best$f1))

write.csv(comparison, "model_comparison_results.csv", row.names = FALSE)

