setwd("C:/Users/075be/Downloads/thesis")
library(dplyr)
library(effsize)   # for Cohen's d
library(rstatix)   # for rank-biserial r

df <- read.csv("gait_features_rich.csv", stringsAsFactors = FALSE)
df$Class <- factor(ifelse(df$Class == "ASD", "ASD", "NonASD"))

sig_feats <- c("RHip_skew", "LKnee_zcr", "RAnkle_skew", "LAnkle_rom",
               "LAnkle_kurt", "TrunkY_max", "StepWidth_skew",
               "LAddAbd_peaks", "StepLen_skew", "CoM_Y_min")

results <- lapply(sig_feats, function(f) {
  asd    <- df[[f]][df$Class == "ASD"]
  nonasd <- df[[f]][df$Class == "NonASD"]
  
  # means and SDs
  m_asd  <- round(mean(asd,    na.rm = TRUE), 3)
  m_non  <- round(mean(nonasd, na.rm = TRUE), 3)
  sd_asd <- round(sd(asd,      na.rm = TRUE), 3)
  sd_non <- round(sd(nonasd,   na.rm = TRUE), 3)
  
  # effect size
  sw <- shapiro.test(df[[f]])$p.value
  if (!is.na(sw) && sw > 0.05) {
    # Cohen's d for normally distributed features
    es   <- cohen.d(asd, nonasd)$estimate
    es_r <- round(es, 3)
    es_t <- "Cohen's d"
  } else {
    # rank-biserial r for non-normal
    w    <- wilcox.test(asd, nonasd, exact = FALSE)
    n1   <- length(asd); n2 <- length(nonasd)
    r    <- 1 - (2 * w$statistic) / (n1 * n2)
    es_r <- round(r, 3)
    es_t <- "r (rank-biserial)"
  }
  
  data.frame(Feature=f, ASD_mean=m_asd, ASD_sd=sd_asd,
             NonASD_mean=m_non, NonASD_sd=sd_non,
             Effect_size=es_r, Effect_type=es_t)
})

do.call(rbind, results)


###############################################

# Load data
df <- read.csv("gait_features_rich.csv", stringsAsFactors = FALSE)

# Optional: remove Subject column if it exists
if ("Subject" %in% colnames(df)) {
  df <- df[, !(colnames(df) %in% "Subject")]
}

# Match your source code labeling
df$Class <- factor(ifelse(df$Class == "ASD", "ASD", "NonASD"),
                   levels = c("NonASD", "ASD"))

# Check exact values first
table(df$Class)

# Compute actual min and max for CoM_Y_min by group
asd_vals <- df$CoM_Y_min[df$Class == "ASD"]
td_vals  <- df$CoM_Y_min[df$Class == "NonASD"]

asd_min <- min(asd_vals, na.rm = TRUE)
asd_max <- max(asd_vals, na.rm = TRUE)

td_min  <- min(td_vals, na.rm = TRUE)
td_max  <- max(td_vals, na.rm = TRUE)

cat("ASD   Min:", asd_min, "\n")
cat("ASD   Max:", asd_max, "\n")
cat("TD    Min:", td_min, "\n")
cat("TD    Max:", td_max, "\n")

