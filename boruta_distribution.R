### fro Review only

boruta_seeds <- c(42, 1, 16, 90, 66)

for (bs in boruta_seeds) {
  set.seed(bs)   # only affects Boruta below
  boruta_res <- Boruta(Class ~ ., data = df,
                       doTrace = 0, maxRuns = 200)
  selected <- getSelectedAttributes(TentativeRoughFix(boruta_res))
  cat("Boruta seed", bs, "-> features:",
      paste(selected, collapse = ", "), "\n")
}