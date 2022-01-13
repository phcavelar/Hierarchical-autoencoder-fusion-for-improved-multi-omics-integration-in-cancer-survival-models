library(here)
library(blockForest)
library(glmnet)
library(coefplot)
library(ranger)
library(tuneRanger)
library(tictoc)
library(readr)
library(vroom)
library(Hmisc)
library(prioritylasso)
library(CoxBoost)

#' Run multi-omics survival benchmark as per the requirements of our study.
#'
#' @description
#' `run_benchmark()` allows for easy benchmarking, as we can abstract
#' all the model specific code handling away to live here and can let
#' our main driver script not worry about subtle API differences between models.
#' @param model String specifiying which model is to be run.
#' Possible values are:
#'   - "blockforest"
#'   - "randomblock_favoring"
#'   - "ranger"
#'   - "glmnet"
#'   - "prioritylasso_favoring"
#'   - "clinical_ridge"
#' @param X_train nxp design training matrix.
#' @param y_train nx1 `Surv` survival training targets.
#' @param X_test nxp design test matrix.
#' @param y_test nx1 `Surv` survival test targets.
#' @param blocks List specifying which block (i.e., modality) each input variable
#' belongs to.
#' @param seed Integer specifying the random seed to be set for reproducibility.
run_benchmark <- function(model,
                          X_train,
                          y_train,
                          X_test,
                          y_test,
                          blocks,
                          seed = 42) {
  allowable_models <- c(
    "blockforest",
    "randomblock_favoring",
    "ranger",
    "glmnet",
    "prioritylasso_favoring",
    "clinical_ridge",
    "coxboost_favoring"
  )
  if (!model %in% allowable_models) {
    stop("Model chosen not available. Please check the documentation for allowable models.")
  }
  if (model %in% c("blockforest", "randomblock_favoring")) {
    set.seed(seed)
    block.method <- "BlockForest"
    always.select.block <- 0
    if (model == "randomblock_favoring") {
      block.method <- "RandomBlock"
      # Favor clinical variables for RandomBlock.
      # Thus, have to make sure that clinical is always the first
      # block, but we handle this.
      always.select.block <- 1
    }
    tic()
    learner <- blockfor(
      X = X_train,
      y = y_train,
      blocks = blocks,
      block.method = block.method,
      always.select.block = always.select.block,
      seed = seed,
      verbose = FALSE
    )
    exectime <- toc()
    exectime <- exectime$toc - exectime$tic
    concordance <- unname(rcorr.cens(
      -rowSums(stats::predict(learner$forest, data.frame(X_test))$chf),
      Surv(y_test)
    )[1])
    list(
      concordance = concordance,
      params = NA,
      selected_variables = NA,
      time = unname(exectime)
    )
  } else if (model == "glmnet") {
    set.seed(seed)
    alpha <- 1
    colnames(y_train) <- c("time", "status")
    colnames(y_test) <- c("time", "status")
    nFolds <- 5
    foldid <- sample(rep(seq(nFolds), length.out = nrow(X_train)))
    tic()
    learner <- cv.glmnet(
      x = X_train,
      y = y_train,
      type.measure = "C",
      family = "cox",
      nfolds = nFolds,
      foldid = foldid,
      alpha = alpha
    )
    exectime <- toc()
    exectime <- exectime$toc - exectime$tic
    selected_variables <- rownames(extract.coef(learner))
    params <- c(lambda.min = learner$lambda.min)
    concordance <- unname(rcorr.cens(
      -predict(learner,
        newx = X_test,
        # Use minimum lambda for all `glmnet` related models.
        s = "lambda.min", arg = "response"
      )[, 1],
      Surv(y_test)
    )[1])
    list(
      concordance = concordance,
      params = params,
      selected_variables = selected_variables,
      time = unname(exectime)
    )
  } else if (model %in% c("coxboost", "coxboost_favoring")) {
    set.seed(seed)
    unpen.index <- NULL
    if (model == "coxboost_favoring") {
      unpen.index <- grep("clinical", colnames(X_train))
    }
    learner <- cv.CoxBoost(
      as.vector(y_train[, 1L]),
      as.vector(y_train[, 2L]),
      x = X_train,
      K = 5,
      unpen.index = unpen.index,
      penalty = 9 * sum(y_train[, 2L]),
      multicore = FALSE
    )
    stepno <- learner$optimal.step
    learner <- CoxBoost(
      time = as.vector(y_train[, 1L]),
      status = as.vector(y_train[, 2L]),
      x = X_train,
      stepno = stepno,
      penalty = 9 * sum(y_train[, 2L]),
      unpen.index = unpen.index
    )

    selected_variables <- colnames(X_train)[which(learner$coefficients[learner$stepno, ] != 0)]
    params <- c(stepno = stepno)

    train_predictions <- predict(
      learner,
      newdata = X_train,
      newtime = as.vector(y_train[, 1L]),
      newstatus = as.vector(y_train[, 2L]),
      type = "lp"
    )[1, ]
    test_predictions <- predict(
      learner,
      newdata = X_test,
      newtime = as.vector(y_test[, 1L]),
      newstatus = as.vector(y_test[, 2L]),
      type = "lp"
    )[1, ]

    concordance <- unname(rcorr.cens(
      -test_predictions,
      Surv(y_test)
    )[1])
    list(
      concordance = concordance,
      params = params,
      selected_variables = selected_variables
    )
  } else if (model == "prioritylasso_favoring") {
    set.seed(seed)
    colnames(y_train) <- c("time", "stop")
    colnames(y_test) <- c("time", "stop")
    nFolds <- 5
    foldid <- sample(rep(seq(nFolds), length.out = nrow(X_train)))
    tic()
    learner <- prioritylasso(
      X_train,
      Surv(y_train[, 1], y_train[, 2]),
      blocks = blocks,
      nfolds = 5,
      cvoffset = TRUE,
      cvoffsetnfolds = 5,
      family = "cox",
      foldid = foldid,
      type.measure = "deviance",
      block1.penalization = FALSE
    )
    exectime <- toc()
    exectime <- exectime$toc - exectime$tic
    selected_variables <- names(which(learner$coefficients != 0))
    params <- c(lambda.min = learner$lambda.min)
    # In case there is very strong correlations within clinical data,
    # this can lead to some coefficients being `NA` as prioritylasso
    # favoring fits the first (i.e., clinical) block unpenalized.
    # Thus, we subset to the non-zero (and thus, non-NA) coefficients
    # to calculate the predictions.
    train_predictions <- exp(X_train[, which(learner$coefficients != 0)] %*% learner$coefficients[which(learner$coefficients != 0)])[, 1]
    test_predictions <- exp(X_test[, which(learner$coefficients != 0)] %*% learner$coefficients[which(learner$coefficients != 0)])[, 1]
    concordance <- unname(rcorr.cens(
      -test_predictions,
      Surv(y_test)
    )[1])
    list(
      concordance = concordance,
      params = params,
      selected_variables = selected_variables,
      time = unname(exectime)
    )
  } else if (model == "ranger") {
    set.seed(seed)
    colnames(y_train) <- c("time", "status")
    colnames(y_test) <- c("time", "status")
    tic()
    learner <- tuneRanger::tuneMtryFast(
      dependent.variable.name = "time",
      status.variable.name = "status",
      data = cbind(y_train, X_train),
      doBest = TRUE,
      trace = FALSE,
      plot = FALSE,
      seed = seed
    )
    exectime <- toc()
    exectime <- exectime$toc - exectime$tic
    selected_variables <- NA
    params <- c(mtry = learner$mtry)
    concordance <- unname(rcorr.cens(
      -rowSums(predict(learner, data = X_test)$chf),
      Surv(y_test)
    )[1])
    list(
      concordance = concordance,
      params = params,
      selected_variables = selected_variables,
      time = unname(exectime)
    )
  }
  # Default to clinical cox ridge
  else if (model == "clinical_ridge") {
    set.seed(seed)
    clinical_col_ix <- grep("clin", colnames(X_train))
    colnames(y_train) <- c("time", "status")
    colnames(y_test) <- c("time", "status")
    nFolds <- 5
    foldid <- sample(rep(seq(nFolds), length.out = nrow(X_train)))
    tic()
    learner <- cv.glmnet(
      x = X_train[, clinical_col_ix],
      y = y_train,
      type.measure = "C",
      family = "cox",
      nfolds = nFolds,
      foldid = foldid,
      alpha = 0
    )
    exectime <- toc()
    exectime <- exectime$toc - exectime$tic
    selected_variables <- rownames(extract.coef(learner))
    params <- c(lambda.min = learner$lambda.min)
    concordance <- unname(rcorr.cens(
      -predict(learner,
        newx = X_test[, clinical_col_ix],
        s = "lambda.min", arg = "response"
      )[, 1],
      Surv(y_test)
    )[1])
    list(
      concordance = concordance,
      params = params,
      selected_variables = selected_variables,
      time = unname(exectime)
    )
  }
}
