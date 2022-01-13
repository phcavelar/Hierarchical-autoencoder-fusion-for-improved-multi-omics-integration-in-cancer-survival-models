library(here)
library(rjson)
library(readr)
library(dplyr)
library(ggplot2)
library(tidyr)

config <- rjson::fromJSON(
  file = here::here("config", "config.json")
)
my_cols <- config$my_cols
plt_frame <- data.frame()

for (cancer in config$cancers) {
  plt_frame <- rbind(
    plt_frame,
    (read_csv(here::here("data", "benchmarks", cancer, "hierarchicalsaenet_scores_all_timed_euler_es_no_hidden_layer.csv"))) %>%
      dplyr::select(model, concordance) %>%
      rename(Model = model) %>%
      mutate(cancer = cancer) %>%
      mutate(Model = "HierarchicalSAE") %>%
      mutate(data = "all")
  )

  plt_frame <- rbind(
    plt_frame,
    (read_csv(here::here("data", "benchmarks", cancer, "hierarchicalsaenet_scores_[-rppa_cnv_meth_mirna_mut]_timed_euler_es_no_hidden_layer.csv"))) %>%
      dplyr::select(model, concordance) %>%
      rename(Model = model) %>%
      mutate(cancer = cancer) %>%
      mutate(Model = "HierarchicalSAE") %>%
      mutate(data = "clinical + gex")
  )


  plt_frame <- rbind(
    plt_frame,
    (read_csv(here::here("data", "benchmarks", cancer, "concatsaenet_scores_all_timed_euler_es.csv"))) %>%
      dplyr::select(model, concordance) %>%
      rename(Model = model) %>%
      mutate(cancer = cancer) %>%
      mutate(Model = "ConcatSAE") %>%
      mutate(data = "all")
  )

  plt_frame <- rbind(
    plt_frame,
    (read_csv(here::here("data", "benchmarks", cancer, "concatsaenet_scores_[-rppa_cnv_meth_mirna_mut]_timed_euler_es.csv"))) %>%
      dplyr::select(model, concordance) %>%
      rename(Model = model) %>%
      mutate(cancer = cancer) %>%
      mutate(Model = "ConcatSAE") %>%
      mutate(data = "clinical + gex")
  )

  plt_frame <- rbind(
    plt_frame,
    (read_csv(here::here("data", "benchmarks", cancer, "msaenet_scores_all_timed_euler_es.csv"))) %>%
      dplyr::select(model, concordance) %>%
      rename(Model = model) %>%
      mutate(cancer = cancer) %>%
      mutate(Model = "MeanSAE") %>%
      mutate(data = "all")
  )

  plt_frame <- rbind(
    plt_frame,
    (read_csv(here::here("data", "benchmarks", cancer, "msaenet_scores_[-rppa_cnv_meth_mirna_mut]_timed_euler_es.csv"))) %>%
      dplyr::select(model, concordance) %>%
      rename(Model = model) %>%
      mutate(cancer = cancer) %>%
      mutate(Model = "MeanSAE") %>%
      mutate(data = "clinical + gex")
  )

  plt_frame <- rbind(
    plt_frame,
    (read_csv(here::here("data", "benchmarks", cancer, "msaenet_max_scores_all_timed_euler_es.csv"))) %>%
      dplyr::select(model, concordance) %>%
      rename(Model = model) %>%
      mutate(cancer = cancer) %>%
      mutate(Model = "MaxSAE") %>%
      mutate(data = "all")
  )

  plt_frame <- rbind(
    plt_frame,
    (read_csv(here::here("data", "benchmarks", cancer, "msaenet_max_scores_[-rppa_cnv_meth_mirna_mut]_timed_euler_es.csv"))) %>%
      dplyr::select(model, concordance) %>%
      rename(Model = model) %>%
      mutate(cancer = cancer) %>%
      mutate(Model = "MaxSAE") %>%
      mutate(data = "clinical + gex")
  )


  plt_frame <- rbind(
    plt_frame,
    data.frame(
      Model = "RandomBlock favoring",
      cancer = cancer,
      concordance = sapply(
        readRDS(here::here("data", "benchmarks", cancer, "randomblock_favoring_10_stratified_timed_euler.rds")),
        function(x) x$concordance
      ),
      data = "all"
    )
  )

  plt_frame <- rbind(
    plt_frame,
    data.frame(
      Model = "RandomBlock favoring",
      cancer = cancer,
      concordance = sapply(
        readRDS(here::here("data", "benchmarks", cancer, "randomblock_favoring_10_stratified_gex_clinical_euler.rds")),
        function(x) x$concordance
      ),
      data = "clinical + gex"
    )
  )

  plt_frame <- rbind(
    plt_frame,
    data.frame(
      Model = "BlockForest",
      cancer = cancer,
      concordance = sapply(
        readRDS(here::here("data", "benchmarks", cancer, "blockforest_10_stratified_timed_euler.rds")),
        function(x) x$concordance
      ),
      data = "all"
    )
  )


  plt_frame <- rbind(
    plt_frame,
    data.frame(
      Model = "BlockForest",
      cancer = cancer,
      concordance = sapply(
        readRDS(here::here("data", "benchmarks", cancer, "blockforest_10_stratified_gex_clinical_euler.rds")),
        function(x) x$concordance
      ),
      data = "clinical + gex"
    )
  )
  plt_frame <- rbind(
    plt_frame,
    data.frame(
      Model = "Clinical Cox PH",
      cancer = cancer,
      concordance = sapply(
        readRDS(here::here("data", "benchmarks", cancer, "clinical_ridge_10_stratified_timed_euler.rds")),
        function(x) x$concordance
      ),
      data = "all"
    )
  )

  plt_frame <- rbind(
    plt_frame,
    data.frame(
      Model = "Lasso",
      cancer = cancer,
      concordance = sapply(
        readRDS(here::here("data", "benchmarks", cancer, "glmnet_10_stratified_timed_euler.rds")),
        function(x) x$concordance
      ),
      data = "all"
    )
  )

  plt_frame <- rbind(
    plt_frame,
    data.frame(
      Model = "Lasso",
      cancer = cancer,
      concordance = sapply(
        readRDS(here::here("data", "benchmarks", cancer, "glmnet_10_stratified_gex_clinical_euler.rds")),
        function(x) x$concordance
      ),
      data = "clinical + gex"
    )
  )
  plt_frame <- rbind(
    plt_frame,
    data.frame(
      Model = "RSF",
      cancer = cancer,
      concordance = sapply(
        readRDS(here::here("data", "benchmarks", cancer, "ranger_10_stratified_timed_euler.rds")),
        function(x) x$concordance
      ),
      data = "all"
    )
  )

  plt_frame <- rbind(
    plt_frame,
    data.frame(
      Model = "RSF",
      cancer = cancer,
      concordance = sapply(
        readRDS(here::here("data", "benchmarks", cancer, "ranger_10_stratified_gex_clinical_euler.rds")),
        function(x) x$concordance
      ),
      data = "clinical + gex"
    )
  )
  plt_frame <- rbind(
    plt_frame,
    data.frame(
      Model = "prioritylasso favoring",
      cancer = cancer,
      concordance = sapply(
        readRDS(here::here("data", "benchmarks", cancer, "prioritylasso_favoring_10_stratified_all_timed_euler.rds")),
        function(x) x$concordance
      ),
      data = "all"
    )
  )

  plt_frame <- rbind(
    plt_frame,
    data.frame(
      Model = "prioritylasso favoring",
      cancer = cancer,
      concordance = sapply(
        readRDS(here::here("data", "benchmarks", cancer, "prioritylasso_favoring_10_stratified_[-rppa_cnv_meth_mirna_mut]_timed_euler.rds")),
        function(x) x$concordance
      ),
      data = "clinical + gex"
    )
  )
}

plt_frame %>%
  filter(Model %in%
    c(
      "Lasso", "Clinical Cox PH", "RandomBlock favoring", "BlockForest", "MaxSAE", "MeanSAE", "ConcatSAE", "HierarchicalSAE",
      "RSF", "prioritylasso favoring"
    ) &
    data == "all") %>%
  group_by(Model) %>%
  summarise(mean = round(mean(concordance), 3), sd = round(sd(concordance), 3)) %>%
  arrange(desc(mean)) %>%
  mutate(Concordance = paste0(mean, " (", sd, ")")) %>%
  select(Model, Concordance) -> result_frame

second_frame <- plt_frame %>% filter(data == "all" & Model %in% c(
  "Lasso", "Clinical Cox PH", "RandomBlock favoring", "BlockForest", "MaxSAE", "MeanSAE", "ConcatSAE", "HierarchicalSAE",
  "RSF", "prioritylasso favoring"
))
second_frame %>%
  group_by(Model, cancer) %>%
  summarise(mean_concordance = mean(concordance), .groups = "keep") -> tmp
baseline_Models <- unique(tmp$Model)[-grep("HierarchicalSAE", unique(tmp$Model))]


p_vals <- data.frame(
  Model = rep(as.character(unique(unique(tmp$Model)[grep("HierarchicalSAE", unique(tmp$Model))])), each = length(baseline_Models)),
  comparison = rep(baseline_Models, length(as.character(unique(unique(tmp$Model)[grep("HierarchicalSAE", unique(tmp$Model))])))),
  p.val = sapply(
    baseline_Models,
    function(x) {
      t.test(
        x = tmp %>% filter(Model == "HierarchicalSAE") %>% pull(mean_concordance),
        y = tmp %>% filter(Model == x) %>% pull(mean_concordance),
        paired = TRUE,
        mu = 0,
        alternative = "greater"
      )$p.value
    }
  )
)

result_frame %>%
  left_join(p_vals %>% select(p.val, comparison), by = c("Model" = "comparison")) %>%
  mutate(p.val = round(p.val, 10)) %>%
  tidyr::replace_na(list(p.val = "-")) -> multi_omics_result_frame

plt_frame %>%
  filter(Model %in%
    c(
      "Lasso", "Clinical Cox PH", "RandomBlock favoring", "BlockForest", "MaxSAE", "MeanSAE", "ConcatSAE", "HierarchicalSAE",
      "RSF", "prioritylasso favoring"
    ) &
    (data != "all" | (Model == "Clinical Cox PH"))) %>%
  group_by(Model) %>%
  summarise(mean = round(mean(concordance), 3), sd = round(sd(concordance), 3)) %>%
  arrange(desc(mean)) %>%
  mutate(Concordance = paste0(mean, " (", sd, ")")) %>%
  select(Model, Concordance) -> result_frame

second_frame <- plt_frame %>% filter((data != "all" | Model == "Clinical Cox PH") & Model %in% c(
  "Lasso", "Clinical Cox PH", "RandomBlock favoring", "BlockForest", "MaxSAE", "MeanSAE", "ConcatSAE", "HierarchicalSAE",
  "RSF", "prioritylasso favoring"
))
second_frame %>%
  group_by(Model, cancer) %>%
  summarise(mean_concordance = mean(concordance), .groups = "keep") -> tmp
baseline_Models <- unique(tmp$Model)[-grep("HierarchicalSAE", unique(tmp$Model))]


p_vals <- data.frame(
  Model = rep(as.character(unique(unique(tmp$Model)[grep("HierarchicalSAE", unique(tmp$Model))])), each = length(baseline_Models)),
  comparison = rep(baseline_Models, length(as.character(unique(unique(tmp$Model)[grep("HierarchicalSAE", unique(tmp$Model))])))),
  p.val = sapply(
    baseline_Models,
    function(x) {
      t.test(
        x = tmp %>% filter(Model == "HierarchicalSAE") %>% pull(mean_concordance),
        y = tmp %>% filter(Model == x) %>% pull(mean_concordance),
        paired = TRUE,
        mu = 0,
        alternative = "greater"
      )$p.value
    }
  )
)

result_frame %>%
  left_join(p_vals %>% select(p.val, comparison), by = c("Model" = "comparison")) %>%
  mutate(p.val = round(p.val, 5)) %>%
  tidyr::replace_na(list(adjusted_p_val = "-")) -> gex_clinical_results_frame

left_join(multi_omics_result_frame, gex_clinical_results_frame, by = "Model") %>% write_csv(
  here::here(
    "tables", "table_3.csv"
  )
)
