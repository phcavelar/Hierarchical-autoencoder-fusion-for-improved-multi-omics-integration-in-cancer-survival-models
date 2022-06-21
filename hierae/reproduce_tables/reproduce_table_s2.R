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
    (read_csv(here::here("~", "msc_thesis", "data", "benchmarks", cancer, "concatsaenet_scores_all_timed_euler_es.csv"))) %>%
      dplyr::select(model, concordance) %>%
      rename(Model = model) %>%
      mutate(cancer = cancer) %>%
      mutate(Model = "ConcatSAE") %>%
      mutate(data = "all")
  )

  plt_frame <- rbind(
    plt_frame,
    (read_csv(here::here("~", "msc_thesis", "data", "benchmarks", cancer, "concatsaenet_scores_[-rppa_cnv_meth_mirna_mut]_timed_euler_es.csv"))) %>%
      dplyr::select(model, concordance) %>%
      rename(Model = model) %>%
      mutate(cancer = cancer) %>%
      mutate(Model = "ConcatSAE") %>%
      mutate(data = "clinical + gex")
  )

  plt_frame <- rbind(
    plt_frame,
    (read_csv(here::here("~", "msc_thesis", "data", "benchmarks", cancer, "msaenet_scores_all_timed_euler_es.csv"))) %>%
      dplyr::select(model, concordance) %>%
      rename(Model = model) %>%
      mutate(cancer = cancer) %>%
      mutate(Model = "MeanSAE") %>%
      mutate(data = "all")
  )

  plt_frame <- rbind(
    plt_frame,
    (read_csv(here::here("~", "msc_thesis", "data", "benchmarks", cancer, "msaenet_scores_[-rppa_cnv_meth_mirna_mut]_timed_euler_es.csv"))) %>%
      dplyr::select(model, concordance) %>%
      rename(Model = model) %>%
      mutate(cancer = cancer) %>%
      mutate(Model = "MeanSAE") %>%
      mutate(data = "clinical + gex")
  )

  plt_frame <- rbind(
    plt_frame,
    (read_csv(here::here("~", "msc_thesis", "data", "benchmarks", cancer, "msaenet_max_scores_all_timed_euler_es.csv"))) %>%
      dplyr::select(model, concordance) %>%
      rename(Model = model) %>%
      mutate(cancer = cancer) %>%
      mutate(Model = "MaxSAE") %>%
      mutate(data = "all")
  )

  plt_frame <- rbind(
    plt_frame,
    (read_csv(here::here("~", "msc_thesis", "data", "benchmarks", cancer, "msaenet_max_scores_[-rppa_cnv_meth_mirna_mut]_timed_euler_es.csv"))) %>%
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

  plt_frame <- rbind(
    plt_frame,
    (read_csv(here::here("data", "benchmarks", cancer, "hierarchicalsaenet_encode_only_scores_all_timed_euler_es_no_hidden_layer.csv"))) %>%
      dplyr::select(model, concordance) %>%
      rename(Model = model) %>%
      mutate(cancer = cancer) %>%
      mutate(Model = "HierarchicalSAE (no decoder)") %>%
      mutate(data = "all")
  )
}

plt_frame %>%
  filter(data == "all") %>%
  group_by(Model, cancer) %>%
  summarise(concordance = paste0(
    round(mean(concordance), 3), " (", round(sd(concordance), 3), ")"
  )) %>%
  pivot_wider(names_from = cancer, values_from = concordance) %>%
  write_csv(
    here::here(
      "tables", "table_s2.csv"
    )
  )
