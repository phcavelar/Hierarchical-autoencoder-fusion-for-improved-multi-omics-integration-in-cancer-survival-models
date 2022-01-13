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
      mutate(ablation = "None")
  )

  plt_frame <- rbind(
    plt_frame,
    (read_csv(here::here("data", "benchmarks", cancer, "hierarchicalsaenet_scores_all_timed_euler_50_epochs.csv"))) %>%
      dplyr::select(model, concordance) %>%
      rename(Model = model) %>%
      mutate(cancer = cancer) %>%
      mutate(Model = "HierarchicalSAE") %>%
      mutate(ablation = "No early stopping (50 epochs)")
  )

  plt_frame <- rbind(
    plt_frame,
    (read_csv(here::here("data", "benchmarks", cancer, "hierarchicalsaenet_encode_only_scores_all_timed_euler_es.csv"))) %>%
      dplyr::select(model, concordance) %>%
      rename(Model = model) %>%
      mutate(cancer = cancer) %>%
      mutate(Model = "HierarchicalSAE (no decoder)") %>%
      mutate(data = "all")
  )

  plt_frame <- rbind(
    plt_frame,
    (read_csv(here::here("data", "benchmarks", cancer, "hierarchicalsaenet_encode_only_scores_0.01_wd_all_timed_euler_es.csv"))) %>%
      dplyr::select(model, concordance) %>%
      rename(Model = model) %>%
      mutate(cancer = cancer) %>%
      mutate(Model = "HierarchicalSAE") %>%
      mutate(ablation = "No decoder + higher weight decay (lambda = 0.01)")
  )

  plt_frame <- rbind(
    plt_frame,
    (read_csv(here::here("data", "benchmarks", cancer, "hierarchicalsaenet_encode_only_scores_0.1_wd_all_timed_euler_es.csv"))) %>%
      dplyr::select(model, concordance) %>%
      rename(Model = model) %>%
      mutate(cancer = cancer) %>%
      mutate(Model = "HierarchicalSAE") %>%
      mutate(ablation = "No decoder + higher weight decay (lambda = 0.1)")
  )

  plt_frame <- rbind(
    plt_frame,
    (read_csv(here::here("data", "benchmarks", cancer, "shae_encode_only_scores_all_timed_euler_50_epochs.csv"))) %>%
      dplyr::select(model, concordance) %>%
      rename(Model = model) %>%
      mutate(cancer = cancer) %>%
      mutate(Model = "HierarchicalSAE") %>%
      mutate(ablation = "No decoder + no early stopping")
  )
}


plt_frame %>%
  group_by(Model, ablation) %>%
  summarise(mean = mean(concordance), concordance = paste0(
    round(mean(concordance), 3), " (", round(sd(concordance), 3), ")"
  )) %>%
  arrange(desc(mean)) %>%
  select(Model, ablation, concordance) %>%
  write_csv(
    here::here("tables", "table_2.csv")
  )
