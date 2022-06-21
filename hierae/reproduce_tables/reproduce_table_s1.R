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
    (read_csv(here::here("data", "benchmarks", cancer, "hierarchicalsaenet_extra_hidden_layer_scores_all_timed_euler_es_no_hidden_layer.csv"))) %>%
      dplyr::select(model, concordance) %>%
      rename(Model = model) %>%
      mutate(cancer = cancer) %>%
      mutate(Model = "HierarchicalSAE") %>%
      mutate(ablation = "None")
  )

  plt_frame <- rbind(
    plt_frame,
    (read_csv(here::here("~", "msc_thesis", "data", "benchmarks", cancer, "hierarchicsalsenet_scores_all_timed_euler_es.csv"))) %>%
      dplyr::select(model, concordance) %>%
      rename(Model = model) %>%
      mutate(cancer = cancer) %>%
      mutate(Model = "HierarchicalSAE") %>%
      mutate(ablation = "Additional hidden layer")
  )

  plt_frame <- rbind(
    plt_frame,
    (read_csv(here::here("data", "benchmarks", cancer, "hierarchicalsaenet_scores_all_timed_euler_es_no_hidden_layer_no_supervision.csv"))) %>%
      dplyr::select(model, concordance) %>%
      rename(Model = model) %>%
      mutate(cancer = cancer) %>%
      mutate(Model = "HierarchicalSAE") %>%
      mutate(ablation = "No supervision of modality-specific autoencoders")
  )

  plt_frame <- rbind(
    plt_frame,
    (read_csv(here::here("~", "msc_thesis", "data", "benchmarks", cancer, "concatsaenet_scores_all_timed_euler_es.csv"))) %>%
      dplyr::select(model, concordance) %>%
      rename(Model = model) %>%
      mutate(cancer = cancer) %>%
      mutate(Model = "ConcatSAE") %>%
      mutate(ablation = "None")
  )

  plt_frame <- rbind(
    plt_frame,
    (read_csv(here::here("~", "msc_thesis", "data", "benchmarks", cancer, "hierarchicalsaenet_encode_only_extra_hidden_layer_scores_all_timed_euler_es.csv"))) %>%
      dplyr::select(model, concordance) %>%
      rename(Model = model) %>%
      mutate(cancer = cancer) %>%
      mutate(Model = "ConcatSAE") %>%
      mutate(ablation = "Additional hidden layer")
  )

  plt_frame <- rbind(
    plt_frame,
    (read_csv(here::here("data", "benchmarks", cancer, "concatsaenet_scores_all_timed_euler_es_no_supervision.csv"))) %>%
      dplyr::select(model, concordance) %>%
      rename(Model = model) %>%
      mutate(cancer = cancer) %>%
      mutate(Model = "ConcatSAE") %>%
      mutate(ablation = "No supervision of modality-specific autoencoders")
  )

  plt_frame <- rbind(
    plt_frame,
    (read_csv(here::here("~", "msc_thesis", "data", "benchmarks", cancer, "msaenet_scores_all_timed_euler_es.csv"))) %>%
      dplyr::select(model, concordance) %>%
      rename(Model = model) %>%
      mutate(cancer = cancer) %>%
      mutate(Model = "MeanSAE") %>%
      mutate(ablation = "None")
  )

  plt_frame <- rbind(
    plt_frame,
    (read_csv(here::here("~", "msc_thesis", "data", "benchmarks", cancer, "msaenet_encode_only_scores_all_timed_euler_es.csv"))) %>%
      dplyr::select(model, concordance) %>%
      rename(Model = model) %>%
      mutate(cancer = cancer) %>%
      mutate(Model = "MeanSAE") %>%
      mutate(ablation = "Additional hidden layer")
  )

  plt_frame <- rbind(
    plt_frame,
    (read_csv(here::here("data", "benchmarks", cancer, "msaenet_scores_all_timed_euler_es_no_supervision.csv"))) %>%
      dplyr::select(model, concordance) %>%
      rename(Model = model) %>%
      mutate(cancer = cancer) %>%
      mutate(Model = "MeanSAE") %>%
      mutate(ablation = "No supervision of modality-specific autoencoders")
  )

  plt_frame <- rbind(
    plt_frame,
    (read_csv(here::here("~", "msc_thesis", "data", "benchmarks", cancer, "msaenet_max_scores_all_timed_euler_es.csv"))) %>%
      dplyr::select(model, concordance) %>%
      rename(Model = model) %>%
      mutate(cancer = cancer) %>%
      mutate(Model = "MaxSAE") %>%
      mutate(ablation = "None")
  )

  plt_frame <- rbind(
    plt_frame,
    (read_csv(here::here("~", "msc_thesis", "data", "benchmarks", cancer, "msaenet_max_encode_only_scores_all_timed_euler_es.csv"))) %>%
      dplyr::select(model, concordance) %>%
      rename(Model = model) %>%
      mutate(cancer = cancer) %>%
      mutate(Model = "MaxSAE") %>%
      mutate(ablation = "Additional hidden layer")
  )

  plt_frame <- rbind(
    plt_frame,
    (read_csv(here::here("data", "benchmarks", cancer, "msaenet_max_scores_all_timed_euler_es_no_supervision.csv"))) %>%
      dplyr::select(model, concordance) %>%
      rename(Model = model) %>%
      mutate(cancer = cancer) %>%
      mutate(Model = "MaxSAE") %>%
      mutate(ablation = "No supervision of modality-specific autoencoders")
  )
}

plt_frame$Model <- factor(plt_frame$Model, levels = c(
  "HierarchicalSAE", "MeanSAE", "MaxSAE", "ConcatSAE"
))

plt_frame$ablation <- factor(plt_frame$ablation, levels = c(
  "None", "Additional hidden layer", "No supervision of modality-specific autoencoders"
))

plt_frame %>%
  group_by(Model, ablation) %>%
  summarise(concordance = paste0(
    round(mean(concordance), 3), " (", round(sd(concordance), 3), ")"
  )) %>%
  write_csv(here::here("tables", "table_s1.csv"))
