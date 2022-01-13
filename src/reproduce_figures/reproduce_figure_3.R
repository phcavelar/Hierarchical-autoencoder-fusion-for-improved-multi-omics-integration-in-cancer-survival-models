library(here)
library(rjson)
library(readr)
library(dplyr)
library(ggplot2)
library(tidyr)
library(ggsignif)
library(gtable)
library(cowplot)

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
  filter(data == "all") %>%
  group_by(Model) %>%
  summarise(mean = mean(concordance), median = median(concordance), sd = sd(concordance)) %>%
  arrange(desc(mean))

plt_frame %>%
  filter(data == "clinical + gex") %>%
  group_by(Model) %>%
  summarise(mean = mean(concordance), median = median(concordance), sd = sd(concordance)) %>%
  arrange(desc(mean))

plt_frame$Model <- factor(plt_frame$Model, levels = c(
  "HierarchicalSAE", "MeanSAE", "MaxSAE", "ConcatSAE",
  unique(plt_frame$Model)[-grep("(HierarchicalSAE|MeanSAE|MaxSAE|ConcatSAE)", unique(plt_frame$Model))]
))



plt_frame %>%
  filter(data == "all") %>%
  ggplot(aes(x = Model, y = concordance, fill = Model)) +
  geom_boxplot(outlier.size = 0.5, fatten = 1.5) +
  facet_wrap(~cancer, scales = "free_y") +
  labs(x = "", y = "Test concordance") +
  scale_x_discrete(labels = NULL, breaks = NULL) +
  theme_bw(base_size = 8) +
  theme(legend.title = element_blank()) -> p


# Source: https://stackoverflow.com/questions/54438495/shift-legend-into-empty-facets-of-a-faceted-plot-in-ggplot2
shift_legend <- function(p) {

  # check if p is a valid object
  if (!"gtable" %in% class(p)) {
    if ("ggplot" %in% class(p)) {
      gp <- ggplotGrob(p) # convert to grob
    } else {
      message("This is neither a ggplot object nor a grob generated from ggplotGrob. Returning original plot.")
      return(p)
    }
  } else {
    gp <- p
  }

  # check for unfilled facet panels
  facet.panels <- grep("^panel", gp[["layout"]][["name"]])
  empty.facet.panels <- sapply(facet.panels, function(i) "zeroGrob" %in% class(gp[["grobs"]][[i]]))
  empty.facet.panels <- facet.panels[empty.facet.panels]
  if (length(empty.facet.panels) == 0) {
    message("There are no unfilled facet panels to shift legend into. Returning original plot.")
    return(p)
  }

  # establish extent of unfilled facet panels (including any axis cells in between)
  empty.facet.panels <- gp[["layout"]][empty.facet.panels, ]
  empty.facet.panels <- list(
    min(empty.facet.panels[["t"]]), min(empty.facet.panels[["l"]]),
    max(empty.facet.panels[["b"]]), max(empty.facet.panels[["r"]])
  )
  names(empty.facet.panels) <- c("t", "l", "b", "r")

  # extract legend & copy over to location of unfilled facet panels
  guide.grob <- which(gp[["layout"]][["name"]] == "guide-box")
  if (length(guide.grob) == 0) {
    message("There is no legend present. Returning original plot.")
    return(p)
  }
  gp <- gtable_add_grob(
    x = gp,
    grobs = gp[["grobs"]][[guide.grob]],
    t = empty.facet.panels[["t"]],
    l = empty.facet.panels[["l"]],
    b = empty.facet.panels[["b"]],
    r = empty.facet.panels[["r"]],
    name = "new-guide-box"
  )

  # squash the original guide box's row / column (whichever applicable)
  # & empty its cell
  guide.grob <- gp[["layout"]][guide.grob, ]
  if (guide.grob[["l"]] == guide.grob[["r"]]) {
    gp <- gtable_squash_cols(gp, cols = guide.grob[["l"]])
  }
  if (guide.grob[["t"]] == guide.grob[["b"]]) {
    gp <- gtable_squash_rows(gp, rows = guide.grob[["t"]])
  }
  gp <- gtable_remove_grobs(gp, "guide-box")

  return(gp)
}

p.new <- p + guides(fill = guide_legend(ncol = 3))

tmp <- shift_legend(p.new)
ggsave(filename = here::here("figures", "fig-3.pdf"), plot = tmp, device = "pdf", dpi = 600, height = 8, width = 17.07852, units = "cm")
