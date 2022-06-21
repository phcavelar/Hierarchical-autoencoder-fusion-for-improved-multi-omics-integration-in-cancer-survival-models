library(here)
library(rjson)
library(readr)
library(dplyr)
library(ggplot2)
library(tidyr)
library(ggsignif)
library(cowplot)
library(gridExtra)
library(grid)
library(lattice)

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
      mutate(data = "Clinical + GEX")
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
      mutate(data = "Clinical + GEX")
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
      mutate(data = "Clinical + GEX")
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
      mutate(data = "Clinical + GEX")
  )
}

first_frame <- plt_frame %>% filter(data == "all" & Model %in% c("HierarchicalSAE", "MaxSAE", "ConcatSAE", "MeanSAE"))
first_frame$Model <- factor(first_frame$Model, levels = c("HierarchicalSAE", "MeanSAE", "MaxSAE", "ConcatSAE"))

first_frame %>%
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


first_frame %>% ggplot(
  aes(x = Model, y = concordance, fill = Model)
) +
  geom_boxplot(outlier.size = 0.5, fatten = 1.5) +
  geom_signif(
    annotations = paste0("p = ", round(p_vals$p.val, 7)),
    xmin = p_vals$Model,
    xmax = p_vals$comparison,
    y_position = c(0.975, 1.05, 1.125),
    tip_length = 0,
    textsize = 3.88 / 2.5
  ) +
  coord_cartesian(ylim = c(0.3, 1.2)) +
  scale_x_discrete(labels = NULL, breaks = NULL) +
  theme_bw(base_size = 8) +
  labs(y = "", x = "") +
  scale_fill_manual(values = c("#F8766D", "#D39200", "#93AA00", "#00BA38")) +
  theme(legend.title = element_blank()) -> a



second_frame <- plt_frame %>% filter(data != "all" & Model %in% c("HierarchicalSAE", "MaxSAE", "ConcatSAE", "MeanSAE"))
second_frame$Model <- factor(second_frame$Model, levels = c("HierarchicalSAE", "MeanSAE", "MaxSAE", "ConcatSAE"))

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

second_frame %>% ggplot(
  aes(x = Model, y = concordance, fill = Model)
) +
  geom_boxplot(outlier.size = 0.5, fatten = 1.5) +
  geom_signif(
    annotations = paste0("p = ", round(p_vals$p.val, 7)),
    xmin = p_vals$Model,
    xmax = p_vals$comparison,
    y_position = c(0.975, 1.05, 1.125),
    tip_length = 0,
    textsize = 3.88 / 2.5
  ) +
  coord_cartesian(ylim = c(0.3, 1.2)) +
  scale_x_discrete(labels = NULL, breaks = NULL) +
  theme_bw(base_size = 8) +
  labs(y = "", x = "") +
  theme(legend.position = "bottom") +
  scale_fill_manual(values = c("#F8766D", "#D39200", "#93AA00", "#00BA38")) +
  guides(fill = guide_legend(nrow = 2, byrow = FALSE)) +
  theme(legend.title = element_blank()) -> b

grobs <- ggplotGrob(b)$grobs
legend <- grobs[[which(sapply(grobs, function(x) x$name) == "guide-box")]]

pgrid <- plot_grid(a + theme(legend.position = "none"), b + theme(legend.position = "none"), ncol = 2, labels = "AUTO", label_size = 10, align = "hv")
plt_grid <- plot_grid(pgrid, legend, ncol = 1, rel_heights = c(1, .325))
y.grob <- textGrob("Test concordance",
  gp = gpar(col = "black", fontsize = 8), rot = 90, y = unit(0.625, "npc"), x = unit(1, "npc")
)

x.grob <- textGrob("",
  gp = gpar(col = "black", fontsize = 8)
)

lel <- arrangeGrob(arrangeGrob(plt_grid, left = y.grob))


ggsave(filename = here::here("figures", "fig-2.pdf"), plot = lel, device = "pdf", dpi = 600, height = 6, width = 8.22, units = "cm")
