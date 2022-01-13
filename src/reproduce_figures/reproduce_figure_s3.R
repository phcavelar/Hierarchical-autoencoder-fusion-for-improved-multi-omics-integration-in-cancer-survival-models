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
    (read_csv(here::here("data", "benchmarks", cancer, "hierarchicalsaenet_scores_[-rppa_cnv_meth_mirna_mut]_timed_euler_es_no_hidden_layer.csv"))) %>%
      dplyr::select(model, concordance) %>%
      rename(Model = model) %>%
      mutate(cancer = cancer) %>%
      mutate(Model = "HierarchicalSAE") %>%
      mutate(data = "clinical + gex")
  )

  plt_frame <- rbind(
    plt_frame,
    (read_csv(here::here("data", "benchmarks", cancer, "hierarchicalsaenet_encode_only_scores_[-rppa_cnv_meth_mirna_mut]_timed_euler_es_no_hidden_layer.csv"))) %>%
      dplyr::select(model, concordance) %>%
      rename(Model = model) %>%
      mutate(cancer = cancer) %>%
      mutate(Model = "ConcatSAE") %>%
      mutate(data = "clinical + gex")
  )
}

shae_baseline <- plt_frame %>%
  filter(Model == "HierarchicalSAE") %>%
  summarise(concordance = median(concordance)) %>%
  pull(concordance)
csae_baseline <- plt_frame %>%
  filter(Model != "HierarchicalSAE") %>%
  summarise(concordance = median(concordance)) %>%
  pull(concordance)

plt_frame <- data.frame()


for (cancer in config$cancers) {
  plt_frame <- rbind(
    plt_frame,
    (read_csv(here::here("data", "benchmarks", cancer, "hierarchicalsaenet_encode_only_scores_block_pruned_cka_all_timed_euler_es_fixed.csv"))) %>%
      mutate(cancer = cancer) %>%
      mutate(Model = "HierarchicalSAE (no decoder)")
  )

  plt_frame <- rbind(
    plt_frame,
    (read_csv(here::here("data", "benchmarks", cancer, "hierarchicalsaenet_scores_block_pruned_cka_all_timed_euler_es_fixed.csv"))) %>%
      mutate(cancer = cancer) %>%
      mutate(Model = "HierarchicalSAE")
  )
}

colnames(plt_frame)[1:7] <- paste0("block_pruned_", colnames(plt_frame)[1:7])

df_for_line <- plt_frame %>%
  filter(Model == "HierarchicalSAE") %>%
  pivot_longer(cols = colnames(plt_frame)[1:7]) %>%
  group_by(name) %>%
  summarise(median = median(value))

plt_frame %>%
  filter(Model == "HierarchicalSAE") %>%
  pivot_longer(cols = colnames(plt_frame)[1:7]) %>%
  ggplot(aes(x = name, y = value)) +
  geom_boxplot(outlier.size = 0.5, fatten = 1.5) +
  facet_wrap(~Model) +
  geom_hline(yintercept = shae_baseline, linetype = 3, color = "red", size = 0.35) +
  labs(x = "", y = "") +
  scale_x_discrete(labels = 7:1) +
  theme_bw(base_size = 8) +
  coord_cartesian(ylim = c(0.2, 1.0)) -> a


df_for_line <- plt_frame %>%
  filter(Model != "HierarchicalSAE") %>%
  pivot_longer(cols = colnames(plt_frame)[1:7]) %>%
  group_by(name) %>%
  summarise(median = median(value))

plt_frame %>%
  filter(Model != "HierarchicalSAE") %>%
  pivot_longer(cols = colnames(plt_frame)[1:7]) %>%
  ggplot(aes(x = name, y = value)) +
  geom_boxplot(outlier.size = 0.5, fatten = 1.5) +
  facet_wrap(~Model) +
  geom_hline(yintercept = csae_baseline, linetype = "dashed", color = "red", size = 0.35) +
  labs(x = "", y = "") +
  scale_x_discrete(labels = 7:1) +
  theme_bw(base_size = 8) +
  coord_cartesian(ylim = c(0.2, 1.0)) -> b




plt_frame <- data.frame()

for (cancer in config$cancers) {
  plt_frame <- rbind(
    plt_frame,
    (read_csv(here::here("data", "benchmarks", cancer, "hierarchicalsaenet_encode_only_scores_block_pruned_all_timed_euler_es_fixed.csv"))) %>%
      mutate(cancer = cancer) %>%
      mutate(Model = "HierarchicalSAE (no decoder)")
  )

  plt_frame <- rbind(
    plt_frame,
    (read_csv(here::here("data", "benchmarks", cancer, "hierarchicalsaenet_scores_block_pruned_all_timed_euler_es_fixed.csv"))) %>%
      mutate(cancer = cancer) %>%
      mutate(Model = "HierarchicalSAE")
  )
}

colnames(plt_frame)[1:7] <- paste0("block_pruned_", colnames(plt_frame)[1:7])

df_for_line <- plt_frame %>%
  filter(Model == "HierarchicalSAE") %>%
  pivot_longer(cols = colnames(plt_frame)[1:7]) %>%
  group_by(name) %>%
  summarise(median = median(value))

plt_frame %>%
  filter(Model == "HierarchicalSAE") %>%
  pivot_longer(cols = colnames(plt_frame)[1:7]) %>%
  ggplot(aes(x = name, y = value)) +
  geom_boxplot(outlier.size = 0.5, fatten = 1.5) +
  facet_wrap(~Model) +
  geom_hline(aes(yintercept = shae_baseline, linetype = "Median concordance of HierarchicalSAE trained on clinical + GEX"), color = "red", size = 0.35) +
  labs(x = "", y = "") +
  scale_x_discrete(labels = 7:1) +
  theme_bw(base_size = 8) +
  coord_cartesian(ylim = c(0.2, 1.0)) +
  scale_linetype_manual(name = "", values = c(3), guide = guide_legend(override.aes = list(color = c("red")))) +
  theme(legend.position = "bottom") -> c


df_for_line <- plt_frame %>%
  filter(Model != "HierarchicalSAE") %>%
  pivot_longer(cols = colnames(plt_frame)[1:7]) %>%
  group_by(name) %>%
  summarise(median = median(value))

plt_frame %>%
  filter(Model != "HierarchicalSAE") %>%
  pivot_longer(cols = colnames(plt_frame)[1:7]) %>%
  ggplot(aes(x = name, y = value)) +
  geom_boxplot(outlier.size = 0.5, fatten = 1.5) +
  facet_wrap(~Model) +
  geom_hline(aes(yintercept = csae_baseline, linetype = "Median concordance of HierarchicalSAE (no decoder) trained on clinical + GEX"), color = "red", size = 0.35) +
  labs(x = "", y = "") +
  scale_x_discrete(labels = 7:1) +
  theme_bw(base_size = 8) +
  coord_cartesian(ylim = c(0.2, 1.0)) +
  scale_linetype_manual(name = "", values = c(2), guide = guide_legend(override.aes = list(color = c("red")))) +
  theme(legend.position = "bottom") -> d


library(cowplot)
pgrid_1 <- plot_grid(a, b, align = "hv")
pgrid_2 <- plot_grid(c + theme(legend.position = "none"), d + theme(legend.position = "none"), align = "hv")
pgrid <- plot_grid(pgrid_1, pgrid_2, ncol = 2, labels = "AUTO", label_size = 10, align = "hv")
legend_one <- get_legend(
  c + theme(legend.box.margin = margin(0, 0, 0, 0))
)

legend_two <- get_legend(
  d + theme(legend.box.margin = margin(0, 0, 0, 0))
)

pgrid_3 <- plot_grid(legend_one, legend_two, rel_widths = c(1, 1.5))

pgrid <- plot_grid(pgrid, pgrid_3, rel_heights = c(3, .3), nrow = 2, ncol = 1)


pgrid
library(grid)
library(gridExtra)

y.grob <- textGrob("Test concordance",
  gp = gpar(col = "black", fontsize = 8), rot = 90, y = unit(0.625, "npc"), x = unit(0.6, "npc")
)

x.grob <- textGrob("Number of modalities remaining",
  gp = gpar(col = "black", fontsize = 8), y = unit(0.6, "npc")
)

lel <- arrangeGrob(arrangeGrob(pgrid, left = y.grob, bottom = x.grob))


ggsave(filename = here::here("figures", "fig-s3.pdf"), plot = lel, device = "pdf", dpi = 600, height = 5, width = 17.07852, units = "cm")
