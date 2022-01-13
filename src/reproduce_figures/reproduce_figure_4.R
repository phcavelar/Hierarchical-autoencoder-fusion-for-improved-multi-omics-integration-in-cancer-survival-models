library(here)
library(rjson)
library(readr)
library(dplyr)
library(ggplot2)
library(tidyr)
library(ggsignif)
library(ComplexHeatmap)
library(cowplot)

overall <- vroom::vroom(
  here::here(
    # "data", "cka", "BLCA", "shae_shae_encode_only_cka_similarity.csv"
    "data", "cka", "BLCA", "hierarchicalsaenet_hierarchicalsaenet_encode_only_cka_similarity.csv"
  )
) %>%
  data.frame() %>%
  as.matrix()

colnames(overall) <- c("Clinical", "GEX", "CNV", "Methylation", "miRNA", "Mutation", "RPPA", "Final")
rownames(overall) <- c("Clinical", "GEX", "CNV", "Methylation", "miRNA", "Mutation", "RPPA", "Final")
i <- Heatmap(overall[nrow(overall):1, ],
  show_row_names = TRUE, cluster_rows = FALSE, cluster_columns = FALSE, show_column_names = TRUE, col = col_fun, name = " ", row_names_side = "left", column_title_side = "bottom", show_heatmap_legend = TRUE, row_names_gp = gpar(fontsize = 6), column_title = "HierarchicalSAE (no decoder)", row_title = "HierarchicalSAE",
  column_names_gp = gpar(fontsize = 6),
  column_title_gp = gpar(fontsize = 6),
  row_title_gp = gpar(fontsize = 6),
  heatmap_legend_param = list(
    legend_height = unit(2.5, "cm"),
    legend_width = unit(0.0, "cm"),
    labels_gp = gpar(fontsize = 6)
  )
)

overall <- vroom::vroom(
  here::here(
    "data", "cka", "SARC", "hierarchicalsaenet_hierarchicalsaenet_encode_only_cka_similarity.csv"
  )
) %>%
  data.frame() %>%
  as.matrix()

colnames(overall) <- c("Clinical", "GEX", "CNV", "Methylation", "miRNA", "Mutation", "RPPA", "Final")
rownames(overall) <- c("Clinical", "GEX", "CNV", "Methylation", "miRNA", "Mutation", "RPPA", "Final")
j <- Heatmap(overall[nrow(overall):1, ],
  show_row_names = TRUE, cluster_rows = FALSE, cluster_columns = FALSE, show_column_names = TRUE, col = col_fun, name = " ", row_names_side = "left", column_title_side = "bottom", show_heatmap_legend = TRUE, row_names_gp = gpar(fontsize = 6), column_title = "HierarchicalSAE (no decoder)", row_title = "HierarchicalSAE",
  column_names_gp = gpar(fontsize = 6),
  column_title_gp = gpar(fontsize = 6),
  row_title_gp = gpar(fontsize = 6),
  heatmap_legend_param = list(
    legend_height = unit(2.5, "cm"),
    legend_width = unit(0.0, "cm"),
    labels_gp = gpar(fontsize = 6)
  )
)

plot_grid(
  grid.grabExpr(draw(i)),
  grid.grabExpr(draw(j)),
  ncol = 2, labels = LETTERS[1:2], label_size = 10, align = "hv"
)

ggsave(filename = here::here("figures", "fig-4.pdf"), device = "pdf", dpi = 600, height = 5, width = 17.07852, units = "cm")
