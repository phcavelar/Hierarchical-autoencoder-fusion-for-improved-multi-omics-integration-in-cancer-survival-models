mean(c(0.6774193548387096,
0.6602209944751382,
0.6776094276094277,
0.6142595978062158,
0.6255707762557078,
0.7049808429118773,
0.716789667896679,
0.6018755328218244,
0.6328273244781784,
0.625349487418453))

median(c(0.6965725806451613,
0.6602209944751382,
0.6708754208754208,
0.6352833638025595,
0.6292237442922375,
0.7337164750957854,
0.7214022140221402,
0.5890878090366581,
0.6489563567362429,
0.6197576887232059))

library(ggplot2)
library(cowplot)
library(magick)
fig_last <- ggdraw() + draw_image(magick::image_read_pdf(here::here("~/Downloads/fig-1a.pdf"), density = 300))
fig_2 <- ggdraw() + draw_image(magick::image_read_pdf(here::here("~/Downloads/fig-1b.pdf"), density = 300))
fig_3 <- ggdraw() + draw_image(magick::image_read_pdf(here::here("~/Downloads/fig-1c.pdf"), density = 300))
fig_4 <- ggdraw() + draw_image(magick::image_read_pdf(here::here("~/Downloads/fig-1d.pdf"), density = 300))
fig_5 <- ggdraw() + draw_image(magick::image_read_pdf(here::here("~/Downloads/fig-1e.pdf"), density = 300))

#ggsave("output.png", fig, width = 1, height = .7, dpi = 1200)



pgrid_1 <- plot_grid(fig_2, fig_3, ncol = 2, labels = c("B", "C"), label_size = 10, align = "hv")
pgrid_2 <- plot_grid(fig_4, fig_5, ncol = 2, labels = c("D", "E"), label_size = 10, align = "hv")
pgrid_3 <- plot_grid(pgrid_1, pgrid_2, nrow = 2, align = "hv")

plot_grid(fig_last, pgrid_3, ncol = 2, labels = c("A"), label_size = 10, align = "hv")

#pgrid <- plot_grid(fig_2, fig_3, fig_4, fig_5, ncol = 4, nrow=1, labels=c("B", "C", "D", "E"), label_size = 10, align = "hv")

ggsave(here::here("~", "Desktop", "fig-1.pdf"), device = "pdf",
       width=17.07852,
       #width=14,
       
       height=6.5, units = "cm")

#pgrid_2 <- plot_grid(fig_last, fig_2, ncol = 2, nrow=1, labels=c("A", "B"), label_size = 10, align = "hv", rel_widths = c(2, 1))

plot_grid(fig_last, pgrid, nrow = 2, ncol = 1, labels = c("A", "B"), label_size = 10, align = "hv", rel_heights = c(2, 1))
#pgrid

ggsave(here::here("~", "Desktop", "fig-1.pdf"), device = "pdf",
       width=17.07852,
       #width=14,
       
       height=8, units = "cm")


fig_1 <- ggdraw() + draw_image(magick::image_read_pdf(here::here("~/Downloads/fig-s1a.pdf"), density = 300))
fig_2 <- ggdraw() + draw_image(magick::image_read_pdf(here::here("~/Downloads/fig-s1b.pdf"), density = 300))
fig_3 <- ggdraw() + draw_image(magick::image_read_pdf(here::here("~/Downloads/fig-s1c.pdf"), density = 300))

#ggsave("output.png", fig, width = 1, height = .7, dpi = 1200)

pgrid <- plot_grid(fig_1, fig_2, fig_3, ncol = 3, nrow=1, labels=c("A", "B", "C"), label_size = 10, align = "hv")

#pgrid_2 <- plot_grid(fig_last, fig_2, ncol = 2, nrow=1, labels=c("A", "B"), label_size = 10, align = "hv", rel_widths = c(2, 1))

plot_grid(fig_last, pgrid, nrow = 2, ncol = 1, labels = c("A", "B"), label_size = 10, align = "hv", rel_heights = c(2, 1))
#pgrid

ggsave(here::here("~", "Desktop", "fig-s1.pdf"), device = "pdf",
       width=17.07852,
       #width=14,
       
       height=4, units = "cm")

fig_1 <- ggdraw() + draw_image(magick::image_read_pdf(here::here("~/Downloads/fig-s2a.pdf"), density = 300))
fig_2 <- ggdraw() + draw_image(magick::image_read_pdf(here::here("~/Downloads/fig-s2b.pdf"), density = 300))
fig_3 <- ggdraw() + draw_image(magick::image_read_pdf(here::here("~/Downloads/fig-s2c.pdf"), density = 300))

#ggsave("output.png", fig, width = 1, height = .7, dpi = 1200)

pgrid <- plot_grid(fig_1, fig_2, fig_3, ncol = 3, nrow=1, labels=c("A", "B", "C"), label_size = 10, align = "hv")

#pgrid_2 <- plot_grid(fig_last, fig_2, ncol = 2, nrow=1, labels=c("A", "B"), label_size = 10, align = "hv", rel_widths = c(2, 1))

ggsave(here::here("~", "Desktop", "fig-s2.pdf"), device = "pdf",
       width=17.07852,
       #width=14,
       
       height=4, units = "cm")

fig_1 <- ggdraw() + draw_image(magick::image_read_pdf(here::here("~/Downloads/fig-s4a.pdf"), density = 300))
fig_2 <- ggdraw() + draw_image(magick::image_read_pdf(here::here("~/Downloads/fig-s4b.pdf"), density = 300))

#ggsave("output.png", fig, width = 1, height = .7, dpi = 1200)

pgrid <- plot_grid(fig_1, fig_2, ncol = 2, nrow=1, labels=c("A", "B"), label_size = 10, align = "hv")

ggsave(here::here("~", "Desktop", "fig-s4.pdf"), device = "pdf",
       width=17.07852,
       #width=14,
       
       height=4, units = "cm")
