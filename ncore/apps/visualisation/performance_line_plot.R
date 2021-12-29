# Copyright (C) 2021  Patrick Schwab, GlaxoSmithKline plc, Sonali Parbhoo, Harvard University
# Copyright (C) 2020  Patrick Schwab, F. Hoffmann-La Roche Ltd
# Copyright (C) 2019  Patrick Schwab, ETH Zurich
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions
#  of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED
# TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
# CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
suppressWarnings(suppressMessages(library(ggplot2)))
suppressWarnings(suppressMessages(library(ggsignif)))
suppressWarnings(suppressMessages(library(tibble)))
suppressWarnings(suppressMessages(library(latex2exp)))
suppressWarnings(suppressMessages(library(dplyr)))
suppressWarnings(suppressMessages(library(readr)))
suppressWarnings(suppressMessages(library(reshape2)))
suppressWarnings(suppressMessages(library(ggsci)))
suppressWarnings(suppressMessages(library(hash)))
suppressWarnings(suppressMessages(library(RColorBrewer)))

palette <- brewer.pal(8, "Dark2")
palette[1] <- "#333333"
palette <- c(
  "NCoRE\n(balanced)"=palette[1],
  "GANITE"=palette[2],
  "GP"=palette[3],
  "kNN"=palette[4],
  "Ridge"=palette[5],
  "TARNET"=palette[6],
  "Deconfounder"=palette[7],
  "NCoRE"=palette[8]
)

symbol_palette <- c(
  "NCoRE"=0,
  "GANITE"=1,
  "GP"=2,
  "kNN"=3,
  "Ridge"=4,
  "TARNET"=5,
  "Deconfounder"=6,
  "NCoRE\n(balanced)"=7
)

options(stringsAsFactors=FALSE)
set.seed(909)

args <- commandArgs(trailingOnly=TRUE)

results <- args[1]
output_directory <- args[2]
file_name <- args[3]
plot_title <- args[4]
x_axis_name <- args[5]
skip_ganite <- ifelse(args[6] == "true", T, F)
with_legend <- ifelse(args[7] == "true", T, F)

results <- eval(parse(text=read_file(results)))

tables <- tibble()
title <- plot_title

name_map <- new.env(hash = TRUE)
name_map[["BalancedCounterfactualRelationEstimator"]] = "NCoRE\n(balanced)"
name_map[["CounterfactualRelationEstimator"]] = "NCoRE"
name_map[["CounterfactualRelationEstimatorNoMixing"]] = "NCoRE"
name_map[["GANITE"]] = "GANITE"
name_map[["GaussianProcess"]] = "GP"
name_map[["KNearestNeighbours"]] = "kNN"
name_map[["LinearRegression"]] = "Ridge"
name_map[["TARNET"]] = "TARNET"
name_map[["Deconfounder"]] = "Deconfounder"


min_values <- new.env(hash = TRUE)
max_values <- new.env(hash = TRUE)
median_values <- new.env(hash = TRUE)
results_names <- names(results)
for(i in 1:length(results_names)) {
  name <- results_names[i]
  if(name=="GANITE" && skip_ganite) {
    next
  }
  subnames <- names(results[[name]])
  for(j in 1:length(subnames)) {
    subname <- subnames[j]
    values <- results[[name]][[subname]]
    tables <- tables %>% rbind(c(values, name,
                                 name_map[[name]], as.numeric(subname)))
  }
}

colnames(tables) <- c(seq(1:100), "names", "converted_names", "time")
collapsed <- melt(tables, id.vars=c("names", "converted_names", "time"))

collapsed$color <- factor(collapsed$converted_names,
                          levels=c("GANITE", "GP", "kNN", "Ridge", "TARNET", "Deconfounder", "NCoRE", "NCoRE\n(balanced)"),
                          labels=c("GANITE", "GP", "kNN", "Ridge", "TARNET", "Deconfounder", "NCoRE", "NCoRE\n(balanced)"))

pdf(file.path(output_directory, file_name), width=ifelse(with_legend, 5, 4.15), height=3.6)

p <- ggplot(collapsed, aes(x=time, y=as.numeric(value), fill=color)) +
  labs(y = 'RMSE', x=TeX(x_axis_name), title=title) +
  stat_summary(geom="ribbon", fun.min=function(z) quantile(z, 0.025),
               fun.max=function(z) quantile(z, 0.975), aes(fill=color),
               alpha=0.25) +
  stat_summary(fun = median, geom="line", aes(color=color)) +
  stat_summary(fun = median, geom="point", aes(color=color, shape=color)) +
  theme(axis.title = element_text(size=14),
        axis.text = element_text(size=12, color="black"),
        axis.text.x = element_text(hjust = 1),
        panel.background = element_rect(fill = "transparent") # bg of the panel
        , plot.background = element_rect(fill = "transparent", color = NA) # bg of the plot
        , panel.grid.major.x = element_line(colour = "black", linetype="dotted") # get rid of major grid
        , panel.grid.minor.x = element_line(colour = "black", linetype="dotted") # get rid of minor grid
        , panel.grid.major.y = element_line(colour = "black", linetype="dotted") # get rid of major grid
        , legend.position=ifelse(with_legend, "right", "none")
        , legend.background = element_rect(fill = "transparent") # get rid of legend bg
        , legend.title = element_blank()
        , legend.box.background = element_rect(fill = "transparent")
        , axis.line = element_line(colour = "black")
        , plot.title = element_text(hjust = 0.5, size=20, face="bold")
  ) + scale_fill_manual(values = palette) + scale_color_manual(values = palette) +
  scale_shape_manual(values=symbol_palette)

print(p)
dev.off()
