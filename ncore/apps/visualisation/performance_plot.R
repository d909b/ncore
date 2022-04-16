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
  "NCoRE"=palette[8],
  "TARNET\n(balanced)"=palette[6])

options(stringsAsFactors=FALSE)
set.seed(909)

args <- commandArgs(trailingOnly=TRUE)

results <- args[1]
output_directory <- args[2]
file_name <- args[3]
plot_title <- args[4]
compare_to <- args[5]
y_axis_label <- args[6]
compare_from <- args[7]

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
name_map[["BalancedTARNET"]] = "TARNET\n(balanced)"
name_map[["Deconfounder"]] = "Deconfounder"
name_map[["MTVAE"]] = "MTVAE"

color_map <- new.env(hash = TRUE)
color_map[["CounterfactualRelationEstimator"]] = "0"
color_map[["CounterfactualRelationEstimatorNoMixing"]] = "0"
color_map[["GANITE"]] = "1"
color_map[["GaussianProcess"]] = "2"
color_map[["KNearestNeighbours"]] = "3"
color_map[["LinearRegression"]] = "4"
color_map[["TARNET"]] = "5"
color_map[["Deconfounder"]] = "7"
color_map[["BalancedCounterfactualRelationEstimator"]] = "8"
color_map[["BalancedTARNET"]] = "9"
color_map[["MTVAE"]] = "10"

min_values <- new.env(hash = TRUE)
max_values <- new.env(hash = TRUE)
median_values <- new.env(hash = TRUE)
results_names <- names(results)
for(i in 1:length(results_names)) {
  name <- results_names[i]
  tables <- tables %>% rbind(c(results[[name]], name, name_map[[name]]))
  min_values[[name]] = quantile(unlist(results[[name]]), 0.025)
  max_values[[name]] = quantile(unlist(results[[name]]), 0.975)
  median_values[[name]] = quantile(unlist(results[[name]]), 0.5)
}

rownames(tables) <- results_names
colnames(tables) <- c(seq(1:100), "names", "converted_names")
collapsed <- melt(tables, id.vars=c("names", "converted_names"))

collapsed$min <- lapply(collapsed$names, function(x) min_values[[x]])
collapsed$max <- lapply(collapsed$names, function(x) max_values[[x]])
collapsed$median <- lapply(collapsed$names, function(x) median_values[[x]])

collapsed$color <- factor(collapsed$converted_names,
                          levels=c("GANITE", "GP", "kNN", "Ridge", "TARNET", "Deconfounder", "NCoRE", "NCoRE\n(balanced)",
                                   "BalancedCounterfactualRelationEstimator", "BalancedTARNET", "MTVAE"),
                          labels=c("GANITE", "GP", "kNN", "Ridge", "TARNET", "Deconfounder", "NCoRE", "NCoRE\n(balanced)",
                                   "BalancedCounterfactualRelationEstimator", "BalancedTARNET", "MTVAE"))

pdf(file.path(output_directory, file_name), width=5, height=3.6)

p <- ggplot(collapsed, aes(x=reorder(converted_names, as.numeric(median)), y=as.numeric(median), fill=color)) +
  labs(y=y_axis_label, title=title) +
  geom_bar(stat="identity", position=position_dodge(), colour="black", width=0.5) +
  geom_jitter(aes(x=reorder(converted_names, as.numeric(median)),
                  y=as.numeric(value),
                  colour=as.factor(color)), size=0.1, alpha=0.4, shape=21, colour="black",
              position=position_jitterdodge(dodge.width=0.9, jitter.width=0.25), notch=TRUE) +
  geom_signif(comparisons = list(c(compare_from, compare_to)), textsize=2.45) +
  theme(axis.title = element_text(size=14),
        axis.text = element_text(size=12, color="black"),
        axis.title.x = element_blank(),
        axis.text.x = element_text(angle = 90, hjust = 1),
        panel.background = element_rect(fill = "transparent") # bg of the panel
        , plot.background = element_rect(fill = "transparent", color = NA) # bg of the plot
        , panel.grid.major.x = element_line(colour = "black", linetype="dotted") # get rid of major grid
        , panel.grid.minor.x = element_line(colour = "black", linetype="dotted") # get rid of minor grid
        , panel.grid.major.y = element_line(colour = "black", linetype="dotted") # get rid of major grid
        , legend.position="none"
        , legend.background = element_rect(fill = "transparent") # get rid of legend bg
        , legend.title = element_blank()
        , legend.box.background = element_rect(fill = "transparent")
        , axis.line = element_line(colour = "black")
        , plot.title = element_text(hjust = 0.5, size=20, face="bold")
  ) + scale_fill_manual(values = palette) + scale_color_manual(values = palette)
print(p)

dev.off()