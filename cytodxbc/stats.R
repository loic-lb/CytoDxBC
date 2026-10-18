library(dplyr)
library(ggplot2)
library("pROC")
library(caret)
library("tidyr")
library(stringr)
library(rms)
library(cartographr)

# Results directory of a cross-validation experiment, as written by script_testing.py
# and script_interpretation_crossval.py
results_dir <- "./experiments/CytoDxBC"
predictions <- read.csv(file.path(results_dir, "predictions_detailed_calibrated.csv"))


metrics_data <- data.frame(fold = character(),
                           sensitivity = numeric(),
                           specificity = numeric(),
                           ppv = numeric(),
                           npv = numeric(),
                           #f1 = numeric(),
                           balanced_accuracy = numeric(),
                           stringsAsFactors = FALSE)

splits <- split(predictions, predictions$split_id)
# Loop through each fold and calculate metrics
for (i in seq_along(splits)) {
  fold_data <- splits[[i]]
  fold_data$pred <- factor(as.integer(factor(fold_data$pred))-1, levels = c(0,1))
  fold_data$label <- factor(as.integer(factor(fold_data$label))-1, levels = c(0,1))
  confusion_matrix <- confusionMatrix(data=as.factor(fold_data$pred), reference = as.factor(fold_data$label), positive="1")

  # Extract metrics
  sensitivity <- confusion_matrix$byClass['Sensitivity']
  specificity <- confusion_matrix$byClass['Specificity']
  ppv <- confusion_matrix$byClass['Pos Pred Value']
  npv <- confusion_matrix$byClass['Neg Pred Value']
  #f1 <- 2 * (ppv * sensitivity) / (ppv + sensitivity)  # F1 is the harmonic mean of precision and recall
  balanced_accuracy <- (sensitivity + specificity) / 2

  # Append to dataframe
  metrics_data <- rbind(metrics_data, data.frame(fold = paste("Fold", i),
                                                 sensitivity = sensitivity,
                                                 specificity = specificity,
                                                 ppv = ppv,
                                                 npv = npv,
                                                 #f1 = f1,
                                                 balanced_accuracy = balanced_accuracy))
}

# Reshape data for plotting
metrics_long <- pivot_longer(metrics_data, cols = -fold, names_to = "metric", values_to = "value")

rename_map <- c(
  sensitivity = "Sensitivity",
  specificity = "Specificity",
  ppv = "PPV",
  npv = "NPV",
  balanced_accuracy = "Balanced Accuracy"
)

# Apply renaming in the dataframe
metrics_long$metric <- factor(metrics_long$metric, levels = names(rename_map))
metrics_long$metric <- rename_map[metrics_long$metric]

new_order <- c(
  "Sensitivity",
  "Specificity",
  "PPV",
  "NPV",
  "F1 Score",
  "Balanced Accuracy"
)

# Apply the new order
metrics_long$metric <- factor(metrics_long$metric, levels = new_order)

percent_without_symbol <- function(x) {
  sprintf("%.0f", 100 * x)  # Convert proportion to number out of 100
}


ggplot(metrics_long, aes(x = metric, y = value, fill = metric)) +
  geom_col(show.legend = FALSE) +
  #scale_y_continuous(labels = percent_without_symbol) +
  labs(y = "Value",
       x="",) +
  theme_poster() +
  theme(
    panel.grid.major = element_line(color = "black", size = 0.25),   # Major grid lines
    panel.grid.minor = element_line(color = "gray", size = 0.125), # Minor grid lines
    panel.grid.major.x = element_blank(),  # Major X grid lines
    panel.grid.minor.x = element_blank(), # Minor X grid lines,
    axis.title = element_text(size = 32),
    axis.text = element_text(size = 28)
  ) +
  #theme(text = element_text(size=30)) +
  scale_fill_brewer(palette="Pastel1")


summary_stats <- metrics_long %>%
  group_by(metric) %>%
  summarise(
    Mean = mean(value),
    SE = sd(value) / sqrt(n()),  # Standard Error
    LowerCI = Mean - SE * 1.96,  # 95% CI lower bound
    UpperCI = Mean + SE * 1.96   # 95% CI upper bound
  )

ggplot(summary_stats, aes(x = metric, y = Mean, fill = metric)) +
  geom_col(show.legend = FALSE) +
  scale_y_continuous(labels = percent_without_symbol) +
  geom_errorbar(aes(ymin = LowerCI, ymax = UpperCI), width = 0.2) +
  labs(y = "Value",
       x="",) +
  theme_poster()+
  theme(
    panel.grid.major = element_line(color = "black", size = 0.25),   # Major grid lines
    panel.grid.minor = element_line(color = "gray", size = 0.125), # Minor grid lines
    panel.grid.major.x = element_blank(),  # Major X grid lines
    panel.grid.minor.x = element_blank(), # Minor X grid lines
    axis.title = element_text(size = 36),
    axis.text = element_text(size = 32)
  ) +
  scale_fill_brewer(palette="Pastel1")


cat_df <- read.csv("./category_lesions.csv")   # CSV with slide_id and dg columns
a = predictions[predictions$split_id==9,]
confusion_table <- confusionMatrix(data=as.factor(a$pred), reference = as.factor(a$label), positive="malignant")
confusion_table

grp1 = cat_df[cat_df$dg %in% c("solid", "inflammatory"),]$slide_id
grp2 = cat_df[cat_df$dg %in% c("cystic", "mesechymal"),]$slide_id

acc_grp1 = c()
acc_grp2 = c()
for (i in 0:9){
  split_df = predictions[predictions$split_id==i,]
  split_df_grp1 = split_df[split_df$slide_id %in% grp1,]
  acc_grp1 = append(acc_grp1, 1 - sum(split_df_grp1$label==split_df_grp1$pred) / nrow(split_df_grp1))
  split_df_grp2 = split_df[split_df$slide_id %in% grp2,]
  acc_grp2 = append(acc_grp2, 1 - sum(split_df_grp2$label==split_df_grp2$pred) / nrow(split_df_grp2))
}


result =  wilcox.test(acc_grp1, acc_grp2, alternative = "great")$p.value
df_p_val <- data.frame(
  group1 = "1",
  group2 = "2",
  label = result,
  y.position = 0.4
)


results <- read.csv(file.path(results_dir, "results.csv"))
sample.mean <- mean(results$PPV)
print(round(sample.mean, 4))
sample.n <- 10
sample.sd <- sd(results$PPV)
sample.se <- sample.sd/sqrt(sample.n)

alpha = 0.05
degrees.freedom = sample.n - 1
t.score = qt(p=alpha/2, df=degrees.freedom,lower.tail=F)
margin.error <- t.score * sample.se
lower.bound <- sample.mean - margin.error
upper.bound <- sample.mean + margin.error
print(c(round(lower.bound, 4), round(upper.bound, 4)))


# predictions_detailed_calibrated.csv already contains the lesion category (dg) of each slide
data_merged <- predictions %>%
  mutate(label_num = as.integer(factor(label)) - 1) %>%   # benign -> 0, malignant -> 1
  mutate(grouped_category = case_when(
    dg %in% c("mesechymal", "cystic") ~ "Mesechymal/Cystic",
    dg %in% c("solid", "inflammatory") ~ "Solid/Inflammatory",
  ))

auc_data <- data_merged %>%
  group_by(dg) %>%
  summarise(AUC = as.numeric(auc(as.factor(label_num), pred_prob_malignant)))

accuracy_data <- data_merged[data_merged$label=="benign",] %>%
  group_by(grouped_category, split_id) %>%
  summarise(Accuracy = mean(pred == label, na.rm = TRUE)) %>%
  ungroup()

ggplot(accuracy_data, aes(x = grouped_category, y = Accuracy, fill = grouped_category)) +
  geom_boxplot() +
  labs(title = "Accuracy Distribution by Lesion Category",
       x = "Lesion Category",
       y = "Accuracy") +
  theme_minimal()

data_merged$error <- ifelse(data_merged$pred == data_merged$label, 0, 1)
error_rates <- data_merged %>%
  group_by(dg) %>%
  summarise(ErrorRate = mean(error))
print(error_rates)

summary_data <- data_merged %>%
  group_by(grouped_category) %>%
  filter(!any(is.na(grouped_category))) %>%
  summarise(
    MeanErrorRate = mean(error)
  )

grouped_errors <- data_merged %>%
  group_by(grouped_category) %>%
  filter(!any(is.na(grouped_category))) %>%
  summarise(MeanErrorRate = mean(pred != label, na.rm = TRUE),
            .groups = 'drop')

ggplot(error_rates, aes(x = dg, y = ErrorRate, fill = dg)) +
  geom_bar(stat = "identity", show.legend = FALSE) +
  labs(x = "",
       y = "Mean Error Rate") +
  theme_minimal()

data_catA <- filter(data_merged, grouped_category == "Solid/Inflammatory")
data_catB <- filter(data_merged, grouped_category == "Mesechymal/Cystic")

wilcox_test <- wilcox.test(data_catA$error, data_catB$error, paired = FALSE, alternative="great")
p_value <- wilcox_test$p.value

error_plot <- ggplot(grouped_errors, aes(x = grouped_category, y = MeanErrorRate, fill = grouped_category)) +
  geom_bar(stat = "identity", show.legend = FALSE) +
  #geom_text(aes(label = sprintf("Wilcoxon test, p = %.3f", p_value)), x = 1.5, y = max(grouped_errors$MeanErrorRate), vjust = -2) +
  labs(#title = "Mean Error Rate by Lesi Category",
       x = "",
       y = "Mean Error Rate") +

  theme_minimal() +
  theme(text = element_text(size = 30))
  #theme(axis.text.x = element_text(angle = 0, hjust = 1))
print(error_plot)
# Find maximum y-value for positioning
max_errors <- grouped_errors %>%
  group_by(grouped_category) %>%
  summarise(MaxMeanErrorRate = max(MeanErrorRate))

# Viewing the result
print(max_errors)

max_y <- max(grouped_errors$MeanErrorRate)

error_plot <- error_plot +
  annotate("segment", x = 1, xend = 1, y = max_y, yend = max_y + 0.02, colour = "black") +   # Left vertical line of the bracket
  annotate("segment", x = 2, xend = 2, y = max_y, yend = max_y + 0.02, colour = "black") +   # Right vertical line of the bracket
  annotate("segment", x = 1, xend = 2, y = max_y + 0.02, yend = max_y + 0.02, colour = "black") +  # Connecting line of the bracket
  annotate("text", x = 1.5, y = max_y + 0.03, label = sprintf("p = %.3f", p_value), vjust = -1)  # P-value text

print(error_plot)

error_plot <- ggplot(summary_data, aes(x = grouped_category, y = MeanErrorRate, fill = grouped_category)) +
  geom_bar(stat = "density") +
  labs(title = "Mean Error Rate by Lesion Category",
       x = "Lesion Category",
       y = "Mean Error Rate") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))



# predictions_val.csv written by script_testing.py --validation for the model to calibrate
predictions_val = read.csv(file.path(results_dir, "predictions_val.csv"))

platt_models <- list()

splits <- split(predictions_val, predictions_val$split_id)
# Loop over each split
for(i in names(splits)) {
  fold_data <- splits[[i]]

  # Platt Scaling
  platt_model <- glm(label ~ pred_prob_malignant, data=fold_data, family=binomial(link="logit"))
  platt_models[[i]] <- platt_model  # Store the model

}

calibrated_probs_platt <- lapply(unique(predictions_val$split_id), function(i) {
  predict(platt_models[[i+1]], splits[[i+1]], type="response")
})

for(i in unique(predictions_val$split_id)) {
  splits[[i+1]]$calibrated_prob_platt <- calibrated_probs_platt[[i+1]]
}

# Recombine all data
calibrated_data <- do.call(rbind, splits)

ggplot(calibrated_data, aes(x=factor(label), y=pred_prob_malignant, fill=factor(label))) +
  geom_density(alpha=0.5) +
  labs(title="Before Calibration") +
  facet_wrap(~split_id)

ggplot(calibrated_data, aes(x=factor(label), y=calibrated_prob_platt, fill=factor(label))) +
  geom_density(alpha=0.5) +
  labs(title="After Calibration (Platt)") +
  facet_wrap(~split_id)

data <- predictions %>%
  mutate(grouped_category = case_when(
    category %in% c("mesechymal", "cystic") ~ "Mesechymal/Cystic",
    category %in% c("solid", "inflammatory") ~ "Solid/Inflammatory",
  ))

predictions$category <- gsub("mesechymal", "mesenchymal", predictions$category)
data <- predictions %>%
  mutate(grouped_category = case_when(
    category %in% c("mucinous carcinoma", "papillary carcinoma") ~ "other\ncarcinomas",
    TRUE ~ as.character(category)
  ))

group_colors <- c("cystic" = "#AAD2CF",
                  "mesenchymal" = "#AAD2CF",
                  "inflammatory" = "#AAD2CF",
                  "solid" = "#AAD2CF",
                  "carcinoma" = "#F8BB9E",
                  "other carcinomas"= "#F8BB9E")

group_colors <- c("Benign" = "#AAD2CF", "Malignant"= "#F8BB9E")

data$log_prob <- log(data$pred_prob / (1 - data$pred_prob))
data$grouped_category = factor(data$grouped_category, levels=c("cystic", "inflammatory", "mesenchymal", "solid", "carcinoma", "other\ncarcinomas"))
data$grouped_category <- factor(data$grouped_category, labels = str_to_title(levels(data$grouped_category)))
data$label <- factor(data$label, levels=c("benign", "malignant"))
data$label <- factor(data$label, labels = str_to_title(levels(data$label)))
ggplot(data, aes(x = grouped_category, y=log_prob, fill = label)) +
  geom_boxplot(show.legend = TRUE) +
  scale_fill_manual(values = group_colors) +
  labs( x = "",
       y = "Confidence score",) +
  theme_poster() +
  theme(
    panel.grid.major = element_line(color = "black", size = 0.25),   # Major grid lines
    panel.grid.minor = element_line(color = "gray", size = 0.125), # Minor grid lines
    panel.grid.major.x = element_blank(),  # Major X grid lines
    panel.grid.minor.x = element_blank(), # Minor X grid lines
    axis.title = element_text(size = 36),
    axis.text = element_text(size = 32)
  ) +
  scale_y_continuous(limits = c(0.8, 16)) +
  theme(legend.position = "bottom", legend.title=element_blank(),
        legend.text = element_text(size=32)) +  # Legend below the plot
  guides(fill = guide_legend(override.aes = list(linetype = 0, shape = NA)))  # Color blocks only in legend

  #theme(text = element_text(size=30))

ggplot(data, aes(x=pred_prob_malignant)) +
  geom_histogram(bins=30, fill="blue", color="black") +
  ggtitle("Histogram of Calibrated Probabilities")


data <- predictions
data$true_label <- factor(as.integer(factor(predictions$label))-1, levels = c(0,1))
lrm_model <- lrm(true_label ~ pred_prob_calibrated_malignant, data = data, x = TRUE, y = TRUE)
cal_obj <- calibrate(lrm_model, method = "boot", B = 1000)  # B is the number of bootstrap resamples
plot(cal_obj)

predictions$true_label <- factor(as.integer(factor(predictions$label))-1, levels = c(0,1))
roc_list <- lapply(split(predictions, predictions$split_id), function(df) {
  roc(df$true_label, df$pred_prob_malignant)
})

# Plot the first ROC curve to set up the plot
plot(roc_list[[2]], main = "ROC Curves for Each CV Fold", col = 1)

# Add the rest of the ROC curves
if (length(roc_list) > 1) {
  sapply(2:length(roc_list), function(i) {
    plot(roc_list[[i]], add = TRUE, col = i)
  })
}

roc_dfs <- lapply(names(roc_list), function(fold) {
  coords <- coords(roc_list[[fold]], "all", ret = c("sensitivity", "specificity", "threshold"))
  data.frame(
    TPR = coords$sensitivity,
    FPR = coords$specificity,
    Fold = fold
  )
})
roc_df <- do.call(rbind, roc_dfs)

# Plot using ggplot2
ggplot(roc_df, aes(x = FPR, y = TPR, color = Fold)) +
  geom_line(size = 1.2) +
  scale_color_brewer(palette = "Set1") +  # Set a color palette to help differentiate the folds
  #scale_linetype_manual(values = c(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)) +  # Different line types for each fold
  labs(x = "False Positive Rate", y = "True Positive Rate", title = "ROC Curves for Each CV Fold") +
  theme_minimal() +
  theme(legend.title = element_blank(),
        plot.title = element_text(hjust = 0.5),
        axis.title = element_text(size = 12),
        legend.text = element_text(size = 10))




tprs <- list()
aucs <- vector()
mean_fpr <- seq(0, 1, length.out = 100)

for(fold in unique(predictions$split_id)) {
  fold_data <- predictions[predictions$split_id == fold, ]
  fold <- fold + 1

  roc_result <- roc(fold_data$true_label, fold_data$pred_prob_malignant)
  aucs[fold] <- auc(roc_result)

  # Interpolate TPR at mean FPR points
  interp_tpr <- approx(roc_result$specificities, roc_result$sensitivities, xout = mean_fpr)$y
  interp_tpr[1] <- 0  # Ensure that the curve starts at 0
  tprs[[fold]] <- interp_tpr
}

mean_tpr <- rowMeans(do.call(rbind, tprs))
mean_tpr[length(mean_tpr)] <- 1  # Ensure that the curve ends at 1
mean_auc <- mean(aucs)
std_auc <- sd(aucs)

std_tpr <- apply(do.call(rbind, tprs), 2, sd)
tprs_upper <- pmin(mean_tpr + std_tpr, 1)
tprs_lower <- pmax(mean_tpr - std_tpr, 0)

roc_data <- data.frame(
  mean_fpr = mean_fpr,
  mean_tpr = mean_tpr,
  tprs_upper = tprs_upper,
  tprs_lower = tprs_lower
)

gg <- ggplot(roc_data, aes(x = mean_fpr, y = mean_tpr)) +
  geom_line(color = "blue", size = 1.2, alpha = 0.8) +
  #geom_ribbon(aes(ymin = tprs_lower, ymax = tprs_upper), fill = "grey", alpha = 0.2, inherit.aes = FALSE) +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "red", size = 1.2) +
  labs(title = "Receiver Operating Characteristic Example",
       subtitle = sprintf("Mean ROC (AUC = %0.2f +/- %0.2f)", mean_auc, std_auc),
       x = "False Positive Rate",
       y = "True Positive Rate") +
  theme_minimal()

# Display the plot
print(gg)
