# In this project, I build a model that predicts whether a Netflix title is a TV Show or a Movie based on its description

library(tidyverse)

# Load the data
netflix_titles <- read_csv("netflix_titles.csv")
netflix_titles
glimpse(netflix_titles)


#  Data Exploration
netflix_titles %>% 
  count(type) %>% 
  mutate(prop = n / sum(n))   # The data is imbalanced. 69.6% of the titles are movies while 30.4% are TV Shows


# Netflix TV Shows/Movie releases across the years
netflix_titles %>% count(type, release_year, sort = TRUE)

p <- netflix_titles %>% 
  group_by(release_year, type) %>% 
  summarize(num_releases = n()) %>% 
  arrange(desc(num_releases)) %>% 
  ggplot(aes(release_year, num_releases, color = type)) +
  geom_line(linewidth = 1.0, show.legend = FALSE) +
  scale_x_continuous(limits = c(1925, 2024)) +
  facet_wrap(~type, scales = "free")

plotly::ggplotly(p)


# What categories were these Netflix titles listed in?
netflix_titles %>% 
  separate_rows(listed_in, sep = ",") %>% 
  mutate(listed_in = str_trim(listed_in)) %>%
  group_by(type, listed_in) %>% 
  summarize(n = n()) %>% 
  arrange(desc(n)) %>% 
  slice_max(n, n = 10) %>% 
  ungroup() %>% 
  mutate(listed_in = factor(listed_in, levels = unique(listed_in)),
         listed_in = fct_reorder(listed_in, n)) %>% 
  ggplot(aes(n, listed_in, fill = type)) +
    geom_col(show.legend = FALSE) +
  facet_wrap(~type, scales = "free")

# Majority of the Netflix movies were listed in International movies, dramas, and comedies 
# where are as majority of the Tv shows were listed in International TV Shows, TV Dramas and TV comedies

netflix_titles %>%
  filter(!is.na(rating)) %>% 
  count(type, rating) %>% 
  top_n(10, n) %>% 
  mutate(rating = fct_reorder(rating, n)) %>% 
  ggplot(aes(n, rating, fill = type)) +
  geom_col(position = position_dodge(width = ))


# Movie descriptions
netflix_titles %>% 
  filter(type == "Movie") %>% 
  sample_n(5) %>% 
  pull(description)

# TV Show descriptions
netflix_titles %>% 
  filter(type == "TV Show") %>% 
  sample_n(5) %>% 
  pull(description)


#Most common words used
library(tidytext)
netflix_titles %>% 
  unnest_tokens(word, description) %>% 
  anti_join(stop_words) %>% 
  count(type, word, sort = TRUE) %>% 
  group_by(type) %>% 
  top_n(10, n) %>% 
  ungroup() %>% 
  mutate(word = reorder_within(word, n, type)) %>% 
  ggplot(aes(n, word, fill = type)) +
  geom_col(show.legend = FALSE) +
  scale_y_reordered() +
  facet_wrap(~type, scales = "free") +
  theme_minimal()

###############################################################################################
# Build a model

# Select the data
netflix <- netflix_titles %>% 
  select(description, type)

# Split the data
library(tidymodels)
set.seed(2345)

netflix_splits <- initial_split(netflix, strata = type)
netflix_train <- training(netflix_splits)
netflix_test <- testing(netflix_splits)

# Resampling folds
set.seed(2543)
netflix_folds <- vfold_cv(netflix_train, strata = type)
netflix_folds


# Recipe
library(textrecipes)
library(themis)

# SMOTE is an over-sampling technique used to address class imbalance in datasets, particularly in classification problems.
# When dealing with imbalanced data, where one class (usually the minority class) has significantly fewer instances 
# than the other(s), models tend to be biased towards the majority class. 
# SMOTE helps mitigate this issue by generating synthetic examples of the minority class to balance the class distribution.

netflix_rec <- recipe(type ~ description, data = netflix_train) %>% 
  step_tokenize(description) %>%     # Don't remove stop words to check whether they add value to the model
  step_tokenfilter(description, max_tokens = 1000) %>% 
  step_tfidf(description) %>% 
  step_normalize(all_numeric_predictors()) %>% 
  step_smote(type)

prep(netflix_rec)

prep(netflix_rec) %>% bake(new_data = NULL)


# Model Specification

# SVM model with a linear kernel
# Often computationally less intensive compared to non-linear kernels and can be faster for large datasets.

# Linear SVMs handle high-dimensional spaces efficiently and are effective in finding a separating hyperplane.
# e.g. Text data, when represented using methods like TF-IDF, often results in a very high-dimensional feature space.
# 

library(LiblineaR)
svm_spec <- svm_linear() %>%    # Works well with text problems
  set_mode("classification") %>% 
  set_engine("LiblineaR")

# Workflow
svm_wf <- workflow() %>% 
  add_model(svm_spec) %>% 
  add_recipe(netflix_rec)

svm_wf

# Resampling
doParallel::registerDoParallel()
set.seed(6734)

svm_res <- fit_resamples(
  svm_wf,
  netflix_folds,
  metrics = metric_set(accuracy, recall, precision),  # Do not include roc_auc, this model doesn't support calculation of probabilities
  control = control_resamples(save_pred = TRUE)
)
svm_res

svm_res %>% 
  collect_metrics()

# Confusion matrix
svm_res %>% 
  conf_mat_resampled(tidy = FALSE)

# Build and fit final model
netflix_final <- last_fit(
  svm_wf, netflix_splits,
  metrics = metric_set(accuracy, recall, precision)
)

netflix_final

netflix_final %>% 
  collect_metrics()

# confusion matrix
netflix_final %>% 
  collect_predictions() %>% 
  conf_mat(type, .pred_class)


# Variable Importance
netflix_fit <- netflix_final %>% 
  extract_fit_parsnip()

tidy(netflix_fit) %>% 
  arrange(-estimate)    # words such as documentary, when, and debt contributed the most to movie prediction

tidy(netflix_fit) %>% 
  arrange(estimate)     # words such as series, docuseries contributed the most to TV Shows

# Visualization
tidy(netflix_fit) %>% 
  filter(!term == "Bias") %>% 
  group_by(sign = estimate > 0) %>% 
  slice_max(abs(estimate), n = 15) %>% 
  ungroup() %>% 
  mutate(term = str_remove(term, "tfidf_description_"),
         sign = if_else(sign, "More from Movies", "More from TV Shows")) %>% 
  ggplot(aes(abs(estimate), fct_reorder(term, abs(estimate)), fill = sign)) +
  geom_col(show.legend = FALSE) +
  facet_wrap(~sign, scales = "free") +
  labs(x = "Coefficient from Linear SVM", y = NULL)

















