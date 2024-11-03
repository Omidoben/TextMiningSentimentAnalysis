# This project involves building a sentiment analysis model that predicts the rating of an Animal Crossing game based on the reviews given by the users.
# It uses a deep learning model, specifically an LSTM model

# An LSTM is a specific kind of network architecture with feedback loops that allow information to persist
# through steps and memory cells that can learn to "remember" and "forget" information through sequences

# They are well suited for text because of their ability to process text as long sequences of words or characters
# and can model structures within text like word dependencies

user_reviews <- read_tsv("user_reviews.tsv")
user_reviews

# EDA

user_reviews %>% 
  count(user_name, text)   # There are 2999 distinct users, with each user giving one review

user_reviews %>% 
  summarize(avg_grade = mean(grade))    # The average rating given  = 4.22

user_reviews %>% 
  count(grade) %>% 
  ggplot(aes(factor(grade), n)) +
  geom_col(aes(fill = "firebrick")) +
  theme_minimal() +
  theme(legend.position = "none") +
  labs(
    title = "Number of Ratings for the Animal Crossing Game",
    subtitle = "(Majority of the users gave a rating of 0)",
    x = "Grade (Rating)",
    y = "Frequency"
  )


# Character lengths of the reviews

user_reviews %>% 
  ggplot(aes(nchar(text))) +
  geom_histogram(alpha = 0.7, fill = "firebrick") +
  scale_x_log10() +
  labs(
    x = "Number of characters per review",
    y = "Number of reviews"
  )

# The distribution appears odd, the data has two thresholds. Nearly half of the reviews have a 
# character length < 400 while the other half have a character length > 800

# Diving deeper into the data to check if there's a reason behind the odd distribution
user_reviews %>% 
  filter(nchar(text) < 400) %>% 
  sample_n(5) %>% 
  pull(text)

# Majority of the reviews with less than 500 characters appear coherent, some even end with full stops


user_reviews %>% 
  filter(nchar(text) >= 800) %>% 
  sample_n(5) %>% 
  pull(text)

# Majority of the reviews with more than 800 characters also appear coherent.
# Some of the reviews end with "… Expand", this appears to be a continuation of the sentences. It shall
# be removed during data pre processing
# Notably, some of the reviews have repeated chunks of text, and there are some non English reviews


# Data Preparation

# The data contains 11 different ratings (from 0 to 10). Since the data has an odd distribution which makes it difficult
# to fit regression models, the ratings are converted to two classes (Bad reviews when the rating <= 6 and Good reviews when the rating > 6)

set.seed(456)
reviews <- user_reviews %>%
  mutate(
    text = str_remove(text, "Expand$"),
    rating = if_else(grade > 6, 1, 0)  # Good (1) and Bad (0)
  )

review_splits <- initial_split(reviews, strata = rating)
review_train <- training(review_splits)
review_test <- testing(review_splits)

# Hyperparameter Flags
flags <- flags(
  flag_integer("max_words", 2000),      # Vocabulary size
  flag_integer("max_length", 250),      # Sequence length
  flag_integer("output_dim", 164)      # Embedding output dimension
)


# Recipe
reviews_rec <- recipe(~ text, data = review_train) %>%
  step_tokenize(text) %>%
  step_stopwords(text) %>%
  step_tokenfilter(text, max_tokens = flags$max_words) %>%
  step_sequence_onehot(text, sequence_length = flags$max_length)


# Prepare and Bake the Recipe
reviews_prep <- prep(reviews_rec)
review_baked <- bake(reviews_prep, new_data = NULL, composition = "matrix")


# Model Definition
lstm_mod <- keras_model_sequential() %>%
  layer_embedding(input_dim = flags$max_words + 1, output_dim = flags$output_dim) %>%
  layer_lstm(units = 32, dropout = 0.5, recurrent_dropout = 0.5) %>%
  layer_dense(units = 1, activation = "sigmoid")


# Compile the Model
lstm_mod %>%
  compile(
    optimizer = "adam",
    loss = "binary_crossentropy",
    metrics = "accuracy"
  )

# Fit the Model
history <- lstm_mod %>%
  fit(
    review_baked,
    review_train$rating,
    epochs = 15,
    validation_split = 0.2,
    batch_size = 256,
    verbose = FALSE
  )

history



##########################################################################################

# Model tuning

library(tfruns)

# Specify the parameter ranges we want to try

hyperparams <- list(
  max_words = c(1500, 2000, 2500),
  max_length = c(200, 250, 300),
  output_dim = c(100, 150, 200)
)  
  

runs <- tuning_run(
  file = "AnimalCrossingLSTM.R",
  runs_dir = "_tuning",
  flags = hyperparams
)
# We manually specify the runs_dir argument, which is where the results of the tuning will be saved

tune_results <- as_tibble(ls_runs())
tune_results

# We can condese the results down a little bit by only pulling out the flags we are looking at and arranging
# them according to their performance

best_runs <- tune_results %>% 
  select(metric_val_accuracy, metric_accuracy, flag_max_words, flag_max_length, flag_output_dim) %>% 
  arrange(desc(metric_val_accuracy))

best_runs  

# From the tuning results, max_words = 2000, max_length = 250, and output_dim = 100 had the best performance

#####################################################################################################


# Fit the LSTM model using these metrics

max_words <- 2000
max_length <- 250
output_dim <- 100


set.seed(456)
reviews <- user_reviews %>%
  mutate(
    text = str_remove(text, "Expand$"),
    rating = if_else(grade > 6, 1, 0)  # Good (1) and Bad (0)
  )

review_splits <- initial_split(reviews, strata = rating)
review_train <- training(review_splits)
review_test <- testing(review_splits)


# Recipe
reviews_rec <- recipe(~ text, data = review_train) %>%
  step_tokenize(text) %>%
  step_stopwords(text) %>%
  step_tokenfilter(text, max_tokens = max_words) %>%
  step_sequence_onehot(text, sequence_length = max_length)


# Prepare and Bake the Recipe
reviews_prep <- prep(reviews_rec)
review_baked <- bake(reviews_prep, new_data = NULL, composition = "matrix")


# Model Definition
lstm_model <- keras_model_sequential() %>%
  layer_embedding(input_dim = max_words + 1, output_dim = output_dim) %>%
  layer_lstm(units = 32, dropout = 0.5, recurrent_dropout = 0.5) %>%
  layer_dense(units = 1, activation = "sigmoid")


# Compile the Model
lstm_model %>%
  compile(
    optimizer = "adam",
    loss = "binary_crossentropy",
    metrics = "accuracy"
  )

# Fit the Model
lstm_history <- lstm_model %>%
  fit(
    review_baked,
    review_train$rating,
    epochs = 15,
    validation_split = 0.2,
    batch_size = 256,
    verbose = FALSE
  )

lstm_history

plot(lstm_history)

# The model performs well on the training data with an accuracy = 0.9761, but it still over fits 
# on the validation set, accuracy = 0.7.
# This may be caused by the small amount of data available for this project 

#################################################################################################


# Evaluate the model on the test set

# Prepare the test data set
review_test_baked <- bake(reviews_prep, new_data = review_test, 
                          composition = "matrix")


# Create a function to calculate metrics (accuracy, ROC_AUC...) 

keras_predict <- function(model, baked_data, response) {
  predictions <- predict(model, baked_data)[, 1]
  
  tibble(
    .pred_1 = predictions,
    .pred_class = if_else(.pred_1 < 0.5, 0, 1),
    rating = response
  ) %>%
    mutate(across(c(rating, .pred_class), ## create factors
                  ~ factor(.x, levels = c(1, 0)))) ## with matching levels
}


val_res <- keras_predict(lstm_model, review_test_baked, review_test$rating)
val_res

val_res %>% metrics(rating, .pred_class, .pred_1)

# The model performs fairly well on the test set. It has an accuracy = 0.847 and an ROC_AUC = 0.922

# Confusion matrix
val_res %>% 
  conf_mat(rating, .pred_class) %>% 
  autoplot(type = "heatmap")


# ROC AUC curve
val_res %>% 
  roc_curve(rating, .pred_1) %>% 
  autoplot()

















