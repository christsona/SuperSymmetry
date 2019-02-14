library(keras)
library(DataExplorer)
library(recipes)
library(tidyverse)
library(rsample)

raw_data = read.csv(file = "/home/neo/Downloads/supersym.csv", header = TRUE, sep=",")            
glimpse(raw_data)

str(raw_data)
dim(raw_data)

plot_missing(raw_data)

train_1 = raw_data[0:250000,]
train_2 = raw_data[250001:500000,]
train_3 = raw_data[500001:750000,]
train_4 = raw_data[750001:999999,]

glimpse(train_1)
glimpse(train_2)
glimpse(train_3)
glimpse(train_4)

rec_obj = recipe(target ~ ., data=train_1) %>% 
  step_center(all_predictors(), -all_outcomes()) %>% 
  step_scale(all_predictors(), -all_outcomes()) %>% 
  prep(data=train_1)

new_train1x = bake(rec_obj, new_data=train_1) %>% select(-target)
new_train1y = train_1[,1]

new_train2x = bake(rec_obj, new_data=train_2) %>% select(-target)
new_train2y = train_2[,1]

new_train3x = bake(rec_obj, new_data=train_3) %>% select(-target)
new_train3y = train_3[,1]

new_train4x = bake(rec_obj, new_data=train_4) %>% select(-target)
new_train4y = train_4[,1]

model1 = keras_model_sequential() %>% 
  layer_dense(unit=16, activation="relu", initializer_he_normal(), input_shape= ncol(new_train1x)) %>% 
  layer_dense(unit=16, activation="relu", initializer_he_normal()) %>% 
  #layer_batch_normalization() %>% 
  layer_dense(unit=1, activation="sigmoid")
  
model1 %>% compile(
  optimizer = optimizer_adam(),
  loss = "binary_crossentropy",
  metrics = c("accuracy")
)

model1 %>% fit(as.matrix(new_train1x), as.numeric(new_train1y), epochs=5, batch_size=128, validation_split=0.2)

model2 = keras_model_sequential() %>% 
  layer_dense(unit=256, activation="relu", initializer_he_normal(), input_shape= ncol(new_train2x)) %>% 
  layer_dense(unit=64, activation="relu", initializer_he_normal()) %>%
  layer_dense(unit=64, activation="relu", initializer_he_normal()) %>% 
  #layer_batch_normalization() %>% 
  layer_dense(unit=1, activation="sigmoid")

model2 %>% compile(
  optimizer = optimizer_rmsprop(),
  loss = "binary_crossentropy",
  metrics = c("accuracy")
)

model2 %>% fit(as.matrix(new_train2x), as.numeric(new_train2y), epochs=5, batch_size=128, validation_split=0.2)

model3 = keras_model_sequential() %>% 
  layer_dense(unit=128, activation="selu", initializer_he_normal(), input_shape= ncol(new_train3x)) %>% 
  layer_dense(unit=75, activation="selu", initializer_he_normal()) %>% 
  #layer_batch_normalization() %>% 
  layer_dense(unit=1, activation="sigmoid")

model3 %>% compile(
  optimizer = optimizer_nadam(),
  loss = "binary_crossentropy",
  metrics = c("accuracy")
)

model3 %>% fit(as.matrix(new_train3x), as.numeric(new_train3y), epochs=5, batch_size=128, validation_split=0.2)

model4 = keras_model_sequential() %>% 
  layer_dense(unit=600, activation="selu", initializer_he_normal(), input_shape= ncol(new_train4x)) %>% 
  layer_dropout(rate=0.1) %>% 
  layer_dense(unit=450, activation="selu", initializer_he_normal()) %>% 
  layer_dropout(rate=0.1) %>% 
  layer_dense(units=200, activation="selu", initializer_he_normal()) %>% 
  layer_batch_normalization() %>% 
  layer_dense(unit=1, activation="sigmoid")

model4 %>% compile(
  optimizer = optimizer_adam(),
  loss = "binary_crossentropy",
  metrics = c("accuracy")
)

model4 %>% fit(as.matrix(new_train4x), as.numeric(new_train4y), epochs=5, batch_size=128, validation_split=0.2)
