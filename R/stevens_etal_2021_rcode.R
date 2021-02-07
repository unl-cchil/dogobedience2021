## ---
##
## Script name: stevens_etal_2021_rcode.R
##
## Purpose of script: Analyze data investigating dog and owner characteristics that predict dog success on Canine Good Citizen test and impulsivity.
##
## Authors: Dr. Jeffrey R. Stevens (jeffrey.r.stevens@gmail.com) and London Wolff (lmwolff3@gmail.com)
##
## Date Created: 2019-01-04
##
## Date Finalized: 2021-02-07
##
## License: All materials presented here are released under the Creative Commons Attribution 4.0 International Public License (CC BY 4.0).
##  You are free to:
##   Share — copy and redistribute the material in any medium or format
##   Adapt — remix, transform, and build upon the material for any purpose, even commercially.
##  Under the following terms:
##   Attribution — You must give appropriate credit, provide a link to the license, and indicate if changes were made. You may do so in any reasonable manner, but not in any way that suggests the licensor endorses you or your use.
##   No additional restrictions — You may not apply legal terms or technological measures that legally restrict others from doing anything the license permits.
##
## ---
##
## Notes:
##
## Instructions: Place this file and the data files in the main directory.
## 	Create a folder called "figures". Set the R working directory to the main directory.
##  At the R command prompt, type
## 	> source("stevens_etal_2021_rcode.R")
## 	This will run the script, adding all of the calculated variables to the workspace and
##  saving figures in the figures directory. If packages do not load properly, install them
##  with install.packages("package_name").
## Data files:
## ---
##
## stevens_etal_2021_data1.csv (primary behavioral, cognitive, and cortisol data set)
## id - Dog id number
## date - Date owner completed survey
## class - Obedience training class
## dog_age - Age of dog in years
## dog_sex - Sex of dog
## dog_neutered - Neuter status (neutered/spayed = Yes, intact = No)
## owner_gender - Gender of owner
## time_train_dog_weekly_num - Number of hours per week spent training
## dog_behavior_bennett_disobedient_score - Disobedience subscale of Bennett & Rohlf behavior scale
## dog_behavior_bennett_aggressive_score - Aggression subscale of Bennett & Rohlf behavior scale
## dog_behavior_bennett_nervous_score - Nervousness subscale of Bennett & Rohlf behavior scale
## dog_behavior_bennett_destructive_score - Destructiveness subscale of Bennett & Rohlf behavior scale
## dog_behavior_bennett_excitable_score - Excitability subscale of Bennett & Rohlf behavior scale
## dog_behavior_bennett_overall_score - Overall score of Bennett & Rohlf behavior scale
## dog_problematic_behaviors_hiby_score - Problematic behavior score of Hiby et al.
## dog_obedience_hiby_score - Obedience score of Hiby et al.
## dias_behavioral_regulation_score - Behavioral regulation subscale of DIAS from Wright et al.
## dias_aggression_score - Aggression subscale of DIAS from Wright et al.
## dias_responsiveness_score - Responsiveness subscale of DIAS from Wright et al.
## dias_overall_score - Overall score for DIAS from Wright et al.
## mdors_score - Monash Dog Owner Relationship Scale score
## personality_extraversion_score - Extraversion score from brief Big-Five personality scale Gosling et al.
## personality_agreeableness_score - Agreeableness score from brief Big-Five personality scale Gosling et al.
## personality_conscientiousness_score - Conscientiousness score from brief Big-Five personality scale Gosling et al.
## personality_stability_score - Stability score from brief Big-Five personality scale Gosling et al.
## personality_openness_score - Openness score from brief Big-Five personality scale Gosling et al.
## lotr_score - Life Orientation Test Revised score from Scheier et al.
## pss_score - Perceived Stress Scale score from Cohen et al.
## crt_score - Cognitive Reflection Task score from Frederick
## numeracy_score - Berlin Numeracy Test score from Cokely et al.
## latency_sit_mean - Mean latency between command and sit behavior (in seconds)
## latency_down_mean - Mean latency between command and down behavior (in seconds)
## cort1 - Cortisol levels before first class meeting (in ug/dL)
## cort2 - Cortisol levels after first class meeting (in ug/dL)
## cort3 - Cortisol levels before last class meeting (in ug/dL)
## cort4 - Cortisol levels after last class meeting (in ug/dL)
## cgc_test - Success in completing Canine Good Citizen test (Pass/Fail)
##
## stevens_etal_2021_data2.csv (item-specifc data for calculating internal consistency reliability)
##  survey - name of survey
##  item_1 - item_13 - individual items (surveys differ on number of items, so NAs represent no items)
##
## stevens_etal_2021_data3.csv (behavioral (sit/down) data for calculating inter-rater reliability)
##  block - Replication block (1, 2, or 3)
##  coder - Coder ID
##  date - Date of behavioral data collection
##  id - Dog ID
##  latency_sit - Latency between command and sit behavior (in seconds)
##  latency_down - Latency between command and down behavior (in seconds)
##
## ---

# Load libraries ----------------------------

library(bayestestR)   # needed for estimating Bayes factors from linear models
library(ggbeeswarm)   # needed for beeswarm plots
library(lme4)         # needed for GLMMs
library(caret)        # needed for predictor correlations
library(rpart)        # needed for CART algorithm
library(C50)          # needed for C5.0 algorithm
library(randomForest) # needed for Random Forest algorithm
library(e1071)        # needed for skewness calculations
library(foreach)      # needed for iteration
library(ggcorrplot)   # needed for pairwise correlation plot
library(patchwork)    # needed for subfigure placement
library(tidymodels)   # needed for machine learning analysis
library(vip)          # needed for calculating variable importance
library(papaja)       # needed for APA formatting
library(tidyverse)    # needed for data processing
library(psych)        # needed for calculating reliability
library(here)         # needed for accessing folders

# Define functions ----------------------------

## Compute log with an offset of 1 to avoid logging 0
adjust_log <- function(x) log(x + 1)

## Function to calculate predictor importance for different models
accuracy_importance <- function(mod_type, engine, mode, df, y) {
  # Prepare data
  df <- rename(df, y = y)  # rename outcome variable to y
  set.seed(345)  # set a random seed

  # Center and scale numeric predictors
  scale_recipe <- recipe(~ ., data = df) %>%
    step_normalize(all_numeric(), -all_outcomes())  # scale and center all numeric predictors
  scale_prep <- prep(scale_recipe) # estimate parameters from training set

  # Conduct 10-fold cross-validation data with 10 repeats
  folds <- vfold_cv(df, v = 10, strata = y, repeats = 10)
  folds %>%
    mutate(juiced = map(splits, ~ bake(scale_prep, new_data = analysis(.))))

  # Assign the model and mode type
  model_type <- get(mod_type)  # convert string to function
  model <- model_type() %>%  # set model type
    set_engine(engine) %>%  # set engine
    set_mode(mode)  # set mode (classification or regression)

  # Create workflow
  model_workflow <- workflow() %>%
    add_model(model) %>%  # add glm model
    add_formula(y ~ .)  # add formula

  # Resample training data
  fit_rs <-
    model_workflow %>%
    fit_resamples(folds)  # apply 10-fold cross-validation
  resample_metrics <- collect_metrics(fit_rs)  # aggregate the accuracy and ROC AUC over the folds

  # Fit the model
  model_fit <- model_workflow %>%
    fit(df)  # fit model on all data

  # Calculate variable importance
  variable_importance <- vi(pull_workflow_fit(model_fit), method = "firm", train = df, scale = TRUE) # calculate importance with feature importance ranking measure approach
  variable_importance <- variable_importance %>%
    mutate(model = engine)  # create column with engine type

  # Create output
  output <- list(fit = model_fit, accuracy = resample_metrics, vi = variable_importance)  # create list of output
  return(output)
}

## Find proportion of a vector for plotting text in particular location
axis_prop <- function(vec, prop) {
  prop * (max(vec) - min(vec)) + min(vec)
}

## Plot logistic regression outcomes
gglogistic <- function(df, x, y, xlabel, ylabel) {
  df2 <- select(df, x = all_of(x), y = all_of(y)) %>%   # select only x and y columns and rename
    filter(!is.na(x))

  # Fit models
  null_model <- glm(y ~ 1, data = df2, family = binomial(link = "logit"))
  alternative_model <- glm(y ~ x, data = df2, family = binomial(link = "logit"))
  alternative_p <- summary(alternative_model)$coefficients[2, 4]
  alternative_bf <- bf_models(alternative_model, denominator = null_model)$BF[1]  # estimate BF from BICs
  df2 <- mutate(df2, y = as.numeric(as.character(y)))

  # Plot data and logistic regression curve
  gglogisticplot <<- ggplot(df2, aes(x = x, y = y)) +
    geom_beeswarm(groupOnX = FALSE, shape = 1) +  # plot data points with beeswarm
    geom_smooth(method = "glm", formula = y ~ x, method.args = list(family = "binomial")) +  # plot logistic regression curve
    scale_y_continuous(breaks = c(0, 1)) +  # plot only 0 and 1 on y-axis
    labs(x = xlabel, y = ylabel) +  # label x- and y-axis
    geom_text(x = 0.5 * (min(df2$x, na.rm = TRUE) + max(df2$x, na.rm = TRUE)), y =  axis_prop(df2$y, 0.9), label = paste("p =", round(alternative_p, 2), ", BF =", round(alternative_bf, 2)), size = 6) +  # add p-values and Bayes factors
    theme_bw() +  # change theme
    theme(panel.grid = element_blank(),  # remove grid lines
          axis.title = element_text(size = 20),  # set axis label font size
          axis.text = element_text(size = 15))  # set tick mark label font size
  output <- list(pvalue = alternative_p, bf = alternative_bf)  # create list of output
  return(output)
}

## Plot regression outcomes
ggregression <- function(df, x, y, xlabel, ylabel) {
  df2 <- select(df, x = all_of(x), y = all_of(y))  # select only x and y columns and rename

  # Fit models
  null_model <- lm(y ~ 1, data = df2)
  alternative_model <- lm(y ~ x, data = df2)
  alternative_p <- summary(alternative_model)$coefficients[2, 4]
  alternative_bf <- bf_models(alternative_model, denominator = null_model)$BF[1]  # estimate BF from BICs
  df2 <- mutate(df2, y = as.numeric(as.character(y)))

  # Plot data and linear regression
  ggregressionplot <<- ggplot(df2, aes(x = x, y = y)) +
    geom_point(shape = 1) +  # plot individual data points
    geom_smooth(method = "lm", formula = y ~ x) +  # plot regression line
    labs(x = xlabel, y = ylabel) +  # label x- and y-axis
    geom_text(x = 0.5 * (min(df2$x, na.rm = TRUE) + max(df2$x, na.rm = TRUE)), y =  axis_prop(df2$y, 0.1), label = paste("p =", round(alternative_p, 2), ", BF =", round(alternative_bf, 2)), size = 6) +  # add p-values and Bayes factors
    theme_bw() +  # change theme
    theme(panel.grid = element_blank(),  # remove grid lines
          axis.title = element_text(size = 20),  # set axis label font size
          axis.text = element_text(size = 15))  # set tick mark label font size
  output <- list(pvalue = alternative_p, bf = alternative_bf)  # create list of output
  return(output)
}

# Create version of apa_print for lmer objects
my_print_lmer <- function(model, predictor) {
  cis <- data.frame(confint(model, level = 0.95))
  cis <- cis %>%
    mutate(factor = row.names(.)) %>%
    rename(lower95 = `X2.5..`, upper95 = `X97.5..`)
  df <- data.frame(summary(model)$coefficients)
  df <- df %>%
    mutate(factor = row.names(.)) %>%
    rename(estimate = `Estimate`, sd = `Std..Error`, p.value = `Pr...t..`) %>%
    left_join(cis, by = "factor") %>%
    dplyr::select(factor, estimate, lower95, upper95, everything())
  output <- paste("$b = ", printnum(filter(df, factor == predictor)$estimate), "$, 95\\% CI $[", printnum(filter(df, factor == predictor)$lower95), "$, $", printnum(filter(df, factor == predictor)$upper95), "]$, $t(", printnum(filter(df, factor == predictor)$df), ") = ", printnum(filter(df, factor == predictor)$t.value), "$, $p = ", printp(filter(df, factor == predictor)$p.value), "$", sep = "")
  return(output)
}


# Input data ----------------------------

all_data <- read_csv(here("data/stevens_etal_2021_data1.csv")) %>%  # read in primary data file
  mutate(cort_reactivity = cort2 - cort1,
         logcort1 = log(cort1),
         logcort2 = log(cort2),
         logcort3 = log(cort3),
         logcort4 = log(cort4),
         logcort_reactivity = logcort2 - logcort1,
         logcort_reactivity2 = logcort4 - logcort3,
         longterm_cort1 = logcort3 - logcort1,
         longterm_cort2 = logcort4 - logcort2,
         logcort_reactivity_diff = logcort_reactivity2 - logcort_reactivity,
         cognitive = crt_score + numeracy_score,
         dog_age_num = str_replace(dog_age, "< 1", "0"),
         dog_age_num = str_replace(dog_age_num, " years old", ""),
         dog_age_num = as.numeric(str_replace(dog_age_num, " year old", "")),
         cgc_pass = ifelse(cgc_test != "Pass" | is.na(cgc_test), 0, 1))

item_data <- read_csv(here("data/stevens_etal_2021_data2.csv"))  # read in item-specific data for reliability analysis

behavioral_data_all <- read_csv(here("data/stevens_etal_2021_data3.csv"))  # read in behavioral data for inter-rater reliability

# Demographics ----------------------------

survey_data <- filter(all_data, !is.na(dog_age_num))  # remove dogs whose owner did not take the survey
survey_data_dias <- filter(survey_data, !is.na(dias_overall_score))  # remove dogs whose owner did not have the DIAS component
dog_sex_nums <- table(survey_data$dog_sex)  # find dog sex distribution
owner_gender_nums <- table(survey_data$owner_gender)  # find owner gender distribution

all_cgc_test <- dim(filter(all_data, !is.na(cgc_test)))[1]  # find total number of dogs taking CGC
all_cgc_notest <- dim(filter(all_data, is.na(cgc_test)))[1]  # find total number of dogs NOT taking CGC
survey_cgc_test <- dim(filter(survey_data, !is.na(cgc_test)))[1]  # find number of survey dogs taking CGC
survey_cgc_notest <- dim(filter(survey_data, is.na(cgc_test)))[1]  # find number of survey dogs taking CGC
survey_cgc_pass <- table(survey_data$cgc_test)[2]  # find number of dogs failing CGC
survey_cgc_fail <- table(survey_data$cgc_test)[1]  # find number of dogs failing CGC
survey_na_sit <- dim(filter(survey_data, is.na(latency_sit_mean)))[1]
survey_na_down <- dim(filter(survey_data, is.na(latency_down_mean)))[1]

# Cortisol
all_cort_nums <- dim(filter_at(all_data, .vars = vars(cort1:cort4), ~ !is.na(.)))[1]
cort_nums <- dim(filter_at(all_data, .vars = vars(cort1:cort4), any_vars(!is.na(.))))[1]
cort_data_long <- all_data %>%
  select(id, logcort1:logcort4) %>%
  pivot_longer(-id, names_to = "cort_sample", values_to = "cort_levels") %>%
  drop_na()
cort_samples_nums <- nrow(cort_data_long)
survey_cort_nums <- dim(filter_at(survey_data, .vars = vars(cort1:cort4), any_vars(!is.na(.))))[1]
all_survey_cort_nums <- dim(filter_at(survey_data, .vars = vars(cort1:cort4), ~ !is.na(.)))[1]

# Calculate reliability for survey measures ----------------------------

## _Dog behavior (Bennett & Rolf 2007) -------------------------
# Create data frames for subscales
dog_behavior_disobedient <- filter(item_data, survey == "dog_behavior_disobedient") %>%  # extract survey
  select(-survey) %>%  # remove naming column
  select_if(function(x) {!all(is.na(x))})  # remove empty columns
dog_behavior_aggressive <- filter(item_data, survey == "dog_behavior_aggressive") %>%  # extract survey
  select(-survey) %>%  # remove naming column
  select_if(function(x) {!all(is.na(x))})  # remove empty columns
dog_behavior_nervous <- filter(item_data, survey == "dog_behavior_nervous") %>%  # extract survey
  select(-survey) %>%  # remove naming column
  select_if(function(x) {!all(is.na(x))})  # remove empty columns
dog_behavior_destructive <- filter(item_data, survey == "dog_behavior_destructive") %>%  # extract survey
  select(-survey) %>%  # remove naming column
  select_if(function(x) {!all(is.na(x))})  # remove empty columns
dog_behavior_excitable <- filter(item_data, survey == "dog_behavior_excitable") %>%  # extract survey
  select(-survey) %>%  # remove naming column
  select_if(function(x) {!all(is.na(x))})  # remove empty columns

# Calculate survey reliability
dog_behavior_disobedient_reliability <- omega(dog_behavior_disobedient, warnings = FALSE, plot = FALSE)
dog_behavior_aggressive_reliability <- omega(dog_behavior_aggressive, warnings = FALSE, plot = FALSE)
dog_behavior_nervous_reliability <- omega(dog_behavior_nervous, warnings = FALSE, plot = FALSE)
dog_behavior_destructive_reliability <- omega(dog_behavior_destructive, warnings = FALSE, plot = FALSE)
dog_behavior_excitable_reliability <- omega(dog_behavior_excitable, warnings = FALSE, plot = FALSE)

## _Dog obedience (Hiby et al. 2004) -------------------
dog_obedience <- filter(item_data, survey == "dog_obedience") %>%  # extract survey
  select(-survey) %>%  # remove naming column
  select_if(function(x) {!all(is.na(x))})  # remove empty columns
dog_obedience_reliability <- omega(dog_obedience, plot = FALSE)  # calculate reliability

## _Dog problematic behaviors (Hiby et al. 2004) -------------------
dog_problematic_behavior <- filter(item_data, survey == "dog_problem_behaviors") %>%  # extract survey
  select(-survey) %>%  # remove naming column
  select_if(function(x) {!all(is.na(x))})  # remove empty columns
dog_problematic_behavior_reliability <- omega(dog_problematic_behavior, warnings = FALSE, plot = FALSE)  # calculate reliability

## _DIAS (Wright et al. 2011) -------------------
# Create data frames for surveys
dias_behavioral_regulation <- filter(item_data, survey == "dias_behavioral_regulation") %>%  # extract survey
  select(-survey) %>%  # remove naming column
  select_if(function(x) {!all(is.na(x))})  # remove empty columns
dias_aggression <- filter(item_data, survey == "dias_aggression") %>%  # extract survey
  select(-survey) %>%  # remove naming column
  select_if(function(x) {!all(is.na(x))})  # remove empty columns
dias_responsiveness <- filter(item_data, survey == "dias_responsiveness") %>%  # extract survey
  select(-survey) %>%  # remove naming column
  select_if(function(x) {!all(is.na(x))})  # remove empty columns

# Calculate survey reliability
dias_behavioral_regulation_reliability <- omega(dias_behavioral_regulation, plot = FALSE)
dias_aggression_reliability <- omega(dias_aggression, plot = FALSE)
dias_responsiveness_reliability <- omega(dias_responsiveness, plot = FALSE)

## _MDORS (Dwyer et al. 2006) -------------------
mdors <- filter(item_data, survey == "mdors") %>%  # extract survey
  select(-survey) %>%  # remove naming column
  select_if(function(x) {!all(is.na(x))})  # remove empty columns
mdors_reliability <- omega(mdors, plot = FALSE)  # calculate reliability

## _Owner personality (Gosling et al., 2003) -------------------
# Create data frames for surveys
extraversion <- filter(item_data, survey == "owner_personality_extraversion") %>%  # extract survey
  select(-survey) %>%  # remove naming column
  select_if(function(x) {!all(is.na(x))})  # remove empty columns
agreeableness <- filter(item_data, survey == "owner_personality_agreeableness") %>%  # extract survey
  select(-survey) %>%  # remove naming column
  select_if(function(x) {!all(is.na(x))})  # remove empty columns
conscientiousness <- filter(item_data, survey == "owner_personality_conscientiousness") %>%  # extract survey
  select(-survey) %>%  # remove naming column
  select_if(function(x) {!all(is.na(x))})  # remove empty columns
stability <- filter(item_data, survey == "owner_personality_stability") %>%  # extract survey
  select(-survey) %>%  # remove naming column
  select_if(function(x) {!all(is.na(x))})  # remove empty columns
openness <- filter(item_data, survey == "owner_personality_openness") %>%  # extract survey
  select(-survey) %>%  # remove naming column
  select_if(function(x) {!all(is.na(x))})  # remove empty columns

# Calculate survey reliability (using Cronbach's alpha because omega could not compute)
extraversion_reliability <- alpha(extraversion)
agreeableness_reliability <- alpha(agreeableness)
conscientiousness_reliability <- alpha(conscientiousness)
stability_reliability <- alpha(stability)
openness_reliability <- alpha(openness)


## _Life Orientation Test Revised (Scheier et al., 1994) -------------------
lotr <- filter(item_data, survey == "lotr") %>%  # extract survey
  select(-survey) %>%  # remove naming column
  select_if(function(x) {!all(is.na(x))})  # remove empty columns
lotr_reliability <- omega(lotr, plot = FALSE)  # calculate reliability

## _Perceived Stress Scale (Cohen et al., 1983) -------------------
pss <- filter(item_data, survey == "perceived_stress") %>%  # extract survey
  select(-survey) %>%  # remove naming column
  select_if(function(x) {!all(is.na(x))})  # remove empty columns
pss_reliability <- omega(pss, plot = FALSE)  # calculate reliability

## _Cognitive Reflection Test (Frederick 2005) -------------------
crt <- filter(item_data, survey == "crt") %>%  # extract survey
  select(-survey) %>%  # remove naming column
  select_if(function(x) {!all(is.na(x))})  # remove empty columns
crt_reliability <- psych::omega(crt, plot = FALSE)  # calculate reliability

## _Berlin Numeracy Test (Cokely et al., 2012) -------------------
numeracy <- filter(item_data, survey == "numeracy") %>%  # extract survey
  select(-survey) %>%  # remove naming column
  select_if(function(x) {!all(is.na(x))}) %>%   # remove empty columns
  mutate_all(list(~replace_na(., 0)))
numeracy_reliability <- omega(numeracy, warnings = FALSE, plot = FALSE)  # calculate reliability

# Behavior coding inter-rater reliability ----------------------------

## Separate data
behavioral_data1 <- filter(behavioral_data_all, block == 1)
behavioral_data2 <- filter(behavioral_data_all, block == 2)
behavioral_data3 <- filter(behavioral_data_all, block == 3)

## _Data subset 1 -------------------
# Prepare data
behavioral_data_sit1 <- behavioral_data1 %>%
  select(-block, -latency_down) %>%  # remove unnecessary columns
  pivot_wider(names_from = coder, values_from = latency_sit) %>%  # pivot data to wide format
  select(-date, -id)  # remove unnecessary columns
behavioral_data_down1 <- behavioral_data1 %>%
  select(-block, -latency_sit) %>%   # remove unnecessary columns
  pivot_wider(names_from = coder, values_from = latency_down) %>%  # pivot data to wide format
  select(-date, -id)  # remove unnecessary columns

# Calculate inter-rater reliability
sit_icc1 <- ICC(behavioral_data_sit1)
down_icc1 <- ICC(behavioral_data_down1)

## _Data subset 2 -------------------
# Prepare data
behavioral_data_sit2 <- behavioral_data2 %>%
  select(-block, -latency_down) %>%  # remove unnecessary columns
  pivot_wider(names_from = coder, values_from = latency_sit) %>%  # pivot data to wide format
  select(-date, -id)  # remove unnecessary columns
behavioral_data_down2 <- behavioral_data2 %>%
  select(-block, -latency_sit) %>%   # remove unnecessary columns
  pivot_wider(names_from = coder, values_from = latency_down) %>%  # pivot data to wide format
  select(-date, -id)  # remove unnecessary columns

# Calculate inter-rater reliability
sit_icc2 <- ICC(behavioral_data_sit2)
down_icc2 <- ICC(behavioral_data_down2)

## _Data subset 3 -------------------
# Prepare data
behavioral_data_sit3 <- behavioral_data3 %>%
  select(-block, -latency_down) %>%  # remove unnecessary columns
  pivot_wider(names_from = coder, values_from = latency_sit) %>%  # pivot data to wide format
  select(-date, -id)  # remove unnecessary columns
behavioral_data_down3 <- behavioral_data3 %>%
  select(-block, -latency_sit) %>%   # remove unnecessary columns
  pivot_wider(names_from = coder, values_from = latency_down) %>%  # pivot data to wide format
  select(-date, -id)  # remove unnecessary columns

# Calculate inter-rater reliability
sit_icc3 <- ICC(behavioral_data_sit3)
down_icc3 <- ICC(behavioral_data_down3)


# Machine learning analysis ----------------------------
## _Prepare data -------------------

# Select data for machine-learning analysis
## All subjects, no DIAS
all_predictors <- survey_data %>%
  select(dog_sex, dog_neutered, dog_age_num, time_train_dog_weekly_num:cgc_pass) %>%
  select(-cgc_test, -(cort1:cort_reactivity)) %>%  # remove owner gender, old cgc column, and cort data
  mutate(cgc_pass = as.factor(cgc_pass))  # convert to factor
survey_predictors <- all_predictors %>%
  select(-contains("cort"), -contains("dias")) # remove individual items to leave summary scores and remove DIAS

# Check predictor skewness
numeric_predictors <- survey_predictors %>%
  select(where(is_numeric)) %>%   # select numeric predictors
  select(-cgc_pass)
## Plot histograms
pivot_longer(numeric_predictors, everything(), names_to = "predictor", values_to = "value") %>%
  mutate(predictor = as.character(fct_recode(predictor, "Dog age" = "dog_age_num", "Dog aggression" =  "dog_behavior_bennett_aggressive_score", "Dog destructiveness" = "dog_behavior_bennett_destructive_score", "Dog disobedience" = "dog_behavior_bennett_disobedient_score", "Dog excitability" = "dog_behavior_bennett_excitable_score", "Dog nervousness" = "dog_behavior_bennett_nervous_score", "Dog problem behaviors (Bennett)" = "dog_behavior_bennett_overall_score", "Dog obedience" =  "dog_obedience_hiby_score", "Dog problem behaviors (Hiby)" = "dog_problematic_behaviors_hiby_score", "Dog sit latency" = "latency_sit_mean", "Dog down latency" = "latency_down_mean", "DIAS aggression" = "dias_aggression_score", "DIAS behavior regulation" = "dias_behavioral_regulation_score", "DIAS responsiveness" = "dias_responsiveness_score", "DIAS overall" = "dias_overall_score", "Owner optimism" = "lotr_score", "Owner stress" = "pss_score", "Owner agreeableness" = "personality_agreeableness_score", "Owner conscientiousness" = "personality_conscientiousness_score", "Owner extraversion" = "personality_extraversion_score", "Owner openness" = "personality_openness_score", "Owner stability" = "personality_stability_score", "Owner cognitive ability" = "cognitive", "Dog-owner relationship" = "mdors_score", "Time spent training" = "time_train_dog_weekly_num", "Owner cognitive reflection" = "crt_score", "Owner numeracy" = "numeracy_score"))) %>%  # rename predictors
  arrange(predictor) %>%  # sort alphabetically
  ggplot(aes(x = value)) +
  geom_histogram(bins = 30) +  # plot histogram
  facet_wrap(~predictor, scales = "free", ncol = 4) +  # separate by predictor
  ggsave(here("figures/predictor_histograms.png"), width = 9, height = 12)

## Calculate skewness for each predictor
skewvalues <- numeric_predictors %>%
  summarise(across(where(is_numeric), ~ skewness(.x, na.rm = TRUE))) %>%
  pivot_longer(everything(), names_to = "predictor", values_to = "skew") %>%   # pivot to long format
  arrange(skew)  # arrange by skewness
highly_skewed <- filter(skewvalues, skew > 1)  # filter predictors with skewness > 1
highly_skewed_predictors <- highly_skewed$predictor  # create vector of highly skewed predictors

# Assign algorithms
classification_algorithms <- data.frame(name = c("Regression", "CART", "C5.0", "Random forest", "Neural network"), engine = c("glm", "rpart", "C5.0", "randomForest", "nnet"), model = c("logistic_reg", "decision_tree", "decision_tree", "rand_forest", "mlp"))  # create data frame of classification algorithms
regression_algorithms <- data.frame(name = c("Regression", "CART", "Random forest", "Neural network"), engine = c("lm", "rpart", "randomForest", "nnet"), model = c("linear_reg", "decision_tree", "rand_forest", "mlp"))  # create data frame of regression algorithms (use linear instead of logistic regression)


# _Preprocess data ---------

preprocess_recipe_cgc <- recipe(cgc_pass ~ ., data = survey_predictors) %>%
  step_meanimpute(pss_score) %>%
  step_log(all_of(highly_skewed_predictors), offset = 1) %>%  # log transform predictors with offset of 1
  step_dummy(all_nominal(), -all_outcomes()) %>%   # set all factor predictors to dummy variables
  step_nzv(all_predictors())  # check for near zero variance

prep_recipe_cgc <- prep(preprocess_recipe_cgc, survey_predictors, retain = TRUE)
preprocessed_data_cgc <- juice(prep_recipe_cgc) %>%
  select(-cgc_pass, everything())  # move cgc_pass to end

# _Select predictors ---------

preprocessed_predictors_cgc <- select(preprocessed_data_cgc, -cgc_pass)  # remove outcome variable

## Examine multicollinearity
correlation_matrix_cgc <- cor(preprocessed_predictors_cgc, use = "pairwise.complete.obs")  # create correlation matrix of predictors
highly_correlated_cgc <- findCorrelation(correlation_matrix_cgc, cutoff = 0.7, names = TRUE)  # find the highly correlated predictors: lotr_score, dias_behavioral_regulation_score, cognitive
correlation_matrix_cgc[, highly_correlated_cgc]  # view correlation matrix of highly correlated predictors
trimmed_predictors_cgc <- select(preprocessed_data_cgc, -crt_score, -numeracy_score, -dog_behavior_bennett_overall_score)  # remove correlated predictors (keep cognitive because it has more variation potential than crt_score and numeracy_score)

## Apply simple filter of Bayes factor (Kuhn & Johnson, 2019)
# Calculate Bayes factors for predictor relationships with CGC pass rate
trimmed_predictor_names_cgc <- names(trimmed_predictors_cgc[-ncol(trimmed_predictors_cgc)])  # get column names of numeric predictors
predictor_bfs_cgc <- foreach(predictor = trimmed_predictor_names_cgc, .combine = "c") %do% {  # for each predictor
  df2 <- trimmed_predictors_cgc %>%
    select(x = all_of(predictor), y = cgc_pass) %>%   # select only x and y columns and rename
    filter(!is.na(x))
  null_model <- glm(y ~ 1, data = df2, family = binomial(link = "logit"))  # run null logistic regression
  alternative_model <- glm(y ~ x, data = df2, family = binomial(link = "logit"))  # run alternative logistic regression
  alternative_bf <- bf_models(alternative_model, denominator = null_model)$BF[1]  # estimate BF from BICs
}
predictor_bf_cgc <- tibble(predictor = trimmed_predictor_names_cgc, bf = predictor_bfs_cgc) %>%   # create tibble of BFs
  arrange(bf)  # sort by BF
selected_predictors_cgc <- filter(predictor_bf_cgc, bf > 0.33)$predictor  # find predictors with BFs > 1/3
cgc_data <- select(preprocessed_data_cgc, all_of(selected_predictors_cgc), cgc_pass)  # select predictors with BFs > 1/3

# _Fit models and find predictor importance ---------

# Calculate accuracy and predictor importance for each algorithm
foreach(algorithm = classification_algorithms$name) %do% {  # for each algorithm
  model <- classification_algorithms$model[classification_algorithms$name == algorithm]  # extract model
  engine <- classification_algorithms$engine[classification_algorithms$name == algorithm]  # extract engine
  acc_imp <- accuracy_importance(model, engine, "classification", df = cgc_data, y = "cgc_pass")  # calculate accuracy and importance
  acc_imp_name <- paste(engine, "_acc_imp", sep = "")  # create name for all data
  acc_name <- paste(engine, "_acc", sep = "")  # create name for accuracy data
  imp_name <- paste(engine, "_vi", sep = "")  # create name for importance data
  assign(acc_imp_name, acc_imp)  # assign all data to variable
  assign(acc_name, acc_imp$accuracy)  # assign accuracy data to variable
  assign(imp_name, acc_imp$vi)  # assign importance data to variable
}

# Create data frame for accuracy
cgc_model_accuracy <- bind_rows(glm_acc, rpart_acc, C5.0_acc, randomForest_acc, nnet_acc) %>%  # combine algorithm accuracy data
  filter(.metric == "accuracy") %>%  # filter accuracy values
  mutate(model = classification_algorithms$name,  # add algorithm name
         ci = std_err * 1.96) %>%  # calculate 95% CI
  select(model, everything()) %>%  # reorder columns
  arrange(desc(mean))  # arrange by mean accuracy
# write_csv(cgc_model_accuracy, "data/cgc_model_accuracy.csv")  # write to file

# Create data frame for predictor importance
cgc_model_vis <- bind_rows(glm_vi, rpart_vi, C5.0_vi, randomForest_vi, nnet_vi) %>%  # combine algorithm importance data
  select(model, predictor = Variable, importance = Importance)  %>%   # reorder and rename columns
  mutate(predictor_name = fct_recode(predictor, "Dog disobedience" = "dog_behavior_bennett_disobedient_score", "Owner stress" = "pss_score", "Owner cognitive ability" = "cognitive", "Dog-owner relationship" = "mdors_score", "Owner extraversion" = "personality_extraversion_score", "Training time" = "time_train_dog_weekly_num", "Dog sit latency" = "latency_sit_mean"),  # create predictor names
         model = fct_recode(model, "Regression" = "glm", "CART" = "rpart", "Random forest" = "randomForest", "Neural network" = "nnet")) %>%  # rename models
  select(model, predictor, predictor_name, importance)  # reorder columns
# write_csv(cgc_model_vis, "data/cgc_model_vis.csv")  # write to file

# Logistic regression
cgc_data <- rename(cgc_data, "Dog disobedience" = "dog_behavior_bennett_disobedient_score", "Owner stress" = "pss_score", "Owner cognitive ability" = "cognitive", "Dog-owner relationship" = "mdors_score", "Owner extraversion" = "personality_extraversion_score", "Training time" = "time_train_dog_weekly_num")

# Scale and center numeric predictors
scale_cgc <- recipe(~ ., data = cgc_data) %>%
  step_normalize(all_numeric(), -all_outcomes())  # scale and center all numeric predictors
prep_scale_cgc <- prep(scale_cgc, cgc_data, retain = TRUE)  # estimate parameters from training set
scaled_data_cgc <- juice(prep_scale_cgc) %>%  # apply recipe to data
  select(-cgc_pass, everything())  # move cgc_pass to end

# Calculate logistic regression
cgc_glm_fit <- glm(cgc_pass ~., data = scaled_data_cgc, family = "binomial")
summary(cgc_glm_fit)
cgc_glm_apa <- apa_print(cgc_glm_fit)  # create APA regression table


# Plots ----------------------------

# _CGC and predictors ---------

# gglogistic(cgc_data, "Dog sit latency", "cgc_pass", "Dog sit latency", "CGC Success")
# cgc_sit_plot <- gglogisticplot
gglogistic(cgc_data, "Dog disobedience", "cgc_pass", "Dog disobedience", "CGC Success")
cgc_disobedient_plot <- gglogisticplot
gglogistic(cgc_data, "Owner stress", "cgc_pass", "Owner stress", "CGC Success")
cgc_stress_plot <- gglogisticplot
gglogistic(cgc_data, "Dog-owner relationship", "cgc_pass", "Dog-owner relationship", "CGC Success")
cgc_mdors_plot <- gglogisticplot
gglogistic(cgc_data, "Owner cognitive ability", "cgc_pass", "Owner cognitive ability", "CGC Success")
cgc_intel_plot <- gglogisticplot
gglogistic(cgc_data, "Training time", "cgc_pass", "Training time", "CGC Success")
cgc_train_plot <- gglogisticplot
gglogistic(cgc_data, "Owner extraversion", "cgc_pass", "Owner extraversion", "CGC Success")
cgc_extraversion_plot <- gglogisticplot

cgc_disobedient_plot + cgc_intel_plot + cgc_stress_plot + cgc_extraversion_plot + cgc_train_plot + cgc_mdors_plot + plot_layout(nrow = 2)
ggsave(here("figures/cgc_plots.png"), width = 10, height = 7)

# __Accuracy  ---------
ggplot(cgc_model_accuracy, aes(x = reorder(model, mean), y = mean)) +
  geom_point() +
  geom_linerange(aes(ymin = mean - ci, ymax = mean + ci)) +
  coord_flip() +
  labs(x = "Algorithm", y = "Predictive accuracy") +  # label axes
  theme_bw() +  # set theme
  theme(panel.grid = element_blank(),  # remove grid lines
        axis.text.x = element_text(size = 20),   # rotate x axis labels
        axis.text.y = element_text(size = 20),   # rotate x axis labels
        axis.title = element_text(size = 25))   # adjust strip font size
ggsave(here("figures/cgc_accuracy_algorithm.png"), width = 8, height = 4)

# __Predictor importance ---------
cgc_model_vis_means <- cgc_model_vis %>%
  group_by(predictor, predictor_name) %>%
  summarise(importance = mean(importance),
            model = "Mean") %>%
  select(model, predictor, predictor_name, importance)
cgc_model_vis_all <- bind_rows(cgc_model_vis, cgc_model_vis_means)
ggplot(cgc_model_vis_all, aes(x = importance, y = reorder(predictor_name, importance))) +
  geom_point(size = 3) +
  facet_wrap(~ factor(model, levels = c("Mean", "C5.0", "Regression", "Random forest", "Neural network", "CART"))) +
  labs(x = "Importance", y = "Predictors") +  # label axes
  theme_bw() +  # set theme
  theme(axis.text.x = element_text(size = 20),   # rotate x axis labels
        axis.text.y = element_text(size = 20),   # rotate x axis labels
        axis.title = element_text(size = 25),   # adjust strip font size
        strip.text = element_text(size = 20))  # adjust strip font size
ggsave(here("figures/cgc_predictor_importance_algorithm.png"), width = 12, height = 8)
