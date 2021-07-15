###### AEMON-J machine learning workshop, Mridul K. Thomas, 2021-07-15 ######
# Workshop web page: https://aquaticdatasciopensci.github.io/day3-machinelearning/
# This is a script intended to introduce the use of basic machine learning techniques
# Focus: fitting different model types, visualising them, and basic evaluation techniques
# Some important details will be glossed over
# More advanced topics you will want to look into: 
#   cross-validation, preprocessing/rescaling/normalising data, boosting
#############################################################################


##### Before you start 
# 
# ake sure R and RStudio are up to date. Right now you should be using R 4.1.0 and RStudio Desktop 1.4.1717 
# Check this before beginning and update if needed. 
# When you are done with this, update all packages as well (Tools -> Check for package updates)
#
#
##### General R tips for beginners
#
# 1) Create a subdirectory (folder) for your scripts and data that you will use in the class. 
#   Set that as your working directory when you start. Use getwd() and setwd() for this. 
#   If you are more comfortable, you can use 'Projects' or Github integration
# 
# 2) Make sure RStudio is NOT saving the history of every session! 
# 
# 3) Put a # at the beginning of the line to make the line a 'comment'. R will ignore it. 
#   Keyboard shortcut: CTRL+SHIFT+C or CMD+SHIFT+C.
# 
# 2) Learn the details of any function by typing '?' before the function name; e.g. 
# ?lm
# 
# 3) The R reference card here is good to understand base R: https://cran.r-project.org/doc/contrib/Short-refcard.pdf 
# 
# 4) Here is a list of 'cheatsheets' for the tidyverse: https://rstudio.com/resources/cheatsheets/
#   This one is particularly relevant: https://rstudio.com/wp-content/uploads/2015/02/data-wrangling-cheatsheet.pdf
#   And some more detail with examples: https://dplyr.tidyverse.org/
# 
# 5) A general (but old) R crash course written by a friend of mine: 
#   https://github.com/ctkremer/R_CrashCourse/blob/master/R_crash_course_042915.R
# 

# Set your own working directory
# Mac users: the format looks like this (change directory to whatever you want):
setwd("~/Dropbox/Work stuff/Teaching/Statistics/")
# PC users: the format looks like this  (change directory to whatever you want):
setwd('C:/Users/Dropbox/Work stuff/Teaching/Statistics/')

# Install packages we will use. 
# You can remove packages that  you already have installed from the list.
install.packages(c('dplyr', 'ggplot2', 'pdp', 'ranger', 'caret', 
                   'nnet', 'ggeffects', 'NeuralNetTools'))

# Load libraries (packages)
library(dplyr)
library(ggplot2)
library(ggeffects)
library(pdp)
library(ranger)
library(caret)
library(nnet)
library(NeuralNetTools)


# Set some defaults to make plots prettier (feel free to change)
# Default plot settings
par(cex.axis = 1.75, cex.lab = 1.75, oma = c(0,0,0,0), mar = c(4.8, 4.5, 2, 2), lwd = 1.5)
theme_set(theme_bw(base_size = 24))




##### I. Visualising simple linear regression with 1 predictor

# Practise simulating data with a simple, known relationship between predictor (x) and target (y)

# runif simulates values from a uniform distribution
# rnorm simulates values from a normal distribution
# After working through this example, change the parameter values for runif and rnorm
#   or change the distributions themselves 
# Then re-run the models to see how well they work

# Define and visualise x
x <- runif(100, 0, 100)
hist(x)

# Define and visualise y. Note that I have added random error using rnorm()
y <- 1.7 * x + rnorm(100, 0, 10)
hist(y)

# Put together in a data frame and remove unnecessary entities
visdat1 <- data.frame(x = x, y = y)
rm(x, y)

# Fit linear regression and examine output
lm1 <- lm(y ~ x, data = visdat1)
summary(lm1)
# Obviously a strong linear relationship with a very low p-value

# visualise their relationship 
ggplot(visdat1, aes(x, y)) + geom_point(size = 3) + 
  geom_smooth(method = 'lm', se = FALSE)
# This visualisation approach is great for this simple example. 
# But it does not generalise well to many predictors, unfortunately. 


# This code will produce a similar plot but will generalise to more dimensions and model types,
#   as you will see
# Visualise the partial effect of 1 predictor
lm1 %>%
  partial(pred.var = "x", chull = TRUE, progress = "text") %>%
  plotPartial(contour = TRUE)


## Fit your first machine learning model to the same data
# A random forest 
rf1 <- ranger(y ~ x, data = visdat1)

# Visualise the partial effect of the predictor in the random forest model
rf1 %>%
  partial(pred.var = "x", chull = TRUE, progress = "text") %>%
  plotPartial(contour = TRUE)
# Note that the model does not capture a completely linear relationship
# If you have a simple, known relationship, a simple model works better!


## Now let's try a more complex nonlinear relationship
x <- seq(0, 15*pi, length.out = 500)
hist(x)

# Define and visualise y
y <- 1.7 * sin(0.4 * x) + rnorm(100, 0, 1)
hist(y)

# Put together in a data frame and remove unnecessary entities
visdat1_sin <- data.frame(x = x, y = y)
rm(x, y)

# visualise their relationship 
ggplot(visdat1_sin, aes(x, y)) + geom_point(size = 3) 

# Now fit a random forest to it
rf1_sin <- ranger(y ~ x, data = visdat1_sin)

# Visualise the partial effect of the predictor in the RF model
rf1_sin %>%
  partial(pred.var = "x", chull = TRUE, progress = "text") %>%
  plotPartial(contour = TRUE)
# The random forest is able to capture the sinusoidal relationship automatically
# Hard to do this with basic statistical models unless the relationship is known 
#   (possible to some extent with GAMs, though you will find RFs are more flexible)

### PRACTICE: 
# Try the above with 
# (i)   greater 'error' (increase the variance in rnorm, see ?rnorm)
# (ii)  changing the coefficient for the equation for y (i.e. the slope or 'beta') on line 86. 
# (iii) simulating other nonlinear relationships e.g. exponentials and saturating relationships



##### II. Visualising models with 2 predictors (NO interaction)

# Clear previous items from environment 
rm(list = ls())

# Define x1 and x2
x1 <- runif(1000, 0, 100)
x2 <- runif(1000, 0, 100)

# Create target variable y from x1 and x2, add random error (NO interaction)
y <- (1.7 * x1) + (0.9 * x2) + rnorm(1000, 0, 10)

# Put together in a data frame and remove unnecessary entities
visdat2 <- data.frame(x1 = x1, x2 = x2, y = y)
rm(x1, x2, y)

# Fit linear regression and examine output
lm2 <- lm(y ~ x1 + x2, data = visdat2)
summary(lm2)

# Visualise the partial effect of 1 of the predictors
lm2 %>%
  partial(pred.var = "x1", chull = TRUE, progress = "text") %>%
  plotPartial(contour = TRUE)

# Visualise the partial effect of 2 predictors
lm2 %>%
  partial(pred.var = c("x1", "x2"), chull = TRUE, progress = "text") %>%
  plotPartial(contour = TRUE)
# Parallel lines on coloured surface indicate lack of interaction

## Now fit ML models
# Because there is randomness in fitting ML models, each fit can be different
# Set a seed before fitting to make your results reproducible
set.seed(127)

# Fit random forest to the same data
rf2 <- ranger(y ~ x1 + x2, data = visdat2)

# Visualise the partial effect of 2 predictors from random forest
# This can take >10 seconds, possibly >1 minute
rf2 %>%
  partial(pred.var = c("x1", "x2"), chull = TRUE, progress = "text") %>%
  plotPartial(contour = TRUE)
# Note the general shape is captured correctly, but there is lots of 'fuzziness'
# The model is very flexible, so the surface is distorted by error in the data 
#   i.e. it is slightly overfitted

# Compare fitted vs. observed
ggplot(visdat2, aes(y, rf2$predictions)) + 
  geom_point() + 
  geom_abline(intercept = 0, slope = 1) +
  ylab('Fitted values') +
  xlab('True values')
# Overall a very good fit

# Estimated R^2 (generally reliable but can be slightly biased)
rf2$r.squared

# Which of the 2 predictors is more important? 
# Let's refit the forest to check this
rf2 <- ranger(y ~ x1 + x2, data = visdat2, importance = 'permutation')

# Evaluate importance of different predictors through permutation
importance(rf2)
# x1 is a stronger predictor of y than x2
# This is consistent with the coefficients we put into the equation(x1 = 1.7, x2 = 0.9)


### Now let's fit a neural network with 1 'hidden layer' to the same data
# Again set a seed for reproducibility
set.seed(12876)

# Fit neural network with the nnet package
n2 <- nnet(y ~ x1 + x2, data = visdat2, size = 10, 
           linout = TRUE, maxit = 1000)

# Visualise the partial effect of 2 predictors from neural network
n2 %>%
  partial(pred.var = c("x1", "x2"), chull = TRUE, progress = "text") %>%
  plotPartial(contour = TRUE)

# Compare fitted vs. observed
ggplot(visdat2, aes(y, predict(n2, visdat2))) + 
  geom_point() + 
  geom_abline(intercept = 0, slope = 1) +
  ylab('Predicted or fitted values') +
  xlab('True values')

### PRACTICE: 
# Try the above with 
# (i)   changing the coefficients of x1 and x2 in the equation for y, and check how RF importance changes
# (ii)  changing one of the relationships to be nonlinear (exponential or sine wave)





##### III. Visualise model with 2 predictors that interact with each other

# Clear previous items from environment 
rm(list = ls())

# Define x1 and x2
x1 <- runif(1000, 0, 100)
x2 <- runif(1000, 0, 100)

# Create target variable y from x1 and x2 WITH interaction, add random error
# The interaction is created by multiplying x1 and x2 together
y <- (1.7 * x1) + (0.9 * x2) + (0.1 * x1 * x2) + rnorm(1000, 0, 30)

# Put together in a data frame and remove unnecessary entities
visdat3 <- data.frame(x1 = x1, x2 = x2, y = y)
rm(x1, x2, y)

# Fit linear regression and examine output
lm3 <- lm(y ~ x1 * x2, data = visdat3)
summary(lm3)

# Visualise the partial effects of one predictor in the lm
lm3 %>%
  partial(pred.var = c("x1"), chull = TRUE, progress = "text") %>%
  plotPartial(contour = TRUE)

# Visualise the partial effects of both interacting predictors
lm3 %>%
  partial(pred.var = c("x1", "x2"), chull = TRUE, progress = "text") %>%
  plotPartial(contour = TRUE)
# Note the curvature in the surface - this indicates an interaction


# # This commented section is FYI; it can be ignored for the workshop. 
# # The visualisation approach above does not work for all packages.
# # So sometimes you have to do this manually. Here's an example of how to do it for lm
# # (i) Create an evenly spaced grid across predictors, 
# # (ii) Then use the fitted regression to predict the y values
# # (iii) Finally plot it using colour
# visdat3_grid <- expand.grid(x1 = 0:1000, x2 = 0:1000)
# visdat3_grid$y <- predict(lm3, visdat3_grid)
# visdat3_grid %>%
#   ggplot(., aes(x1, x2, z = y)) + geom_contour_filled() +
#   scale_x_continuous(expand = c(0, 0)) +
#   scale_y_continuous(expand = c(0, 0)) + 
#   labs(fill = 'y')
# 
# # Removing from environment
# rm(visdat3_grid)

# Plot the true values against the fitted values from the model:
ggplot(visdat3, aes(y, predict(lm3))) + 
  geom_point() + 
  geom_abline(intercept = 0, slope = 1) +
  ylab('Fitted values') +
  xlab('True values')


### Fit a random forest to predict the surface

# Again set a seed for reproducibility
set.seed(17263)

# Fit model. 
# NOTE: ML models do not require you to specify the interaction! They are
#   automatically estimated
rf3 <- ranger(y ~ x1 + x2, data = visdat3)

# Visualise the interaction estimated by the model:
# This may take a minute or two to run
# If your computer is slow, ask for someone else to show this to you
rf3 %>%
  partial(pred.var = c("x1", "x2"), chull = TRUE, progress = "text") %>%
  plotPartial(contour = TRUE)

# Compare fitted vs. observed
ggplot(visdat3, aes(y, rf3$predictions)) + 
  geom_point() + 
  geom_abline(intercept = 0, slope = 1) +
  ylab('Fitted values') +
  xlab('True values')


### Fit a neural network to predict the surface

# Again set a seed for reproducibility
set.seed(135972)

# NOTE: ML models do not require you to specify the interaction! They are
#   automatically estimated
n3 <- nnet(y ~ x1 + x2, data = visdat3, size = 10, linout = TRUE, maxit = 1000)

# Visualise the interaction estimated by the model:
n3 %>%
  partial(pred.var = c("x1", "x2"), chull = TRUE, progress = "text") %>%
  plotPartial(contour = TRUE)

# Compare fitted vs. observed
ggplot(visdat3, aes(y, predict(n3, visdat3))) + 
  geom_point() + 
  geom_abline(intercept = 0, slope = 1) +
  ylab('Fitted values') +
  xlab('True values')

## This worked great! The interaction looks good
# But neural networks can be tricky to train. 
# Try this now - same code, but with a different starting seed:
set.seed(1972)

# Fit the model
n3_b <- nnet(y ~ x1 + x2, data = visdat3, size = 10, linout = TRUE, maxit = 1000)

# Let's compare the fitted against the true values: 
# Compare fitted vs. observed
ggplot(visdat3, aes(y, predict(n3_b, visdat3))) + 
  geom_point() + 
  geom_abline(intercept = 0, slope = 1) +
  ylab('Fitted values') +
  xlab('True values')

# Looks less good but not too bad. Let's Visualise the interaction estimated by the model:
n3_b %>%
  partial(pred.var = c("x1", "x2"), chull = TRUE, progress = "text") %>%
  plotPartial(contour = TRUE)
# The model did not pick up the interaction
# Just changing the starting seed led to a completely different estimated surface!
# This is why neural networks require more care than RFs. RFs are rather robust.
# We will deal with this problem later in the script


### PRACTICE: 
# Try this with:
# (i)  more and less data points
# (ii) other, nonlinear relationships




##### IV. Visualise model with 3 predictors that interact with each other

# Clear previous items from environment 
rm(list = ls())

# Define x1, x2 and x3
x1 <- runif(1000, 0, 100)
x2 <- runif(1000, 0, 100)
x3 <- runif(1000, 0, 100)

# Create target variable y from x1, x2  and x3 WITH interactions, add random error
y <- (1.7 * x1) + (0.9 * x2) + (-0.6 * x3) + (0.1 * x1 * x2) + (-0.15 * x1 * x3) + rnorm(1000, 10, 100)
# Note here that I have included two 2-way interactions (x1:x2, x1:x3) but some interactions 
#     have been left out (x2:x3, x1:x2:x3)
# Feel free to change this and experiment! 

# Put together in a data frame and remove unnecessary entities
visdat4 <- data.frame(x1 = x1, x2 = x2, x3 = x3, y = y)
rm(x1, x2, x3, y)

# Fit model and examine summary
lm4 <- lm(y ~ x1 * x2 * x3, data = visdat4)
summary(lm4)
# Look at the interactions in the summary - note that the p-values here reflect the interactions that I set up
# IMPORTANT: if there is an interaction (like x1:x3), the p-value for the main effects (x1 and x3 separately) are 
#   meaningless.

# Visualise partial effect of 1 predictor (other predictors are set at their mean)
lm4 %>%
  partial(pred.var = "x3", chull = TRUE, progress = "text") %>%
  plotPartial(contour = TRUE)

# Visualise partial effect of all 3 interacting predictors
# This can take >1 minute to run
lm4 %>%
  partial(pred.var = c("x1", "x2", "x3"), chull = TRUE, progress = "text") %>%
  plotPartial(contour = TRUE)
# Notice the complex nonlinear surface we have created

# # FYI - not important for the workshop but perhaps useful to think about
# # This is another way to visualise model output. 
# # This works mainly with linear models but the principles 
# #   can be applied to machine learning models as well. 
# pr <- ggpredict(lm4, type = "fixed", 
#                 terms = c("x1",
#                           "x2", 
#                           "x3"))
# 
# plot(pr, show.legend = TRUE) 

# Plot the true values against the values predicted by the fitted model:
ggplot(visdat4, aes(y, predict(lm4))) + 
  geom_point() + 
  geom_abline(intercept = 0, slope = 1) +
  ylab('Fitted values') +
  xlab('True values')


### Fit random forest to same data

# Again set a seed for reproducibility
set.seed(183723)

# Note again that interactions are not specified for machine learning models
rf4 <- ranger(y ~ x1 + x2 + x3, data = visdat4)

# WARNING: This next line can take a long time to run! Expect several minutes at least, hours at most!
# There are faster ways to calculate and plot this but I'm using this package 
#   for convenience because it works with many different kinds of models
# If you work with models like this yourself, you might want to use other, faster methods/packages
rf4 %>%
  partial(pred.var = c("x1", "x2", "x3"), chull = TRUE, progress = "text") %>%
  plotPartial(contour = TRUE)

# Compare fitted vs. observed
ggplot(visdat4, aes(y, rf4$predictions)) + 
  geom_point() + 
  geom_abline(intercept = 0, slope = 1) +
  ylab('Fitted values') +
  xlab('True values')
# Looks generally good but some bias at extreme values


### Fit neural network to same data
# Again set a seed for reproducibility
set.seed(281546)

n4 <- nnet(y ~ x1 * x2 * x3, data = visdat4, size = 10, linout = TRUE, maxit = 1000)

# # Trying a bigger network to deal with the more complex surface
# n4 <- nnet(y ~ x1 * x2 * x3, data = visdat4, size = 100, linout = TRUE, maxit = 1000)
# 
# # Trying a smaller network 
# n4 <- nnet(y ~ x1 * x2 * x3, data = visdat4, size = 25, linout = TRUE, maxit = 1000)

# Visualising the partial effects of all 3 interacting predictors
# WARNING: This next line can take multiple minutes to run
n4 %>%
  partial(pred.var = c("x1", "x2", "x3"), chull = TRUE, progress = "text") %>%
  plotPartial(contour = TRUE)
# Fit looks very bad!
# Let's compare fitted vs. observed
ggplot(visdat4, aes(y, predict(n4, visdat4))) + 
  geom_point() + 
  geom_abline(intercept = 0, slope = 1) +
  ylab('Fitted values') +
  xlab('True values')
# Clear problems - it's only predicting about 5 distinct values

# How do we fix this? 
# Neural networks can be finicky to 'train' (i.e. fit to data)
# It is almost always helpful to rescale/normalise all your data - this helps the algorithm to learn better & faster
# There are multiple ways this normalisation is done
# We will use a common method call min-max normalization 
# https://en.wikipedia.org/wiki/Feature_scaling#Rescaling_(min-max_normalization)

# Using an automated function to rescale data
pp <- preProcess(visdat4, method = 'range')
visdat4_normalised <- predict(pp, visdat4)

# Refitting the neural network to rescaled data
n4_normalised <- nnet(y ~ x1 * x2 * x3, data = visdat4_normalised, size = 10, linout = TRUE, maxit = 1000)

# Does the normalisation help?
# Visualising the partial effects of all 3 interacting predictors
# WARNING: This next line can take multiple minutes to run
n4_normalised %>%
  partial(pred.var = c("x1", "x2", "x3"), chull = TRUE, progress = "text") %>%
  plotPartial(contour = TRUE)

# Let's compare fitted vs. observed
ggplot(visdat4_normalised, aes(y, predict(n4_normalised, visdat4_normalised))) + 
  geom_point() + 
  geom_abline(intercept = 0, slope = 1) +
  ylab('Fitted values') +
  xlab('True values')

### PRACTICE: 
# Try this with:
# (i)  more 'nodes' in the neural network (increase the 'size' parameter in the nnet function)
# (ii) other forms of rescaling: https://en.wikipedia.org/wiki/Feature_scaling






##### V. Selecting the right set of parameters using cross validation 

# Clear previous items from environment 
rm(list = ls())

# Define x1, x2 and x3
x1 <- runif(1000, 0, 100)
x2 <- runif(1000, 0, 100)
x3 <- runif(1000, 0, 100)

# Create target variable y from x1, x2  and x3 WITH interactions, add random error
y <- (1.7 * x1) + (0.9 * x2) + (-0.6 * x3) + (0.1 * x1 * x2) + (-0.15 * x1 * x3) + rnorm(1000, 10, 100)
# Note here that I have included two 2-way interactions (x1:x2, x1:x3) but some interactions 
#     have been left out (x2:x3, x1:x2:x3)
# Feel free to change this and experiment! 

# Put together in a data frame and remove unnecessary entities
visdat5 <- data.frame(x1 = x1, x2 = x2, x3 = x3, y = y)


### Cross-validation with random forests

# Using the caret package to cross validate data (i.e. choose model parameters)
rf5 <- train(y ~ x1 + x2 + x3, data = visdat5, 
             method = 'ranger', 
             # trControl = trainControl(method = 'repeatedcv', repeats = 5, verboseIter = FALSE), # More rigorous, much more computation
             trControl = trainControl(method = 'cv', verboseIter = FALSE),
             metric = "RMSE")

# Look at how performance changes with parameter choice
plot(rf5)
# Lowest RMSE is best parameter set
# This model is automatically saved

# But we can manually choose which model parameters to evaluate:
# Only 10 parameter combinations chosen here for an example; you will want to do more

rfgrid <- expand.grid(mtry = c(2, 3),
                      splitrule = 'extratrees',
                      min.node.size = c(1, 2, 5, 10, 20))

# Make sure you're not choosing too many combinations! 
# You are now (in the worst case) multiplying the time it took to compute the earlier 
#   models by the factor shown by the following line
dim(rfgrid)[1] * 10 #because we are doing 10-fold cross validation

# Now let's evaluate the best set of parameters
rf5_grid <- train(y ~ x1 + x2 + x3, data = visdat5, 
             method = 'ranger', 
             # trControl = trainControl(method = 'repeatedcv', repeats = 5, verboseIter = FALSE), # More rigorous, much more computation
             trControl = trainControl(method = 'cv', verboseIter = FALSE),
             metric = "RMSE",
             tuneGrid = rfgrid)

# View summary of cross-validation
rf5_grid

# Plot the comparison. Lowest RMSE is best; this model is automatically chosen.
plot(rf5_grid)

# Visualise partial effects of all 3 interacting predictors
# WARNING: Can take several minutes
rf5_grid %>%
  partial(pred.var = c("x1", "x2", "x3"), chull = TRUE, progress = "text") %>%
  plotPartial(contour = TRUE)



### Cross-validation with neural networks
# 
# Generate normalised dataset for neural networks (not needed for RFs, remember)
pp <- preProcess(visdat5, method = 'range')
visdat5_normalised <- predict(pp, visdat5)

# Remove unnecessary elements
rm(pp)

# Defining set of neural network model parameters to evaluate:
# Only 4 parameter combinations chosen here for an example; you will want to do more
nnetgrid <- expand.grid(size = c(5, 25),
                        decay = c(0.01, 0.1))

# Make sure you're not running this too many times! 
# You are now (in the worst case) multiplying the time it took to compute the earlier 
#   models by the factor shown by the following line 
dim(nnetgrid)[1] * 10 #because we are doing 10-fold cross validation

# Now let's evaluate the best set of parameters
n5_grid <- train(y ~ x1 + x2 + x3, data = visdat5_normalised, 
                 method = 'nnet', 
                 linout = TRUE,
                 maxit = 1000,
                 # trControl = trainControl(method = 'repeatedcv', repeats = 5, verboseIter = FALSE), # More rigorous, much more computation
                 trControl = trainControl(method = 'cv', verboseIter = FALSE),
                 metric = "RMSE",
                 tuneGrid = nnetgrid)


# Look at how performance changes with parameter choice
plot(n5_grid)

# View summary
n5_grid

# Compare true values with fitted (after normalisation)
ggplot(visdat5_normalised, aes(y, predict(n5_grid, visdat5_normalised))) + 
  geom_point() + 
  geom_abline(intercept = 0, slope = 1) +
  ylab('Fitted values') +
  xlab('True values')


# Visualise partial effects of all 3 interacting predictors
# WARNING: Can take several minutes
n5_grid %>%
  partial(pred.var = c("x1", "x2", "x3"), chull = TRUE, progress = "text") %>%
  plotPartial(contour = TRUE)

# Visualise the neural network itself
# The line thickness indicates the strength of the connection between nodes
# The 'weights' roughly correspond to coefficients but are not very useful by themselves
plotnet(n5_grid)


### PRACTICE
# 
# Try this with 
# (i) more predictor variables (add x4, x5, etc. and change the equation for y to depend on them as well)







##### VI. Evaluating models in an unbiased manner - holdout datasets 
# (more important for neural networks)

# Clear previous items from environment 
rm(list = ls())

# Define x1, x2 and x3
x1 <- runif(1000, 0, 100)
x2 <- runif(1000, 0, 100)
x3 <- runif(1000, 0, 100)

# Create target variable y from x1, x2  and x3 WITH interactions, add random error
y <- (1.7 * x1) + (0.9 * x2) + (-0.6 * x3) + (0.1 * x1 * x2) + (-0.15 * x1 * x3) + rnorm(1000, 10, 100)
# Note here that I have included two 2-way interactions (x1:x2, x1:x3) but some interactions 

# Put together in a data frame and remove unnecessary entities
visdat6 <- data.frame(x1 = x1, x2 = x2, x3 = x3, y = y)

# Generate normalised dataset for neural networks (not needed for RFs, remember)
pp <- preProcess(visdat6, method = 'range')
visdat6_normalised <- predict(pp, visdat6)

# Remove unnecessary elements
rm(x1, x2, x3, y, pp)

# Randomly select 20% of the data to use as holdout for evaluating the model
# IMPORTANT: This is much more complex when there is structure in your data!
# You need additional steps for spatial/temporal/hierarchical data
trainIndex <- createDataPartition(visdat6_normalised$y, p = 0.8, 
                                  list = FALSE, 
                                  times = 1)

# Split the data into training and test datasets based on the random selections above
visdat6norm_Train <- visdat6_normalised[ trainIndex,]
visdat6norm_Test  <- visdat6_normalised[-trainIndex,]

# Defining grid to cross-validate on the TRAINING SET ONLY 
# Only 4 parameter combinations chosen here for an example; you will want to do more
nnetgrid <- expand.grid(size = c(5, 25),
                        decay = c(0.01, 0.1))

# Now let's evaluate the best set of parameters
n6_grid <- train(y ~ x1 + x2 + x3, data = visdat6norm_Train, 
                 method = 'nnet', 
                 linout = TRUE,
                 maxit = 1000,
                 # trControl = trainControl(method = 'repeatedcv', repeats = 5, verboseIter = FALSE), # More rigorous, much more computation
                 trControl = trainControl(method = 'cv', verboseIter = FALSE),
                 metric = "RMSE",
                 tuneGrid = nnetgrid)

# Examine results of CV
n6_grid
plot(n6_grid)
# Best model automatically chosen

# Now evaluate performance on both the train and the test sets

# Generate predictions for the training dataset
preds_train <- predict(n6_grid, visdat6norm_Train)

# Generate predictions for the test dataset
preds_test <- predict(n6_grid, visdat6norm_Test)

# R^2 for training data
postResample(pred = preds_train, obs = visdat6norm_Train$y)

# R^2 for test data
postResample(pred = preds_test, obs = visdat6norm_Test$y)


# Both results similar which is good. 
# BUT: Results on the training data are expected to be biased
# So trust the results on the holdout test set instead

# Note: This can be done through a cross-validation step as well! 
# But this requires a 2-level nested cross-validation

