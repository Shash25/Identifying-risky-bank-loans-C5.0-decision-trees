#########  Example - identifying risky bank loans using C5.0 decision trees #############

# Step 1 - collecting data #
# Data with these characteristics are available in a dataset donated to the UCI Machine Learning Data Repository
#(http://archive.ics.uci.edu/ml) by Hans Hofmann
#of the University of Hamburg. They represent loans obtained from a credit agency
#in Germany.

# Step 2 - exploring and preparing the data #

#credit <- read.csv("E:\\NuralTechSoft\\Machine Learning\\Machine learning examples\\Machine Learning with R\\credit.csv")
str(credit)
credit <- read.csv(file.choose(), header=T)

# Let's take a look at some of the table() output for a couple of features of loans that
#seem likely to predict a default. The checking_balance and savings_balance
#features indicate the applicant's checking and savings account balance, and are
#recorded as categorical variables:

table(credit$checking_balance)
table(credit$savings_balance)

# Since the loan data was obtained from Germany, the currency is recorded in Deutsche Marks (DM).

#Some of the loan's features are numeric, such as its term (months_loan_duration),
#and the amount of credit requested (amount).
summary(credit$months_loan_duration)
summary(credit$amount)

# The loan amounts ranged from 250 DM to 18,420 DM across terms of 4 to 72 months,
# with a median duration of 18 months and amount of 2,320 DM.

# The default variable indicates whether the loan applicant was unable to meet the
# agreed payment terms and went into default. A total of 30 percent of the loans went
# into default:
table(credit$default)

#### Default- 2
#### Non-Default- 1

# A high rate of default is undesirable for a bank because it means that the bank is
# unlikely to fully recover its investment. If we are successful, our model will identify
# applicants that are likely to default, so that this number can be reduced.

### Data preparation - creating random training and test datasets###


# We will split our data into two portions:
# a training dataset to build the decision tree and a test dataset to evaluate the
# performance of the model on new data. We will use 90 percent of the data for
# training and 10 percent for testing.

# The following command creates a randomly-ordered credit data frame. The
# set.seed() function is used to generate random numbers in a predefined sequence,
# starting from a position known as a seed (set here to the arbitrary value 12345). It
# may seem that this defeats the purpose of generating random numbers, but there
# is a good reason for doing it this way. The set.seed() function ensures that if the
# analysis is repeated, an identical result is obtained.

set.seed(12345)
credit_rand <- credit[order(runif(1000)), ]

# runif(n, min = , max = ) is used to generate n uniform random numbers lie in the interval (min, max).


# To confirm that we have the same data frame sorted differently, we'll compare
# values on the amount feature across the two data frames. The following code shows
# the summary statistics:
summary(credit$amount)
summary(credit_rand$amount)

# We can use the head() function to examine the first few values in each data frame:
head(credit$amount)
head(credit_rand$amount)

# Now, we can split into training (90 percent or 900 records), and test data (10 percent
# or 100 records) as we have done in previous analyses:
credit_train <- credit_rand[1:900, ]
credit_test <- credit_rand[901:1000, ]

# If all went well, we should have about 30 percent of defaulted loans in each of the datasets.
prop.table(table(credit_train$default))
prop.table(table(credit_test$default))

# This appears to be a fairly equal split, so we can now build our decision tree.




##### Step 3 - training a model on the data #####3

# If not installed previously, then use 
install.packages("C50")
library(C50)
# require(C50)
# For the first iteration of our credit approval model, we'll use the default C5.0
# configuration, as shown in the following code. The 21st column in credit_train is
# the class variable, default, so we need to exclude it from the training data frame as
# an independent variable, but supply it as the target factor vector for classification:

credit_train$default<-as.factor(credit_train$default)
str(credit_train$default)
credit_model <- C5.0(credit_train[-21], credit_train$default)

# The credit_model object now contains a C5.0 decision tree object. We can see some
# basic data about the tree by typing its name:
credit_model

#To see the decisions, we can call the summary() function on the model:
summary(credit_model)



###### Step 4 - evaluating model performance ########

# To apply our decision tree to the test dataset, we use the predict() function as
# shown in the following line of code:
 credit_pred <- predict(credit_model, credit_test)



# This creates a vector of predicted class values, which we can compare to the actual
# class values using the CrossTable() function in the gmodels package. Setting the
# prop.c and prop.r parameters to FALSE removes the column and row percentages
# from the table. The remaining percentage (prop.t) indicates the proportion of
# records in the cell out of the total number of records.

library(gmodels)
CrossTable(credit_test$default, credit_pred,
             prop.chisq = FALSE, prop.c = FALSE, prop.r = FALSE,
             dnn = c('actual default', 'predicted default'))


######  Step 5 - improving model performance #####

credit_boost10 <- C5.0(credit_train[-21], credit_train$default,
                       trials = 10)
# While examining the resulting model, we can see that some additional lines have
# been added indicating the changes:
credit_boost10

# Let's take a look at the performance on our training data:
summary(credit_boost10)


# The classifier made 31 mistakes on 900 training examples for an error rate of 3.4
# percent. This is quite an improvement over the 13.9 percent training error rate we
# noted before adding boosting! However, it remains to be seen whether we see a
# similar improvement on the test data. Let's take a look:
credit_boost_pred10 <- predict(credit_boost10, credit_test)
CrossTable(credit_test$default, credit_boost_pred10, prop.chisq = FALSE, prop.c = FALSE, prop.r = FALSE, dnn = c('actual default', 'predicted default'))

################### END ########################