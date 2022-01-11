# End-to-end-ML-Project-Chapter-Summary
This repo contains a summary of the second chapter of Hands-On Machine Learning with Scikit-Learn and Tensorflow by Aurélien Géron
No copyright violation intended! The only purpose is to provide my view of what I understood while reading the chapter


**TDLR:**
1. Look at the big picture
2. Get the data
3. Discover and visualize the data to gain insights
4. Prepare the data for Machine Learning Algorithms
5. Select a model and train it
6. Fine-tune your model
7. Present your solution
8. Launch, monitor, and maintain your system

# Look at the big picture

**Frame the problem:**

1. The first question to ask your boss is what exactly is the business objective; building a model is probably not the end goal. How does the company expect to use and benefit from this model? This is important because it will determine how you frame the problem, what algorithms you will select, what performance measure you will use to evaluate your model, and how much effort you should spend tweaking it.
2. The next question to ask is what the current solution looks like (if any). It will often give you a reference performance, as well as insights on how to solve the problem

With this information, you need to frame the problem: is it supervised, unsupervised or reinforcement learning? Is it a classification task, a regression task, or something else? Should you use batch learning or online learning techniques? 

**Select a performance measure:**

- The next step is to select a performance measure

# Get the data:

**Download data:**

- Maybe you already have data. If not, you will need to find it

**Take a quick look at the data structure:**

- Open a jupyter notebook and take a quick look:
    - Check a few rows using head()
    - Get a quick description of the data using info() - total number of rows, and each attribute’s type and number of non-null values
    - Check the summary of the numerical attributes using describe() (The 25%, 50%, and 75% rows show the corresponding percentiles: a percentile indicates the value below which a given percentage of  observations in a group of observations falls.)
    - Another quick way to get a feel of the type of data you are dealing with is to plot a histogram for each numerical attribute. df.hist(bins=50, figsize=(20,15))

**Create a Test Set**

- Create a test set right at the beginning and set it aside
- Only use it right at the end, when you’ve already done cross-validation and you have only 2 or 3 models to decide.

# Discover and Visualize the data to gain insights

1. Use different kinds of visualization
2. Look for correlations using .corr() or scatter_matrix (pandas method)
3. Experimenting with attribute combinations
    1. One last thing you may want to do before actually preparing the data for ML algorithms is to try out various attribute combinations that make sense given the problem

# Prepare the data for Machine Learning Algorithms

Pro tip: write functions that prepare the data. Then you will gradually build a library of transformation functions that you can reuse in future projects. You can also use these functions in your live system to transform the new data before feeding it to your algorithms.

If in any data cleaning function we use predefined values, we should compute those values (ex: median value) on the training set, and use them in the training set but also use the same value in the test set.

**Data Cleaning:**

1. Missing data
2. Outliers
3. Duplicates
4. Categorical Values

**Feature Engineering:**

1. Feature Generation - Create new features 
2. Dimensionality reduction techniques
3. Feature Scaling
    1. Min-max scaling (normalization) - values are shifted and rescaled so that they end up ranging from 0 to 1
    2. Standardization - first it subtracts the mean value (so standardized values always have a zero mean), and then it divides by the standard deviation so that the resulting distribution has unit variance. unlike min-max scaling, standardization does not bound values to a specific range, which may be a problem for some algorithms (e.g., neural networks often expect an input value ranging from 0 to 1).

**Use transformation pipelines to orchestrate all the methods ⇒** from sklearn.pipeline import Pipeline or from sklearn.compose import ColumnTransformer

# Select and Train Models

In this step, we are going to simply test multiple models without tweaking much the hyperparameters

Use scikit-learn’s k-fold cross_Validation method ⇒ from sklearn.model_selection import cross_val_score

**The goal is to shortlist a few (two to five) promising models.** ⇒ test multiple models (checking the results and seeing if they overfit or underfit) without tweaking the hyperparameters. If they underfit, test a more complex model. If they overfit test regularization for example

# Fine-Tune your model

Now that we have a shortlist of promising models, you need to fine-tune them

We can use grid search or random search. Maybe, in the beginning, random search it’s a better option. Then, when we have a more a less well-defined set of good hyperparameters range, use grid search to find the best combination

**Evaluate the system on the test set ⇒** After tweaking your models for a while, you eventually have a system that performs sufficiently well. Now is the time to evaluate the final model on the test set.

# Launch, monitor, and maintain your system

1. Write monitoring code to check your system live performance at regular intervals and trigger alerts when it drops
2. Evaluating your system’s performance will require sampling the system’s predictions and evaluating them. This will generally require human analysis. These analysts may be field experts, or workers on a crowdsourcing platform
3. You should also make sure you evaluate the system’s input data quality. Sometimes performance will degrade slightly because of a poor quality signal (e.g., a malfunctioning sensor sending random values, or another team’s output becoming stale), but it may take a while before your system’s performance degrades enough to trigger an alert.
4. Finally, you will generally want to train your models on a regular basis using fresh data. You should automate this process as much as possible.
