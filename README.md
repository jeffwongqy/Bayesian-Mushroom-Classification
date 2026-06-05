# Bayesian Mushroom Classification: Edible or Poisonous? 

<img width="1300" height="715" alt="mushroom" src="https://github.com/user-attachments/assets/f60b5f7c-0f19-412d-8cc2-24f3f062f876" />

## 1. Introduction
Distinguishing between edible and poisonous mushroom species presents a classic classification challenge due to highly overlapping physical traits. This micro-project utilizes the UCI Mushroom Dataset to analyze how various morphological characteristics - such as cap surface, odor, gill color, etc - impact the probability of a mushroom being toxic. Moving beyond traditional deterministic machine learning models, this study implements a Bayesian Hierarchical Logistic Regression framework using PyMC. By treating the local growth environment (habitat) as a random effect, the model accounts for spatial non-independence and unobserved ecological variance, allowing for a statistically rigorous quantification of predictive uncertainty. 

## 2. Objective
- To build a Bayesian hierarchical logistic regression model that estimates the probability of a mushroom being poisonous based on its physical characteristics.
- To incorporate habitat as a random effect, capturing how localised environments shift the baseline risk of encountering toxic mushrooms via partial pooling.
- To evaluate the stability and reliability of the Markov Chain Monte Carlo (MCMC) chains using trace plots, autocorrelation metrics, and the Gelman-Rubin diagnostic (R).
- To calculate and interpret the posterior odds ratios of categorical predictors, isolating exactly how specific physical traits scale the odds of a mushroom being poisonous.
- To execute a visual posterior predictive check (PPC), confirming that the model's simulated outputs accurately replicate the true underlying distribution of the observed data.

## 3. Preprocessing 
### 3.1 Data Ingestion 
These steps mount your Google Drive environment and load the raw mushroom CSV dataset into a pandas DataFrame and preview its initial structural rows. 

````python
filepath = "/content/drive/MyDrive/Bayesian Mushroom Classification/mushrooms.csv"
mushroom_df = pd.read_csv(filepath)
mushroom_df.head()
````

