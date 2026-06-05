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
This step mount your Google Drive environment and load the raw mushroom CSV dataset into a pandas DataFrame and preview its initial structural rows. 

````python
filepath = "/content/drive/MyDrive/Bayesian Mushroom Classification/mushrooms.csv"
mushroom_df = pd.read_csv(filepath)
mushroom_df.head()
````

### 3.2 Target Variable Mapping
This step performs binary custom mapping on the raw target column, converting the class 'e' (edible) into 0 and 'p' (poisonous) into 1. 

````python
def class_label(row):
  if row['class'] == 'e':
    return 0
  else:
    return 1

mushroom_df['poisonous'] = mushroom_df.apply(class_label, axis = 1)

````

### 3.3 Feature Selection 
This step isolates the specific categorical morphological traits from the dataset that will serve as the predictor variables for the regression model. 

````python
features = ['cap-surface', 'gill-color', 'odor', 'ring-number',
            'spore-print-color', 'stalk-surface-above-ring',
            'veil-color']
````

### 3.4 Predictor Encoding 
This step uses the `OrdinalEncoder` to transform the text categories across all selected physical mushroom traits into dense numerical integer codes for algebraic matrix multiplication. 

````python
oe = OrdinalEncoder()
mushroom_df[features] = oe.fit_transform(mushroom_df[features])
````

### 3.5 Target Encoding 
This step uses `LabelEncoder` to transform the target class into numerical integer codes.  

````python
le = LabelEncoder()
y = le.fit_transform(mushroom_df['poisonous'])
````

## 4. Model Training 
### 4.1 Priors
- `beta_0`, `beta_cap`, `beta_gill`, `beta_odor`, `beta_ring`, `beta_spore`, `beta_stalk`, and `beta_veil` are assigned Normal(0, 5) priors, expressing the belief that each predictor's effect is likely near zero but can vary widely (i.e. represent uncertainty about predictor effects before seeing the data):

$$\beta_{j} \sim N(0, 5)$$

- `sigma_habitat` is assigned a HalfNormal(5) prior to ensure the habitat-effect standard deviation is positive (i.e. to control the variability of habitat coefficients):

$$\sigma_{habitat} \sim HalfNormal(5)$$

- Habitat coefficients (`beta_grasses`, `beta_paths`, etc) follow hierarchical priors and allow habitat effects to share information and reduce overfitting:

$$\beta_{habitat} \sim N(0, \sigma_{habitat})$$

### 4.2 Likelihood
- Habitat categories are converted into binary indicators (`is_grasses`, `is_paths`, etc.), where 1 means the mushroom belongs to that habitat and 0 otherwise.
- The linear predictor combines all predictor effects:

$$\eta = \beta_{0} + \sum \beta_{j}X_{j} + \sum \beta_{habitat} H$$


