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
This step mounts your Google Drive environment and loads the raw mushroom CSV dataset into a pandas DataFrame and previews its initial structural rows. 

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

- `sigma_habitat` is assigned a HalfNormal(5) priors, ensuring the habitat-effect standard deviation is positive (i.e. to control the variability of habitat coefficients):

$$\sigma_{habitat} \sim HalfNormal(5)$$

- Habitat coefficients (`beta_grasses`, `beta_paths`, etc) follow hierarchical priors and allow habitat effects to share information and reduce overfitting:

$$\beta_{habitat} \sim N(0, \sigma_{habitat})$$

### 4.2 Likelihood
- Habitat categories are converted into binary indicators (`is_grasses`, `is_paths`, etc.), where 1 means the mushroom belongs to that habitat and 0 otherwise.
- The linear predictor combines all predictor effects to calculate the log-odds of a mushroom being poisonous:

$$\eta = \beta_{0} + \sum \beta_{j}X_{j} + \sum \beta_{habitat} H$$

- The inverse logistic function converts log-odds into a probability to ensure that predicted probabilities lie between 0 and 1:

$$p = \frac{1}{1 + e^{-\eta}}$$

- The observed outcome follows a Bernoulli distribution to model whether each mushroom is poisonous (1) or edible (0):

$$y_{i} \sim Bernoulli(p_{i})$$


### 4.3 Posterior
- Bayes theorem combines the priors and likelihood to estimate updated parameter distributions (i.e. update beliefs about predictor effects after observing the data):

$$P(\theta|y) \propto P(y|\theta) P(\theta)$$

- `pm.sample()` uses the NUTS algorithm to generate posterior samples to approximate the posterior distribution and quantify uncertainty in the estimates:

$$\theta^{(1)}, \theta^{(2)}, ..., \theta^{(500)}$$

````python
with pm.Model() as mushroom_model:
  # priors
  beta_0 = pm.Normal('beta_0', mu = 0 , sigma = 5)
  beta_cap = pm.Normal('beta_cap', mu = 0, sigma = 5)
  beta_gill = pm.Normal('beta_gill', mu = 0, sigma = 5)
  beta_odor = pm.Normal('beta_odor', mu = 0, sigma = 5)
  beta_ring = pm.Normal('beta_ring', mu = 0, sigma = 5)
  beta_spore = pm.Normal('beta_spore', mu = 0, sigma = 5)
  beta_stalk = pm.Normal('beta_stalk', mu = 0, sigma = 5)
  beta_veil = pm.Normal('beta_veil', mu = 0, sigma = 5)

  sigma_habitat = pm.HalfNormal('sigma_habitat', sigma = 5)
  beta_grasses = pm.Normal('beta_grasses', mu = 0, sigma = 1) * sigma_habitat
  beta_paths = pm.Normal('beta_paths', mu = 0, sigma = 1) * sigma_habitat
  beta_leaves = pm.Normal('beta_leaves', mu = 0, sigma = 1) * sigma_habitat
  beta_meadows = pm.Normal('beta_meadows', mu = 0, sigma = 1) * sigma_habitat
  beta_urban = pm.Normal('beta_urban', mu = 0, sigma = 1) * sigma_habitat
  beta_waste = pm.Normal('beta_waste', mu = 0, sigma = 1) * sigma_habitat
  beta_woods = pm.Normal('beta_woods', mu = 0, sigma = 1) * sigma_habitat

  # likelihood
  is_grasses = (mushroom_df['habitat'] == 'g').astype(int).values
  is_paths = (mushroom_df['habitat'] == 'p').astype(int).values
  is_leaves = (mushroom_df['habitat'] == 'l').astype(int).values
  is_meadows = (mushroom_df['habitat'] == 'm').astype(int).values
  is_urban = (mushroom_df['habitat'] == 'u').astype(int).values
  is_waste = (mushroom_df['habitat'] == 'w').astype(int).values
  is_woods = (mushroom_df['habitat'] == 'd').astype(int).values

  linear_predictor = (beta_0 + beta_cap * mushroom_df['cap-surface'] +
                      beta_gill * mushroom_df['gill-color'] +
                      beta_odor * mushroom_df['odor'] +
                      beta_ring * mushroom_df['ring-number'] +
                      beta_spore * mushroom_df['spore-print-color'] +
                      beta_stalk * mushroom_df['stalk-surface-above-ring'] +
                      beta_veil * mushroom_df['veil-color'] +
                       (beta_grasses  * is_grasses) +
                      (beta_paths * is_paths) +
                       (beta_leaves * is_leaves) +
                      (beta_meadows * is_meadows) +
                       (beta_urban * is_urban) +
                      (beta_waste * is_waste) +
                       (beta_woods * is_woods))

  p = pm.math.invlogit(linear_predictor)
  y_obs = pm.Bernoulli('y_obs', p = p, observed = y)

  # posterior
  trace = pm.sample(draws = 500,
                    tune = 250,
                    chains = 3,
                    return_inferencedata = True,
                    nuts_sampler = 'numpyro')

````

## 5. Model Diagnostics and Convergence
### 5.1 Trace Plots
The posterior density curves (left column) across all three independent chains overlay each other beautifully, establishing that they converged on the same numerical solutions. Correspondingly, the sampling paths (right column) show perfectly stationary, dense, and tightly integrated histories without any geometric sticking, drifting, or multi-modality. This indicates complete parameter space exploration. 

<img width="1189" height="3189" alt="trace plots" src="https://github.com/user-attachments/assets/877c0432-3750-4518-9dda-3fb38da6e36b" />


### 5.2 Autocorrelation Plots 
The autocorrelation coefficients for nearly all parameters drop off to absolute zero by lag 1 or 2, staying well within the boundaries of statistical insignificance (the shaded gray zones). 

- While the group-level random effects (beta_grasses, beta_paths, etc.) display a slight, expected geometric decay, they safely taper off within 5 to 10 lags.

<img width="2934" height="5510" alt="autocorrelation plots" src="https://github.com/user-attachments/assets/a9df2df8-9123-4004-b76b-89552fa850f1" />


### 5.3 Gelman-Rubin Diagnostic 
The R metric measures the ratio of variance between the chains to the variance within the chains. In this case, every single fixed effect and random effect in the summary table reports an R of exactly 1.00 or 1.01. This mathematically guarantees that the independent chains have reached a stable equilibrium. 

<img width="435" height="375" alt="summary" src="https://github.com/user-attachments/assets/b314bf64-ce07-48d9-b84c-f432dab78d92" />

### 5.4 Bulk Effective Sample Size 
The effective sample size tells us how many independent, uncorrelated draws we managed to extract from the total sampling history. 
- For the primary physical features (fixed effects), the ESS values are massively high, guaranteeing that the posterior means and narrow standard deviation are highly precise.
- For the hierarchical habitat variables, the ESS drops down slightly. This is an entirely expected structural artifact of random effects trying to share information via partial pooling across small sub-group distributions. Any ESS above 100 is completely safe for inference. 


## 6.0 Posterior Predictive Validation
### 6.1 Posterior Predictive Check (PPC)
A Posterior Predictive Check evaluates model fit by using the parameters learned from the data to simulate entirely new, synthetic datasets, then overlaying them against the true ground truth.


<img width="657" height="550" alt="pcc" src="https://github.com/user-attachments/assets/8cf71913-4057-4092-aa36-5e9ba4e25b09" />

- The true underlying distribution of the binary classification data is shown by the solid black line (observed). The model's generated expectation is shown by the dashed orange line (Posterior Predictive Mean). The fact that these two lines overlap almost perfectly proves that the model accurately captures the exact proportion of edible versus poisonous mushrooms in the dataset.

- The individual light blue lines represent unique simulation runs from the posterior draws. They form a clean, tight band symmetrically enveloping the ground truth. This shows that the model's uncertainty is well-calibrated - it is neither wildly overconfident nor completely lost.

## 7. Conclusion
The model exhibits exceptional performance and complete convergence with overlapping and stationary trace chains, and rapid autocorrelation decay proves optimal posterior exploration. Complete equilibrium is mathematically guaranteed by uniform R values with < 1.01 and robust effective sample sizes, while the posterior predictive checks confirms highly accurate, unbiased toxicity classifications. 

## 8. Libraries
- Pandas
- PyMC
- arviz
- Matplotlib

## 9. Dataset
https://archive.ics.uci.edu/dataset/73/mushroom

## References



