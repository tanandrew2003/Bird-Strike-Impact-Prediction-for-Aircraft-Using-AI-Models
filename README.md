# Bird Strike Impact Prediction for Aircraft Using AI Models

This project applies machine learning techniques to predict the **severity of aircraft damage following a bird strike** using historical aviation, flight-condition, geographic, and wildlife data.

The objective is to classify each bird strike incident into one of three damage categories:

* **N** – No Damage
* **M** – Medium Damage
* **S_D** – Severe Damage

The model is intended as a **decision-support tool** for identifying potentially serious bird strike incidents and prioritizing further inspection or operational review.

---

## Project Overview

Bird strikes are a persistent aviation safety and operational challenge. While many incidents result in little or no aircraft damage, some can lead to costly repairs, delays, diversions, and serious safety consequences.

This project investigates whether machine learning can use contextual information available around the time of a bird strike to estimate the likely level of aircraft damage.

The analysis focuses on both overall prediction performance and the ability to correctly identify **Severe Damage** cases, where missing a high-risk incident may have greater operational consequences.

---

## Dataset

The project uses the **Aircraft Wildlife Strikes 1990–2023** dataset from Kaggle.

**Source:** [Aircraft Wildlife Strikes 1990–2023](https://www.kaggle.com/datasets/dianaddx/aircraft-wildlife-strikes-1990-2023)

The original dataset contains:

* **288,810 bird strike records**
* **100 variables**

To reduce potential data leakage, variables describing consequences that occurred after the strike were excluded from the modeling process.

The selected predictors mainly describe:

* Time of incident
* Geographic location
* Aircraft characteristics
* Flight conditions
* Wildlife characteristics

After data cleaning and complete-case filtering, **45,066 observations** remained for modeling.

---

## Target Variable

The original damage categories were consolidated into three classes:

| Class   | Meaning       |
| ------- | ------------- |
| **N**   | No Damage     |
| **M**   | Medium Damage |
| **S_D** | Severe Damage |

Before class balancing, the cleaned dataset contained:

| Class | Number of Records |
| ----- | ----------------: |
| N     |            40,410 |
| M     |             3,545 |
| S_D   |             1,111 |

Because the target variable was highly imbalanced, resampling was applied so that the models could learn patterns from all three classes more effectively.

A **stratified 80/20 train-test split** with `random_state = 42` was used for model development and evaluation.

---

## Data Preprocessing

The main preprocessing steps included:

* Removing variables that could cause post-strike data leakage
* Removing `PRECIPITATION` because approximately **96.1% of values were missing**
* Applying complete-case filtering
* Standardizing numerical features when required
* One-hot encoding categorical features
* Consolidating aircraft types into `AIRCRAFT_FAMILY`
* Balancing the three target classes before training

Key numerical variables included:

* `INCIDENT_YEAR`
* `LATITUDE`
* `LONGITUDE`
* `NUM_ENGS`
* `HEIGHT`
* `SPEED`
* `DISTANCE`

Key categorical variables included:

* `TIME_OF_DAY`
* `TYPE_ENG`
* `PHASE_OF_FLIGHT`
* `SKY`
* `SIZE`
* `AIRCRAFT_FAMILY`
* `NUM_STRUCK`
* `INCIDENT_MONTH`

---

## Exploratory Data Analysis

Exploratory analysis was conducted to better understand the relationship between bird strike characteristics and aircraft damage severity.

The results suggested that bird strike outcomes vary across different operational conditions, including:

* Flight phase
* Bird size
* Number of birds struck
* Aircraft characteristics
* Geographic location
* Flight speed and altitude
* Seasonal patterns

These observations suggest that aircraft damage severity is influenced by multiple interacting factors rather than a single predictor.

---

## Machine Learning Models

Four supervised classification models were evaluated:

### 1. Multinomial Logistic Regression

Used as a linear and interpretable baseline model.

### 2. CART Decision Tree

Used as an interpretable nonlinear model and for examining feature importance.

### 3. Random Forest

An ensemble tree-based model designed to capture nonlinear relationships and feature interactions.

### 4. XGBoost

A gradient-boosted tree model used to improve predictive performance through sequential ensemble learning.

---

## Evaluation Metrics

Model performance was evaluated using:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix

Special attention was given to **Severe Damage Recall**, because failing to identify a severe case may have greater consequences than generating an additional false positive.

---

## Feature Engineering

Several additional features were created to improve model performance.

### Cyclical Month Encoding

Seasonality was represented using:

* `MONTH_SIN`
* `MONTH_COS`

This transformation allows months such as December and January to remain close to each other in the feature space.

### Interaction Features

Two interaction variables were created:

`SPEED_HEIGHT = SPEED × HEIGHT`

`DIST_SPEED_RATIO = DISTANCE / SPEED`

These variables capture relationships between multiple flight-condition variables.

### Geographic Clustering

K-means clustering was applied to latitude and longitude.

Values of `k = 2` to `15` were evaluated, and silhouette analysis identified **k = 2** as the preferred clustering solution.

The resulting feature was:

`LOCATION_CLUSTER`

### Height Refinement

Zero `HEIGHT` values recorded during airborne flight phases such as Climb, Descent, and Approach were replaced with the median positive airborne height.

---

## Model Performance

The following results were obtained on the held-out test dataset:

| Model                                     |  Accuracy |  Macro F1 | Severe Damage Recall |
| ----------------------------------------- | --------: | --------: | -------------------: |
| Logistic Regression                       |     60.9% |     0.600 |                60.1% |
| CART                                      |     58.2% |     0.580 |                68.2% |
| Random Forest                             |     67.6% |     0.670 |                78.5% |
| Random Forest – Lower S_D Threshold       |     63.9% |     0.613 |            **87.9%** |
| XGBoost                                   |     71.5% |     0.713 |                80.7% |
| Logistic Regression + Feature Engineering |     60.0% |     0.594 |                61.4% |
| **Random Forest + Feature Engineering**   | **72.3%** | **0.720** |            **79.8%** |
| XGBoost + Feature Engineering             |     71.8% |     0.718 |                79.4% |

---

## Best Performing Model

The **Feature-Engineered Random Forest** achieved the strongest overall performance.

### Performance

* **Accuracy:** 72.3%
* **Weighted F1-score:** 0.721
* **Severe Damage Precision:** 77.4%
* **Severe Damage Recall:** 79.8%
* **Severe Damage F1-score:** 78.6%

Class-specific recall was approximately:

* **No Damage:** 77.0%
* **Medium Damage:** 59.9%
* **Severe Damage:** 79.8%

The Medium Damage category remained the most difficult class to distinguish because it overlaps with both No Damage and Severe Damage cases.

Feature engineering improved Random Forest accuracy from:

**67.6% → 72.3%**

and improved Medium Damage recall from approximately:

**50.0% → 59.9%**

---

## Key Findings

### 1. Ensemble Models Performed Best

Random Forest and XGBoost clearly outperformed Logistic Regression and CART.

This suggests that bird strike damage severity involves nonlinear relationships and feature interactions that simpler models cannot fully capture.

### 2. Feature Engineering Improved Random Forest

Adding seasonal, interaction, geographic, and refined flight-condition features improved Random Forest accuracy from:

**67.6% to 72.3%**

### 3. Several Features Were Important Predictors

Important predictive factors included:

* Bird size
* Number of birds struck
* Phase of flight
* Incident year
* Aircraft characteristics
* Flight speed
* Geographic location

These variables should be interpreted as **predictive associations rather than causal relationships**.

### 4. Severe Damage Detection Involves a Trade-Off

Reducing the classification threshold for Severe Damage increased Severe Damage recall to:

**87.9%**

However, overall accuracy decreased to:

**63.9%**

This highlights the trade-off between maximizing overall prediction performance and prioritizing the detection of potentially severe incidents.

---

## Operational Interpretation

The model could potentially support early post-strike assessment by providing an initial estimate of likely aircraft damage severity.

| Prediction                          | Possible Interpretation                                         |
| ----------------------------------- | --------------------------------------------------------------- |
| Severe Damage – High Confidence     | Prioritize engineering and operational assessment               |
| Severe Damage – Moderate Confidence | Escalate for priority review and collect additional information |
| Medium Damage                       | Conduct appropriate post-strike inspection                      |
| No Damage – High Confidence         | Follow standard post-strike procedures                          |
| Any Class – Low Confidence          | Obtain additional information before relying on the prediction  |

These examples are illustrative only.

Any real-world aviation application would require external validation, threshold calibration, and input from aviation safety and engineering experts.

---

## Limitations

Several limitations should be considered:

* Complete-case filtering reduced the dataset from **288,810 to 45,066 records**, which may introduce selection bias.
* Class balancing means the modeled class distribution does not reflect the natural frequency of damage severity in real-world bird strikes.
* Some predictors, such as bird size and number of birds struck, may not always be immediately observable.
* Geographic clustering provides only a simplified representation of location.
* The model has not been externally validated for operational deployment.
* Model predictions represent statistical associations and should not be interpreted as causal relationships.

---

## Future Improvements

Possible future improvements include:

* External validation by year, airport, or geographic region
* More advanced missing-value handling
* Cost-sensitive learning
* Further threshold optimization for Severe Damage
* Integration of real-time weather information
* Integration of bird radar data
* More detailed geographic and airport-level features
* Incorporating aviation domain-expert knowledge
* Building a deployment-oriented machine learning pipeline

---

## Conclusion

This project demonstrates that machine learning can be used to distinguish among three levels of aircraft damage severity following bird strike incidents using historical operational, aircraft, geographic, temporal, and wildlife information.

The **Feature-Engineered Random Forest** achieved the strongest overall performance with:

**72.3% accuracy**

while the Feature-Engineered XGBoost achieved:

**71.8% accuracy**

The results also show that the preferred model depends on the operational objective.

If overall performance is the priority, the Feature-Engineered Random Forest provides the strongest balance.

If minimizing missed Severe Damage cases is more important, lowering the Severe Damage classification threshold can increase recall to **87.9%**, although this comes at the cost of lower overall accuracy.

Overall, the project highlights both the potential and limitations of machine learning as a decision-support tool for aviation safety.

---

## Tools & Technologies

* Python
* Pandas
* NumPy
* Scikit-learn
* XGBoost
* Matplotlib
* Jupyter Notebook

---

## Full Analysis

The complete data preprocessing, exploratory analysis, feature engineering, model development, and evaluation are available in the Jupyter Notebook:

[`BirdStrikePrediction.ipynb`](./BirdStrikePrediction.ipynb)

---

## Data Source

**Aircraft Wildlife Strikes 1990–2023**

Kaggle Dataset:

https://www.kaggle.com/datasets/dianaddx/aircraft-wildlife-strikes-1990-2023
