# Credit Risk Model Project

## Credit Scoring Business Understanding

### 1. Basel II Framework & Model Interpretability
The Basel II Accord heavily influences modern banking regulations by linking a financial institution's capital requirements directly to its risk profile. Under the Internal Ratings-Based (IRB) approach, banks can use their own internal models to estimate key risk parameters:
* **Probability of Default (PD)**
* **Loss Given Default (LGD)**
* **Exposure at Default (EAD)**

Because these estimates dictate the mandatory capital reserves a bank must hold to remain solvent, regulators require strict oversight. An interpretable and well-documented model is crucial for several reasons:
* **Regulatory Auditability:** Regulators must be able to audit the mathematical mechanics, data lineages, and underlying assumptions to ensure the model does not understate risk.
* **Justification of Capital Buffers:** Transparent models provide a verifiable link between asset risk and capital requirements, preventing arbitrary risk manipulation.
* **Anti-Discrimination Compliance:** Financial institutions must legally prove that their scoring algorithms do not rely on prohibited discriminatory variables (e.g., race, gender, religion).
* **Adverse Action Notice:** When a credit applicant is rejected, regulations often mandate that the lender provide specific, actionable reasons (e.g., "debt-to-income ratio too high"), which is only possible with an interpretable model.

### 2. Default Proxies & Associated Business Risks
In credit risk modeling, historical data rarely contains a clean "default" label. Instead, practitioners rely on **proxy variables**, most commonly **"90+ Days Past Due (DPD) within a 12-month performance window."** 

Using a proxy variable is necessary because:
* True insolvency or legal bankruptcy is a rare, lagging indicator that takes years to formalize.
* Financial institutions need a standardized operational metric to flag accounts that have a high statistical probability of unrecoverable loss.

However, proxy-based prediction introduces significant operational and business risks:
* **False Positives (Type I Error):** The model flags a borrower as a default risk because they missed payments due to temporary life disruptions (e.g., changing bank accounts or travel). The business loses profitable interest revenue and damages customer relationships.
* **False Negatives (Type II Error):** The model misses a borrower who systematically pays on day 89 to avoid the 90-DPD threshold. This exposes the bank to toxic, unbacked credit defaults.
* **Economic Cyclicality:** During macroeconomic shocks (e.g., recessions), a 90-DPD proxy might skyrocket, triggering over-conservative lending pullbacks that amplify a credit crunch.

### 3. Model Architecture Trade-Offs in Regulated Finance

When choosing between a traditional **Logistic Regression with Weight of Evidence (WoE)** framework and an advanced **Gradient Boosting Machine (GBM)**, institutions must balance regulatory compliance against predictive power:


| Feature / Dimension | Logistic Regression + WoE | Gradient Boosting (e.g., LightGBM/XGBoost) |
| :--- | :--- | :--- |
| **Predictive Performance** | Lower; assumes linear relationships and struggles with complex feature interactions. | Higher; automatically captures non-linear trends and deep interaction effects. |
| **Interpretability** | **High.** Scorecards show exact points added or subtracted per feature bin. | **Low ("Black Box").** Feature importances show *what* matters, but not *how* a single decision was made. |
| **Regulatory Acceptance** | Universally accepted under Basel IRB standards; easily vetted by risk committees. | Highly scrutinized; often requires complex post-hoc explainability wrappers (e.g., SHAP/LIME). |
| **Operational Stability** | High stability; monotonic WoE transformations prevent erratic scoring behavior. | Lower stability; highly sensitive to data drift and out-of-distribution inputs. |
| **Implementation Complexity**| Low; easily translated into simple SQL case statements or lookup scorecards. | High; requires a robust containerized infrastructure (e.g., FastAPI + Docker) to serve live inference. |

**Strategic Conclusion:** While Gradient Boosting provides superior risk differentiation and minimizes credit losses, Logistic Regression with WoE remains the industry gold standard for core regulatory reporting due to its mathematical safety, transparency, and explicit compliance with consumer protection laws.

## Exploratory Data Analysis (Task 2)

### Dataset Overview

The dataset contains 95,662 transaction records and 16 features describing customer transactions on the Xente eCommerce platform. Features include transaction identifiers, customer information, product details, transaction values, timestamps, pricing strategies, and a fraud indicator.

### Key EDA Findings

#### 1. Severe Class Imbalance

The target variable (`FraudResult`) is highly imbalanced:

* Non-fraud transactions: 95,469 (99.8%)
* Fraud transactions: 193 (0.2%)

This indicates a rare-event classification problem where metrics such as Precision, Recall, F1-Score, ROC-AUC, and PR-AUC will be more informative than accuracy.

#### 2. Highly Skewed Financial Features

The `Amount` and `Value` variables exhibit extreme positive skewness, suggesting that most transactions are small while a limited number are exceptionally large.

#### 3. Significant Outliers

Outlier analysis identified substantial numbers of extreme observations:

* Amount: 25.55% outliers
* PricingStrategy: 16.53% outliers
* Value: 9.43% outliers

These observations may represent unusual customer behavior and potentially fraudulent activity.

#### 4. Strong Correlation Between Amount and Value

Correlation analysis showed that `Amount` and `Value` are nearly perfectly correlated (r = 0.99), indicating significant redundancy between these features.

#### 5. Constant Features

Both `CurrencyCode` and `CountryCode` contain a single unique value throughout the dataset and provide no predictive information. These features are candidates for removal during preprocessing.

### Implications for Feature Engineering

Based on the EDA results, future work will focus on:

* Removing constant features.
* Creating temporal features from transaction timestamps.
* Handling extreme skewness and outliers.
* Addressing class imbalance.
* Building customer-level RFM features.
* Defining a proxy credit-risk target variable for downstream modeling.
## Feature Engineering Pipeline (Task 3)

### Objective

The objective of Task 3 was to build a robust, automated, and reproducible feature engineering pipeline that transforms raw transaction data into a model-ready dataset suitable for credit risk modeling.

The implementation leverages Scikit-learn's `Pipeline` and `ColumnTransformer` to ensure consistency between training and future inference workflows.

---

### Pipeline Architecture

The preprocessing workflow consists of the following stages:

```text
Raw Data
    │
    ▼
Aggregate Customer Features
    │
    ▼
Date-Time Feature Extraction
    │
    ▼
Weight of Evidence (WoE) Encoding
    │
    ▼
Missing Value Imputation
    │
    ▼
Feature Scaling
    │
    ▼
One-Hot Encoding
    │
    ▼
Model-Ready Feature Matrix
```

---

### 1. Aggregate Customer Features

To capture customer transaction behavior, several customer-level aggregate features were generated using `CustomerId`.

| Feature                  | Description                               |
| ------------------------ | ----------------------------------------- |
| TotalTransactionAmount   | Total amount transacted by a customer     |
| AverageTransactionAmount | Mean transaction amount per customer      |
| TransactionCount         | Number of transactions performed          |
| StdTransactionAmount     | Standard deviation of transaction amounts |

These features provide behavioral insights that may be indicative of creditworthiness and fraud risk.

---

### 2. Date-Time Feature Extraction

The `TransactionStartTime` timestamp was decomposed into several temporal features:

| Feature          | Description         |
| ---------------- | ------------------- |
| TransactionHour  | Hour of transaction |
| TransactionDay   | Day of month        |
| TransactionMonth | Month               |
| TransactionYear  | Year                |

Temporal patterns often reveal customer spending habits and fraudulent activity trends.

---

### 3. Weight of Evidence (WoE) Encoding

High-cardinality categorical variables were transformed using Weight of Evidence (WoE).

WoE is widely used in credit scoring because it:

* Creates a monotonic relationship with risk.
* Handles categorical variables efficiently.
* Improves interpretability for logistic regression models.
* Aligns with traditional credit-risk scorecard methodologies.

The following features were encoded using WoE:

* ProviderId
* ProductId
* ProductCategory
* ChannelId
* CountryCode

---

### 4. Categorical Encoding

Low-cardinality categorical features were encoded using One-Hot Encoding.

| Feature      |
| ------------ |
| CurrencyCode |

The encoder was configured with `handle_unknown="ignore"` to support unseen categories during inference.

---

### 5. Missing Value Handling

Missing values were handled automatically within the pipeline.

#### Numerical Features

Median imputation was applied:

```python
SimpleImputer(strategy="median")
```

#### Categorical Features

Most-frequent imputation was applied:

```python
SimpleImputer(strategy="most_frequent")
```

This ensures the pipeline remains robust when encountering incomplete records.

---

### 6. Feature Standardization

Numerical variables were standardized using:

```python
StandardScaler()
```

Standardization transforms features to:

* Mean = 0
* Standard Deviation = 1

This prevents variables with large scales from dominating model learning.

---

### Generated Features

The final pipeline produced 17 model-ready features:

#### Numerical Features

* Amount
* Value
* PricingStrategy
* TotalTransactionAmount
* AverageTransactionAmount
* TransactionCount
* StdTransactionAmount
* TransactionHour
* TransactionDay
* TransactionMonth
* TransactionYear
* ProviderId_WOE
* ProductId_WOE
* ProductCategory_WOE
* ChannelId_WOE
* CountryCode_WOE

#### Encoded Categorical Features

* CurrencyCode_UGX

---

### Pipeline Output

Dataset dimensions before and after feature engineering:

| Dataset           | Shape        |
| ----------------- | ------------ |
| Raw Dataset       | (95,662, 15) |
| Processed Dataset | (95,662, 17) |

The increase in feature count reflects the addition of engineered behavioral, temporal, and WoE-transformed variables.

---

### Reproducibility

The complete preprocessing workflow was encapsulated within a single Scikit-learn Pipeline object and serialized using Joblib.

```python
joblib.dump(
    {
        "pipeline": pipeline,
        "feature_names": feature_names
    },
    "models/data_pipeline.pkl"
)
```

This allows the exact same transformations to be applied consistently during:

* Model training
* Validation
* Production inference

---

### Key Outcomes

* Built a fully automated feature engineering pipeline.
* Generated customer behavioral features.
* Extracted temporal transaction features.
* Applied Weight of Evidence encoding for credit-risk modeling.
* Implemented robust missing-value handling.
* Standardized numerical variables.
* Produced a reproducible, model-ready dataset.
* Saved the fitted preprocessing pipeline for downstream modeling tasks.

The resulting dataset serves as the foundation for proxy target construction, customer risk segmentation, and credit score model development in subsequent tasks.
## Proxy Target Variable Engineering (Task 4)

### Objective

The original dataset does not contain a true loan default label. To enable supervised credit risk modeling, a proxy target variable was engineered using customer transaction behavior.

The objective was to identify customers with low engagement patterns that may indicate elevated credit risk.

---

### RFM Feature Construction

Customer behavior was summarized using the RFM framework:

#### Recency (R)

Measures the number of days since a customer's most recent transaction.

```text
Recency = Snapshot Date - Most Recent Transaction Date
```

#### Frequency (F)

Measures the total number of transactions performed by a customer.

```text
Frequency = Count(TransactionId)
```

#### Monetary (M)

Measures the total transaction value generated by a customer.

```text
Monetary = Sum(Amount)
```

A snapshot date equal to one day after the latest transaction in the dataset was used to ensure consistent recency calculations.

---

### Customer Segmentation

The RFM features were standardized using StandardScaler and segmented using K-Means clustering.

Configuration:

```python
KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)
```

Standardization was applied to ensure that Recency, Frequency, and Monetary contributed equally during clustering.

---

### Cluster Profiles

The resulting customer segments were:

| Cluster | Recency | Frequency | Monetary     |
| ------- | ------- | --------- | ------------ |
| 0       | 61.88   | 7.72      | 81,721       |
| 1       | 12.73   | 34.80     | 272,574      |
| 2       | 29.00   | 4091.00   | -104,900,000 |

---

### High-Risk Cluster Identification

Cluster 0 was selected as the high-risk segment because it exhibited:

* Highest Recency (customers inactive for the longest period)
* Low Frequency (few transactions)
* Relatively Low Monetary activity

These characteristics indicate reduced engagement and a higher likelihood of future inactivity, making this segment an appropriate proxy for elevated credit risk.

---

### Target Variable Creation

A new binary target variable was generated:

```text
is_high_risk
```

| Value | Description        |
| ----- | ------------------ |
| 1     | High-risk customer |
| 0     | Low-risk customer  |

Customers belonging to Cluster 0 were assigned a value of 1.

---

### Target Distribution

| Class         | Count  |
| ------------- | ------ |
| Low Risk (0)  | 84,653 |
| High Risk (1) | 11,009 |

This distribution provides a substantially more balanced target than the original fraud label and is suitable for downstream credit risk modeling.

---

### Dataset Integration

The generated target variable was merged back into the transaction-level dataset using CustomerId.

Final processed dataset:

| Dataset           | Shape        |
| ----------------- | ------------ |
| Raw Dataset       | (95,662, 15) |
| Processed Dataset | (95,662, 18) |

The additional column corresponds to the engineered proxy target variable.

---

### Key Outcomes

* Calculated customer-level RFM metrics.
* Standardized RFM features.
* Segmented customers using K-Means clustering.
* Identified the least engaged customer segment.
* Created a binary proxy credit-risk target variable.
* Integrated the target into the processed dataset.
* Produced a model-ready dataset for supervised credit risk modeling.
