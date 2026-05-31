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
