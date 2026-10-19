# Model Imbalance & Baseline Comparison Table

| Experiment      | Model               | Resampling         |   PR-AUC |   ROC-AUC |   Precision |   Recall |     F1 |     F2 |   Expected Cost ($) |   Train Time (s) |
|:----------------|:--------------------|:-------------------|---------:|----------:|------------:|---------:|-------:|-------:|--------------------:|-----------------:|
| XGB_SMOTE       | xgboost             | smote              |   0.8608 |    0.9792 |      0.8245 |   0.8493 | 0.8367 | 0.8442 |             13287.1 |            19.49 |
| XGB_SMOTETomek  | xgboost             | smote_tomek        |   0.8608 |    0.9792 |      0.8245 |   0.8493 | 0.8367 | 0.8442 |             13287.1 |           294.65 |
| XGB             | xgboost             | none               |   0.8592 |    0.9782 |      0.9515 |   0.8055 | 0.8724 | 0.831  |             14074.2 |             9.87 |
| XGB_weighted    | xgboost             | none               |   0.8592 |    0.9757 |      0.9266 |   0.8301 | 0.8757 | 0.8478 |             13869   |            10.61 |
| RF_balanced     | random_forest       | none               |   0.842  |    0.945  |      0.9329 |   0.7616 | 0.8386 | 0.7907 |             18232.4 |            42.84 |
| RF              | random_forest       | none               |   0.8405 |    0.9476 |      0.927  |   0.8    | 0.8588 | 0.8225 |             14937.7 |            62.52 |
| LogReg          | logistic_regression | none               |   0.7417 |    0.9694 |      0.8308 |   0.5918 | 0.6912 | 0.6279 |             28555.2 |             2.48 |
| BalancedRF      | balanced_rf         | none               |   0.7265 |    0.9773 |      0.074  |   0.9014 | 0.1368 | 0.2786 |             49853.5 |             8.29 |
| XGB_RUS         | xgboost             | random_undersample |   0.7237 |    0.9749 |      0.0471 |   0.9123 | 0.0896 | 0.1952 |             74616.2 |             2.08 |
| LogReg_balanced | logistic_regression | none               |   0.7092 |    0.9701 |      0.0715 |   0.9014 | 0.1325 | 0.2714 |             50891.5 |             3.39 |

*Note: Evaluated with 5-fold Stratified CV out-of-fold predictions. Threshold = 0.50.*
