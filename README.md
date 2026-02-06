ML Classification of Nigerian Service-Based Tariff (SBT) Bands

Project Objective:
To develop a machine learning model capable of accurately classifying electricity distribution feeders in Nigeria into the five regulatory Service Bands (A–E), as mandated by the Nigerian Electricity Regulatory Commission (NERC). The model is designed to support tariff auditing, infrastructure planning, and performance monitoring.

Data & Contextual Adaptation:
Using a synthetic dataset of 1,000 feeders modeled on real Nigerian grid conditions, the project simulated key local factors:

DisCo-specific variations across major distribution companies (e.g., IKEDC, AEDC).

Infrastructure degradation via feeder age (Gamma-distributed) and functional transformer status, which dynamically reduce supply hours.

Geographic and seasonal influences through Urban/Rural zoning and weather impacts.

Real-world noise by injecting a 5% label error to mimic reporting inaccuracies and billing disputes, ensuring the model learns robust, probabilistic patterns.

Methodology:

Algorithm: XGBoost Classifier, chosen for its efficiency with structured data and ability to handle non-linear relationships.

Preprocessing: Categorical variables (DisCo, zone) were one-hot encoded; target bands were label-encoded.

Critical Anti-Leakage Measure: The feature average_daily_supply was excluded during training. This forced the model to learn from underlying drivers—such as supply stability, location, and infrastructure quality—rather than relying on a direct calculation of the target variable.

Key Results & Insights:

Feature Importance:

Supply stability (measured by standard deviation of daily supply) emerged as the top predictor, highlighting that consistent power delivery is as critical as total supply hours for achieving higher service bands (e.g., Band A).

Model Performance:

High accuracy in classifying feeders at the extremes (Bands A and E).

Some confusion observed among middle bands (B, C, D), reflecting real-world ambiguity where feeders may oscillate between adjacent categories due to borderline performance.

Confusion Matrix Analysis:

Validates the model’s alignment with practical scenarios, where classification challenges are greatest for feeders with overlapping characteristics.

Conclusion & Future Applications:
The model successfully demonstrates that machine learning can be an effective tool for:

Auditing and validating existing band allocations.

Predicting impact of infrastructure investments (e.g., transformer upgrades) on service band improvement.

Supporting data-driven decision-making for DisCos and regulators to prioritize upgrades and enhance grid equity.

Future Scope:

Integration of real-time sensor data for dynamic band assessment.

Expansion to include customer complaint metrics and revenue collection efficiency.

Development of a user-friendly dashboard for regulatory and utility stakeholders.

This project provides a robust, context-aware framework for applying predictive analytics in the Nigerian electricity sector, with potential adaptation to other emerging markets with similar grid challenges
