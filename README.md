# LOAN-DEFAULT-RISK-PREDICTION
To design and develop predictive machine learning that can identify potential loan defaults based on applicant financial profiles. The project emphasizes the integration of Explainable AI (XAI) techniques to interpret the decision-making process of the model.

#1. Data Preprocessing

	•	Load the dataset
	•	Handle missing values (if any)
	•	Encode categorical variables
	•	Scale numerical features using standardization
	•	Drop irrelevant columns (e.g., LoanID)

#2. Exploratory Data Analysis (EDA)

	•	Analyze class imbalance in the target variable
	•	Explore distributions and correlations among features
	•	Identify key patterns or anomalies in the data

#3. Train/Test Split

	•	Split the dataset into training and testing sets (e.g., 80/20)
	•	Ensure the split is performed before applying SMOTE to avoid data leakage

#4. SMOTE on Training Set

	•	Apply Synthetic Minority Over-sampling Technique (SMOTE) on the training set
	•	Balance the classes to reduce bias toward the majority class

#5. Train Model (XGBoost)

	•	Train the XGBoost classifier using the SMOTE-augmented training data
	•	Compare baseline models (Logistic Regression, Random Forest) to validate choice