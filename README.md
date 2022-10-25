# smoking_activity_prediction

Project Aim
1. The aim of this project is to use Machine Learning algorithms to identify the presence of smoking in the human body based on various body signal inputs.
2. This project also intends to conduct research on the performance comparison of the ensemble methods such as Random Forest, Gradient Boosted Decision Trees, and XGBoost against KNN algorithm.

Research Questions
1. When compared to the performance of the KNN algorithm based on the principle of euclidean distance, can ensemble algorithms show any substantial difference in their performance in the context of a two-class classification?
2. What is the most suitable algorithm for the detection of smoking activity based on body signals?
3. How useful are the body signals in determining the presence of smoking in the human body based?

Objectives
1. Carry out Exploratory Data Analysis in order to get an understanding of the attributes included in the dataset.
2. Clean the data and perform data pre-processing to transform the data into a standard format such that it can be fed through the algorithms.
3. Develop machine learning algorithms efficiently and derive the best possible results with hyperparameter tuning the algorithms whenever necessary.
4. Determine how well the ensemble methods perform against the KNN algorithm in the context of a two-class classification.
5. Evaluate the performance of the classifiers on train and test datasets.

Dataset Information
- The dataset that is being used in this study has over 55.6k data points with 27 different features in the dataset making this a high-dimensional large dataset. 
- The features in the dataset include different body-related information such as an individual’s height, gender, weight, eyesight, haemoglobin, blood pressure, cholesterol, triglyceride, urine protein, serum creatinine, etc., are some of the features. 
- The target variable or the dependent variable has 2 classes on which the whole data is classified. The dataset is highly imbalanced with 63.3% of the data points belonging to the smoking class and the rest 36.7% of the data belonging to non-smokers.
- The dataset that is being used in this study is secondary data that is collected from the Kaggle platform and the dataset can be accessed through the following link:
https://www.kaggle.com/datasets/kukuroo3/body-signal-of-smoking

Data Analysis
1. The dataset is checked for any missing values but the dataset does not have any null values.
2. The dataset is checked for any duplicated values but the dataset does not have any null values.
3. The column names 'ID' is dropped as its irrelevent.
4. Most of the features in the dataset were visualized to identify the patterns in the detection of smoking through body signals.

Observations after EDA
- In the dataset, 63.27% of population are non smokers and 36.73% are smokers.
- Number of male smokers are more than the number of female smokers. A very small population of females has smoking signals in their bodies.
- The dataset has the majority of the population belonging to the age groups between 40-60.
- Most smokers were found in the age group of 40 and the age 35 has the highest percentage of smokers
- The population is well distributed across the people with a height between 150 and 180 cms.
- People over the height of 170 cm, there is a high chance that people are more likely to have smoking signals in their bodies.
- The population is well distributed across the people with a height between 40 and 90 kgs.
- People over weight 75 kgs, there is a high chance that people are more likely to have smoking signals in their bodies.
- Normal systolic blood pressure(Systolic blood pressure measures the pressure in your arteries when your heart beats) being less than 120mm Hg, it is oberved that the number of non smokers with abnormal systolic blood pressure(>120) i.e are higher than the number of smokers with abnormal systolic blood pressure. This unexpected observation may be due to randomness or might imply that smoking does not affect systolic blood pressure.
- Hemoglobin levels are higher in smokers compared to that of non-smokers indicating a high hemoglobin count that occurs most commonly when your body requires an increased oxygen-carrying capacity.
- The population of people with tartar is marginally greater than the population of people without tartar and most percentages of smoking signals were found in people with tartar.
- The 4 features are correlated and hence dropped from the dataset.
- Height, Weight and hemoglobin affect the class "smoking" the most.
- From observing the sample pairplot, the datapoints are highly overlapped, thus linear classification algorithms like Logistic regression and Support Vector Machine cannot be used for model building. Hence we will use KNN algorithm based on the principle of euclidean distance or Decision Tree or Random Forest algorithms which works on the principle of non-linear classification.
- Outliers are detected and removed using IQR(Inter-Quartile Range) method.

Data Pre-Processing
1. Categorical variables are transformed to the numeric representation using label encoder that assigns the numeric label to each category in a column.
2. The balancing of the data was performed using an oversampling technique called SMOTE which increases the number of data points of minority class to match the number of data points in majority class.
3. The dataset is split into a training dataset that has 75% of the whole data and the rest 25% were included in the test dataset.

Model Building
1. Machine Learning algorithms like KNN, Random Forest, Gradient Boosted Decision Tree and XGBoost were used for building the model.
2. Hyperparameter tuning is performed for each algorithms to identify the best parameter value.
3. Confusion matrix is used for each algoritms to understand the number of misclassifications on the test dataset.
4. Evaluation metrics like accuracy, precision, recall and f1-score are used to check the performance of each algorithm.

Results
- For KNN algorithm, the test accuracy is 76.8% with 4,067 misclassications out of total 17,542 datapoints.
- For Random Forest algorithm, the test accuracy is 86.4% with 2,383 misclassications out of total 17,542 datapoints.
- For Gradient Boosted Decision Tree algorithm, the test accuracy is 79.3% with 3,615 misclassications out of total 17,542 datapoints.
- For XGBoost algorithm, the test accuracy is 82.4% with 3,074 misclassications out of total 17,542 datapoints.

Conclusion
- Random Forest algorithm is the most suitable algorithm in determining smoking activity through body signals with the accuracy of 86.4%.
