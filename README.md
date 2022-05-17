# Citizens-Income-Prediction_Comparision-Of-ML-Algorithms

It is essential for any government or an organization to have a rough idea about the benefit plans for the citizens. In recent years, the issue of economic disparity has been a major source of worry. Making the poor's lives better does not appear to be the main criterion for solving this problem. People in the United States feel that rising economic inequality is unacceptably high, and they desire a fair distribution of wealth in society. Some of those plans include the mode of living and how much income, citizens make in the country. This helps in devising plans for the people based on their requirements.

We estimate the income of citizens and classify whether they have income greater than 50,000 dollars per year or not, based on various dependent factors of a person. Then, we process the samples by removing some data inconsistencies and handling the missing values. Further, we study on the features by correlating them and filtering them out. We visualize the key features in the dataset. Finally, We use algorithms like Logistic Regression, K Nearest Neighbours, Naive-Bayes Classifier, Support Vector Machines, Decision Trees, and Random Forest to train the dataset to perform classification.

We use this problem as the basis for the problem of comparison of how different machine learning models perform on a same dataset. We train multiple systems on six different algorithms and compare the models using  some evaluation metrics. We analyse the performance of each algorithm and come to a conclusion on which algorithm works best for the current dataset.

## Table of contents

- [Prerequisites](#prerequisites)
    - [Environment](#environment)
    - [Technologies Used](#technologies-used)
    - [Dataset Description](#dataset-description)

- [Model Implementation](#modules)
    - [Data Preprocessing](#dataprep)
        - [Handling Missing Values](#missingvalues)
        - [Removing inconsistencies in data](#inconsistencies)
        - [Encoding Categorical Data](#encoding)
    - [Feature Study and Selection](#features)
        - [Filtering Features based on Dataset Analysis](#dataset-analysis)
        - [Feature-to-Feature Correlation Analysis](#correlation-analysis)
    - [Data Visualization](#visualization)
    - [Preparing Train and Test Datasets](#traintest)
    - [Training Machine Learning Models](#training-models)
        - [Logistic Regression](#lgr)
        - [K Nearest Neighbors](#knn)
        - [Naive Bayes Classifier](#nbc)
        - [Support Vector Machine](#svm)
        - [Decision Tree](#dt)
        - [Random Forest Classifier](#rfc)
    - [Results - Comparirision of Models](#results)
        - [Classification Report](#cr)
        - [Confusion Matrix](#cf)
        - [Accuracy Scores](#acc)
        - [Precision Scores](#pre)
        - [Recall Scores](#rec)
        - [F1-Scores](#f1)

- [Conclusion](#conclusion)
- [Developers](#developers)
- [Links](#links)
- [References](#references)            

## Prerequisites <a name='prerequisites'></a>

### Environment <a name='environment'></a>

1. Python 3 Environment (Ancaonda preferred)
2. Python modules required:NumPy,Pandas, Scikit-learn, Warnings, Opencv2,Matplotlib, Seaborn
3. Web Browser/Jupyter

OR
- Any Python3 IDE installed with above modules.


### Technologies Used <a name='technologies-used'></a>

1. Anaconda Jupyter Notebook

### Dataset Description <a name='dataset-description'></a>

The data we use for this project is taken from the Adult Income Dataset which is published by Machine Learning Repository at the University of California, Irvine (UCI). This dataset comprised 48,842 samples from 42 countries, each with 14 characteristics. It has eight categorical and six continuous variables that comprise information on age, education, nationality, marital status, relationship status, occupation, job classification, gender, race, weekly working hours, capital loss, and capital gain. The income level is the binomial label in the data set, which predicts whether a person makes more than 50,000 dollars per year or not, based on a set of variables. 

| **    Features   **   | **    Description   **                                                                           |
|-----------------------|--------------------------------------------------------------------------------------------------|
|     age               |     (continuous,   positive integer)      The age of   person.                                   |
|     work-class        |     (categorical, 9   distinct values)      Employment   status of person                        |
|     fnlwgt            |     (continuous,   positive integer)      Number of people   represented by      this row.       |
|     education-num     |     (categorical, 13 distinct values)        The education level,      in numeric form.          |
|     education         |     (categorical, 13 distinct values)        The education level of person.                      |
|     marital-status    |     (categorical, 7   distinct values)      Marital status   of a person.                        |
|     occupation        |     (categorical, 15   distinct values)     Occupation of   Person.                              |
|     relationship      |     (categorical, 6 distinct values)      Relationship of person      in terms of the family.    |
|     race              |     (categorical, 5   distinct values)      Race of the   person.                                |
|     sex               |     (boolean)      Gender of person                                                              |
|     capital-gain      |     (continuous)      Gain of capital   in dollars.                                              |
|     capital-loss      |     (continuous)      Loss of capital   in dollars.                                              |
|     hours-per-week    |     (continuous,   positive integer)      Working hours   per week.                              |
|     native-country    |     (categorical, 41   distinct values)      Native Country   of person.                         |
|     income            |     (boolean)      income of person   per year      given in   brackets  >50K & <= 50K           |



## Model Implementation<a name='modules'></a>

> ### Data Preprocessing <a name = 'dataprep'></a>

#### Handling Missing Values <a name = 'missingvalues'></a>

- When a exploratory data analysis is performed, we have found that there are few entries with value as ‘?’ in three columns namely workclass, occupation and native-country. We considered these values to be missing value entries and when checked there are about 5882 data entries with missing values. All of these are of type string and these features have minimal effect on the outcome of income. Hence, we have removed the data entries having missing values.

#### Removing inconsistencies in data <a name = 'inconsistencies'></a>

-  When the data in the income is analyzed, we found that instead of having two classes i.e., >50K and <=50K, there are four classes >50K, >50K., <=50K. and <=50K. It is obvious that the data entered is having discrepancies – having an extra ‘.’ in the data. Hence, we modify data by removing the abovementioned inconsistency.

#### Encoding Categorical Data <a name = 'encoding'></a>

- As mentioned in the dataset description, the data of most of the Categorical features are non-numeric, we have Label-encoded the data where all features are encoded in alphabetical order starting from 0.

> ### Feature Study and Selection <a name = 'features' ></a>

#### Filtering Features based on Dataset Analysis  <a name = 'dataset-analysis'></a>

- When the dataset is analyzed, we found there are 14 features in the dataset on which the outcome of the income class is said to be directly dependent. But when analyzed more thoroughly we found that few of the features are not relevant to the problem. The feature fnlgwt is the id referencing the survey through which the data entry is collected. This has no effect on the training model. Hence, we have removed it from feature set. And few features are indirectly pointing out few other features. Hence, using these features will just increase the learning time but does not affect the outcome of the models.  We found that education gives the same information as that of education-num, and it is easier to use education-num for model training as this is a numeric feature. Hence, we have removed education from the feature set. Similarly, relationship and marital-status gives almost the same information. Hence, we have decided to remove one feature – relationship. And from the literature survey for the dataset is done, we found that capital-gain and capital-loss are gain and loss a person had on investing and concluded irrelevant to the income of person. Therefore, this feature is removed from the data. 

#### Feature-to-Feature Correlation Analysis <a name = 'correlation-analysis'></a>

- After removing features based on literature analysis, we performed Feature-to-Feature and Feature-to-Label Correlation analysis on the data to get a picture of which features are affecting the outcome of income the least – to remove least useful features. We found marital-status, workclass, occupation, race, native-country are having a correlation of -0.19, 0.016, 0.05, 0.07, 0.02 respectively with income. Therefore, we have not considered the above features for the model training. The final Feature set now consists of age, education-num, sex, hours-per-week for predicting income. Correlation Matrix in the form of a heatmap between the final features is shown in the Figure

![alt tag](https://github.com/kysgattu/Citizens-Income-Prediction_Comparision-Of-ML-Algorithms/blob/main/Project-Images/correlation.png)

> ### Data Visualization <a name = 'visualization'></a>

- All continuous features were visualized using Box and Whisker Plots to readily comprehend the measures of their central inclinations, as shown in Figures

![alt tag](https://github.com/kysgattu/Citizens-Income-Prediction_Comparision-Of-ML-Algorithms/blob/main/Project-Images/age.png)
![alt tag](https://github.com/kysgattu/Citizens-Income-Prediction_Comparision-Of-ML-Algorithms/blob/main/Project-Images/edu.png)
![alt tag](https://github.com/kysgattu/Citizens-Income-Prediction_Comparision-Of-ML-Algorithms/blob/main/Project-Images/sex.png)
![alt tag](https://github.com/kysgattu/Citizens-Income-Prediction_Comparision-Of-ML-Algorithms/blob/main/Project-Images/hours.png)

> ### Preparing Train and Test Datasets <a name = 'traintest'></a>

- The entire dataset is shuffled in a consistent manner, ensuring that all the distinct attribute categories were included in the Training Set and Testing Set. The dataset has now been divided into training and testing sets with seventy percent of the data is made available for training, while the remaining thirty percent is used for testing.

> ### Training Machine Learning Models <a name = 'training-models'></a>


#### Logistic Regression <a name = 'lgr'></a>

-  Logistic Regression is a classifier that works based on learning function of P(Y/X). In this a relationship is established between features and class which is used to detect the class of test data. We create a Logistic Regression model to train the system by minimizing the cost/loss function by using Gradient Descent. Our Model mainly consists of a Sigmoid Function that takes features and weights as input and gives 0 or 1 as the outputs. 
> Sigmoid Function ŷ =g(z)1/(1+e^(-z) )
- After the model makes a prediction, we evaluate the result using the loss function. In this process, we calculate derivatives of the loss function with respect to their weights. Derivatives can tell us which way to modify the weight and by how much to reduce the model's loss. 
- And We update weight until local minimum is found and the model doesn’t improve any further. Now we multiply each derived value with the learning parameter in gradient descent function and subtract it from the weight. Now we fit the model based on the results of gradient descent. We use 1000 iterations with learning rate of 0.5 to fit the model. We use this model to predict the income of persons in the testing data.

#### K Nearest Neighbors <a name = 'knn'></a>

- K Nearest Neighbor Classifier uses the distance between the datapoints to predict the class. In our model we use Euclidean Distance to find out the distance from the data points to the target point and then we arrange the datapoints in the increasing order of their distances .
> Euclidian Distance= √(〖(q1-p1)〗^2+〖(q2-p2)〗^2 )
- Then, we count the n number of nearest datapoints from the target point. Then the number of datapoints in each class is counted and the class which has maximum number of datapoints is assigned to the target class. We have tested the model with multiple values of number of nearest neighbors and found around k value 20, we are getting an optimal result of accuracy. We train the model with number of nearest neighbors as 20 and train the model. We use this model to predict the income of persons in the testing data.

#### Naive Bayes Classifier <a name = 'nbc'></a>

- Naïve Bayes classifier uses Bayes Theorem of probability to define the probability of a data value to be mapped to a particular class. Bayes Theorem states if there are two events A and B then probability of A given B is probability of B given A time probability of A divided by probability of B.
> P(A|B) = P(B|A) * P(A) / P(B)
- In this project, we calculate the probability of features in training set to be part of classes in target. All the features are independent of each other, and we are interested only in class of outcome. Since all the probabilities lie between 0 and 1, multiplying them will result in small value and we might run into overflow problems, therefore, to prevent this we apply log function. Then multiplication becomes addition. Hence, class can be found by 
> y = argmaxy log(P(x1|y)) + log(P(x2|y)) + . . . . +log(P(xn|y)) + log(P(y))
- So, to calculate posterior probability(P(y|X)) i.e., is to predict out class label we need to calculate prior probability (P(y)) which is nothing but frequency of each class in the training data set. for calculating the conditional probability i.e., p(xi|y) we use Gaussian distribution formula.
> p(xi|y)1/(σ√2π)  exp⁡(-1/2  ((x-μ)2)/σ2)
- We calculate prior probability, then calculate conditional probability for each feature and add them all to get posterior probability for that respective class label. Then finally we chose the class label which has the highest value posterior probability. We fit the model using above method using all data entries in training set and use this model to predict the income of persons in the testing data.

#### Support Vector Machine <a name = 'svm'></a>

- Support Vector Machines classify the data by locating a dividing line (or hyperplane) between two classes of data. SVM is an algorithm that takes data as input and, if possible, generates a line that separates the classes. For the current system we use predefined methods in scikit-learn library of python to train the system. In this project, we trained multiple models using different kernels of Support Vector Classifier method defined in SVM. We train the system using 'linear', 'poly', 'rbf', 'sigmoid' kernels separately and calculate train and test scores. Then based on these scores we finalize the kernel which has the best train and test scores and train the SVM Model using that kernel. We found that the polynomial kernel is giving the best scores and we used the ‘poly’ kernel to train the model and we use this model to predict the income of persons in the testing data.

#### Decision Tree <a name = 'dt'></a> 

- Decision Trees uses CART algorithm to perform the classification on a dataset. Tree starts from a particular  feature and bases on the value of the feature it decides which feature to be checked next and this goes on until all the features are used (or specified number of features are used). The important step in the decision tree classifier is to decide which feature is to be selected in each step. For this we use DecisionTreeClassifier method defined in scikit-learn of python. A decision tree can be trained, or the best split can be decided by either using Gini Index or Entropy of Information Gain on the dataset. The DecisionTreeClassifier in scikit-learn takes two input arrays consisting of data features and target space and input and gives a complete decision tree on the features. In our project, we initially train multiple systems by using both the Gini index and entropy criteria and choosing different approaches for maximum features like 'auto' , 'sqrt' and 'log2'. We calculate train and test scores of each decision tree developed and choose the tree with best scores as our final model. By following this method, we found that tree built using Gini index and log2 maximum features is giving best scores. This tree is used to train the model and used to predict income of persons in test data.

#### Random Forest Classifier <a name = 'rfc'></a>

- Random Forest Classifier uses the Ensembling process in which each tree in the ensemble is a decision tree built from a sample of data drawn with replacement from the training data. Selection of these data samples is random and the best split for these trees is found from all input features, or a random set of features selected from features in max_features. Using Random Forests rectifies problems of decision trees like overfitting and high variance. In our model we use RandomForestClassifier defined in scikit-learn library of python. In the similar way as we did in Decision Tree, we build multiple models using Gini index and entropy criteria and choosing different approaches for maximum features like 'auto' , 'sqrt' and 'log2'. And for all cases we used number of trees as hundred which is default parameter defined in scikit-learn method. We calculate train and test scores of each decision tree developed and choose the tree with best scores as our final model. By following this method, we found that model built using Gini index and sqrt maximum features is giving best scores. This model is used to train the model and used to predict income of persons in test data.

> ### Results - Comparirision of Models <a name = 'results'></a>

We have applied the trained models on the features of data in test dataset to predict the outcome i.e., income bracket of person. And then compared the resultant prediction to the corresponding target feature in testing set to map how accurate the trained models are. To compare the models, we use few model evaluation metrics defined in scikit-learn library of python.

#### Classification Reports  <a name = 'cr'></a>

![alt tag](https://github.com/kysgattu/Citizens-Income-Prediction_Comparision-Of-ML-Algorithms/blob/main/Project-Images/clasrep.png)

#### Confusion Matrix  <a name = 'cf'></a>

![alt tag](https://github.com/kysgattu/Citizens-Income-Prediction_Comparision-Of-ML-Algorithms/blob/main/Project-Images/confusion-matrix.png)

#### Accuracy Score  <a name = 'acc'></a>

![alt tag](https://github.com/kysgattu/Citizens-Income-Prediction_Comparision-Of-ML-Algorithms/blob/main/Project-Images/acc.png)

#### Precision Score  <a name = 'pre'></a>

![alt tag](https://github.com/kysgattu/Citizens-Income-Prediction_Comparision-Of-ML-Algorithms/blob/main/Project-Images/pre.png)

#### Recall Score  <a name = 'rec'></a>

![alt tag](https://github.com/kysgattu/Citizens-Income-Prediction_Comparision-Of-ML-Algorithms/blob/main/Project-Images/rec.png)

#### F1 Score  <a name = 'f1'></a>

![alt tag](https://github.com/kysgattu/Citizens-Income-Prediction_Comparision-Of-ML-Algorithms/blob/main/Project-Images/f1.png)

### Conclusion <a name = 'conclusion'></a>

Observing the obtained results of the metric scores of all the trained algorithms we find that Naïve Bayes and Support Vector Machine have best Accuracy Scores; Support Vector Machine have best Precision on classes;  Random Forest has better Recall than other and when analyzed the F1-Score all the algorithms except Logistic Regression have almost 66-68%. And Random Forest and K Nearest Neighbors have better F1-Score. Since the F1-Score is Harmonic mean of Precision and Recall, we consider F1-Score to be more deciding factor in finding most accurate model. And the data each class in the dataset is imbalanced. Hence F1-Score is more feasible than Accuracy. 
Hence, when compared F1-Scores, Random Forest and K Nearest Neighbors have best scores, but when both algorithms are compared, we find Precision is better for the K Nearest Neighbor. And Recall is better for Random Forest. But the precision difference is more for K Nearest Neighbor. Therefore, from all the above analysis we conclude that K-Nearest Neighbor Classifier with 20 nearest neighbors is the best possible model for predicting the income of a person.


## Developer <a name='developers'></a>
* [Kamal Yeshodhar Shastry Gattu](https://github.com/kysgattu)
* [Venkata Sriram Rachapoodi](https://github.com/sriram-rachapoodi)
* [Aditya S Karnam](https://github.com/iamkarnam1999)

## Links <a name='links'></a>

GitHub:     [G K Y SHASTRY](https://github.com/kysgattu)

Contact me:     <gkyshastry0502@gmail.com> , <kysgattu0502@gmail.com>

## References <a name='references'></a>

[Adult Income Dataset](https://archive.ics.uci.edu/ml/datasets/adult)

>**Note:** ***Run Model_Comparision.ipynb or Run Model_Comparision.py***

