# Twitter Sentiment Analysis

Introduction: 

Around 6000 new tweets are posted every second by Twitter users around the world. This corresponds to roughly 500 million tweets per day, from which valuable insights can be extracted using text mining tools like sentiment analysis. Sentiment analysis has been of great interest to companies and organizations in the areas of business, marketing and politics, as a tool to gauge public reaction towards products launches and campaigns. This project is aimed at using 3 different machine learning models to classify the sentiment of tweets as ‘positive’ or ‘negative’. The strengths and weaknesses of each model will be analyzed and compared to determine the suitability of each for the task.  

Related Work: 

Based on research done in “Probabilistic vs Deterministic Short Text Sentiment Classifiers“ by Ramaneek Gill & Alexandros Tagtalenidis [1] , they have focused on analysing how Bayesian Networks impact the precision and recall of a text sentiment classifier in comparison to the Gaussian class-conditional of a Linear Discriminant analysis model. They then compared the results of probabilistic models - Naive Bayesian Network & Linear Discriminant analysis to deterministic models - SVM & Neural Networks with the task of classifying positive & negative sentiment to see if generative algorithms out-perform deterministic models. 

Dataset: 

The Stanford dataset is used in this project. The training dataset has ~1.6 million tweets. The dataset is processed to remove the emoticons. The dataset is in CSV format with columns/features such as Sentiment of tweet ,Tweet id ,Date of tweet ,Query for finding tweet ,Username and Tweet text.  The test dataset has ~500 tweets. The feature primarily used in classification is sentiment of the tweet. It has ID’s corresponding to the sentiment such as 2 for the Neutral tweet , 4 for a positive tweet and 0 for a negative tweet. Since the Project focuses on classifying tweets with positive and negative sentiments, the tweets with neutral sentiment (i.e sentiment ID = 2) are removed as part of the preprocessing. 
Methodology: 

We implemented two different approaches for semantic analysis of tweets:
* `Bag-Of-Words` Approach
* `Word2Vec Embeddings` Approach

Bag-Of-Words:

This approach involves encoding each tweet based on the presence and absence of the top 2000 words. For example: "hello this is a great day" , In this tweet, `hello` and `great` are among the top 2000 words and this tweet would turn into a 2000 length vector with the position of the words `hello` and `great` having 1's and the rest 0's.
In data preprocessing, we did the following steps:
1. Removed punctuations
2. Removed of stop words
3. Built a global dictionary of words
4. Chose the top 2000 words and used them as features for each tweet
5. Ramaneek Gill and Alexandros Tagtalenidis [1] used both training and test data to build their global vocabulary which has the potential risk of leaking information from test data into the model. To address this issue, we built the global dictionary with just the training data and test the model with unseen (test) data.

Word2Vec Embeddings:

This approach involves converting every word in the vocabulary into a vector representation. The idea is that words with similar meaning will be close to each other in the vector space.
In data preprocessing we did the following:
1. Removed the stop words
2. Removed the punctuations
3. Removed any neutral tweets from both training and test set.
4. Tokenized each tweet
5. Created a word2vec embedding for each word in the vocabulary using the gensim package where each word in the training data is represented as a vector.
6. Created a tf-idf (term frequency - inverse document frequency) for each token in the training data
7. Generated the training data array and test data array by taking a weighted average of each word in the tweet. The weight used is the tf-idf of the word in the training data. 

We implemented three models for classifying tweets: 
1. `Support Vector Machines` (Deterministic model): We implemented SVM as they are good for text classification. SVMs do not scale with large dimensions, hence we employed Principal Component Analysis (PCA) for dimensionality reduction before passing it through the SVM.
2. `Naive Bayes Classifier` (Probabilistic model): We implemented Naive Bayes Classifier as this treats the features as independent of each other and we wanted to study how this bayesian independence assumptions affect the classification. We implemented Gaussian, Bernoulli and Multinomial Naive Bayes models.
3. `Neural Networks` (Deterministic model): We also implemented a neural network and experimented by varying the learning rate and number of hidden layers in the network
After the classification phase, the performance of the algorithms is evaluated according to different measures like accuracy, confusion matrix, precision-recall curve, etc.
