### loads the required libraries
# data loading and visualization
import pandas as pd
import numpy as np
import seaborn as sns
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import matplotlib 

# data cleaning and preprossesing
import string
import nltk 
# nltk.download('stopwords')
from nltk.corpus import stopwords
# course youstopwords.words('english')
from sklearn.feature_extraction.text import CountVectorizer

# machine learning libraries to build and train our models
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression # for logistic regression model
from sklearn.ensemble import GradientBoostingClassifier 


### loads the dataset (Amazon_reviews)
# use r before the file path to treat it as a raw string, 
# so Python doesn’t interpret backslashes as escape characters.
# or use double backslashes 
amazon_reviews_data = pd.read_csv("C:\\Users\\kengn\\Desktop\\PA PETER KENGNE\\PROJECTS\\pro_0\\amazon_reviews.csv")
amazon_reviews_data.info()


### Exploratory data analysis
# leverage seaborn to check if we have any missing elements in the dataframee e
# the emply plot tells us there are no null elements
# Also, the info() tells us no null elements are present 
heatmap = sns.heatmap(amazon_reviews_data.isnull(), yticklabels=False, cbar=False, cmap="Blues")
plt.show()
amazon_reviews_data.hist( bins=33, figsize=(11, 3), color='Purple')
plt.show()
# sns.countplot(amazon_reviews_data['feedback'], feedback='feelings')


### plot word cloud
positve_feedback = amazon_reviews_data[amazon_reviews_data['feedback']==1]
negative_feedback = amazon_reviews_data[amazon_reviews_data['feedback']==0]
# converts all the reviews to a single list
# Replace NaN with an empty string
amazon_reviews_data['verified_reviews'] = amazon_reviews_data['verified_reviews'].fillna("")
# Convert the column to a list
reviews = amazon_reviews_data['verified_reviews'].tolist()
# Join the list into a single string
reviews_as_one_string = " ".join(reviews)
# Print the length of the combined string
print(len(reviews_as_one_string))

plt.figure(figsize=(20, 20))
plt.imshow(WordCloud().generate(reviews_as_one_string))
plt.show()


### Data Cleaning
string.punctuation 
def clean_reviews(reviews_as_one_string):
    punctuation_removed = [char for char in reviews_as_one_string if char not in string.punctuation]
    punctuation_removed = ''.join(punctuation_removed)
    stopwords_removed = [word for word in punctuation_removed.split() if word.lower() not in stopwords.words('english') ]
    return stopwords_removed
reviews_cleaned = amazon_reviews_data['verified_reviews'].apply(clean_reviews)
print(reviews_cleaned[5])
vectorizer = CountVectorizer(analyzer=clean_reviews, dtype=np.uint8)
vectorized_reviews = vectorizer.fit_transform(amazon_reviews_data['verified_reviews'])
print(vectorizer.get_feature_names_out)
print(vectorized_reviews.toarray())

vectorized_reviews.shape
X = pd.DataFrame(vectorized_reviews.toarray())
print(X)
Y = amazon_reviews_data['verified_reviews']


### train, deploy and evaluate models
# Naive Bayes classifier model

print(X.shape)
print(Y.shape)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

naive_bayes_classifier = MultinomialNB()
naive_bayes_classifier.fit(X_train, Y_train)

# predicting the test result
Y_predict_test = naive_bayes_classifier.predict(X_test)
cm = confusion_matrix(Y_test, Y_predict_test)
sns.heatmap(cm, annot=True, cmap="Blues", fmt="d")
print("Confusion Matrix:\n", cm)
plt.show()
print(classification_report(Y_test, Y_predict_test))

# Logistic Regression Model
#lrc_model = LogisticRegression()
#lrc_model.fit(X_train, Y_train)

#Y_predict = lrc_model.predict(X_test)

#cm = confusion_matrix(Y_predict, Y_test)
#sns.heatmap(cm, annot=True, cmap="Blues", fmt="d")
#plt.show()
#print(classification_report(Y_test, Y_predict))

# Gradient Boosting Classifier
#gbc_model = GradientBoostingClassifier()
#gbc_model.fit(X_train, Y_train)

#Y_predict = gbc_model.predict(X_test)

#cm = confusion_matrix(Y_predict, Y_test)
#sns.heatmap(cm, annot=True, cmap="Blues", fmt="d")
#plt.show()

#print(classification_report(Y_test, Y_predict))