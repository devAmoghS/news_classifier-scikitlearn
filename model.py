from nltk import PorterStemmer, word_tokenize
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

from news_classifier.data import news


def train(classifier, X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    classifier.fit(X_train, y_train)
    print("Accuracy: %s" % classifier.score(X_test, y_test))
    return classifier


trial1 = Pipeline([
    ('vectorizer', TfidfVectorizer()),
    ('classifier', MultinomialNB())
])  # Accuracy: 0.8425297113752123

trial2 = Pipeline([
    ('vectorizer', TfidfVectorizer(stop_words=stopwords.words('english'))),
    ('classifier', MultinomialNB())
])  # Accuracy: 0.8745755517826825

trial3 = Pipeline([
    ('vectorizer', TfidfVectorizer(stop_words=stopwords.words('english'))),
    ('classifier', MultinomialNB(alpha=0.05))
])  # Accuracy: 0.9140492359932089


def stemming_tokenizer(text):
    stemmer = PorterStemmer()
    return [stemmer.stem(w) for w in word_tokenize(text)]


trial4 = Pipeline([
    ('vectorizer', TfidfVectorizer( stop_words=stopwords.words('english'), tokenizer=stemming_tokenizer)),
    ('classifier', MultinomialNB(alpha=0.05))
])  # Accuracy: 0.9083191850594228

train(trial4, news.data, news.target)
