import csv
import re
import heapq
from math import sqrt
from itertools import product
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


positive_lemmatized = []
negative_lemmatized = []
train_positive_list = []
train_negative_list = []
test_postitive_list =[]
test_negative_list = []
test_negative_array = []
test_positive_array = []
train_negative_array = []
train_positive_array = []
test_reviews = {}
stop_words = set(stopwords.words('english'))
REPLACE_NO_SPACE = re.compile("[.;:!\'?,\"()\[\]]")
REPLACE_WITH_SPACE = re.compile("(<br\s*/><br\s*/>)|(\-)|(\/)")

def read_data():
    with open ('IMDB_Dataset.csv', encoding="utf8") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        train_count = 0
        train_positive = 0
        train_negative = 0


        next(csv_reader)

        for row in csv_reader:
            clean_string = REPLACE_NO_SPACE.sub("", row[0].lower())
            clean_string = REPLACE_WITH_SPACE.sub(" ", clean_string)

            filtered_words = [w for w in clean_string.split() if not w in stop_words]
            lemmatizer = WordNetLemmatizer()
            lemmatized_text = [' '.join([lemmatizer.lemmatize(word) for word in filtered_words])]
            if row[1] == "positive" and train_positive < 12500:
                train_positive_list.extend(lemmatized_text)
                train_positive_array.append(lemmatized_text)
                train_positive += 1
            elif row[1] == "positive" and train_positive >= 12500:
                test_postitive_list.extend(lemmatized_text)
                test_positive_array.append(lemmatized_text)
                test_reviews[row[0]] = 1
            elif row[1] == "negative" and train_negative < 12500:
                train_negative_list.extend(lemmatized_text)
                train_negative_array.append(lemmatized_text)
                train_negative += 1
            else:            
                test_negative_list.extend(lemmatized_text)
                test_negative_array.append(lemmatized_text)
                test_reviews[row[0]] = 0
    return train_positive_list, train_negative_list, test_reviews


train_p, train_n, test_lemmatized = read_data()
train_lemmatized = train_p + train_n


cv = CountVectorizer(binary=True)
cv.fit(train_lemmatized)
x = cv.transform(train_lemmatized)
x_test = test_lemmatized

target = [1 if i < 12500 else 0 for i in range(25000)]

x_train, x_val, y_train, y_val = train_test_split(
    x, target, train_size = 0.75
)
    
lr = LogisticRegression(C=0.05)
lr.fit(x, target)

stats = {word: coef for word, coef in zip(cv.get_feature_names(), lr.coef_[0])}



correct = 0
line_count = 0

for review in test_lemmatized:
    positive_coef = 0
    negative_coef = 0
    coef = 0
    for word in review.split():
        if word in stats and abs(stats[word]) > 0.1:
            # if stats[word] > 0:
            #     positive_coef += 1
            # else:
            #     negative_coef += 1
            coef += stats   [word]
    # if test_lemmatized[review] == 1 and abs(positive_coef) > abs(negative_coef):
    #     correct += 1
    # if test_lemmatized[review] == 0 and abs(positive_coef) < abs(negative_coef):
    #     correct += 1
    if test_lemmatized[review] == 1 and coef > 0:
        correct += 1
    elif test_lemmatized[review] == 0 and coef < 0:
        correct += 1
    line_count += 1


print("correct: ", correct)
print("accuracy: ", (correct * 100)/float(line_count))

# for best_positive in sorted(stats.items(), key=lambda x: x[1], reverse=True)[:5]:
#     print (best_positive)    

# for best_negative in sorted(stats.items(), key=lambda x: x[1])[:5]:
#     print (best_negative)

# f_scores = open("scores.txt", "w", encoding="utf8")
# f_scores.write(str(scores))
# f_scores.close()

# positive_dict.append(heapq.nlargest(10, positive_count, key=positive_count.get))
# negative_dict.append(heapq.nlargest(10, negative_count, key=negative_count.get))


f_train = open("train_lemmatized.txt", "w", encoding="utf8")
f_train.write(str(train_lemmatized))
f_train.close()

f_test = open("test_lemmatized.txt", "w", encoding="utf8")
f_test.write(str(test_lemmatized))
f_test.close()

f_train = open("train_positive.txt", "w", encoding="utf8")
f_train.write(str(train_positive_array))
f_train.close()

f_test = open("test_negative.txt", "w", encoding="utf8")
f_test.write(str(test_negative_array))
f_test.close()

f_stats = open("coefficients.txt", "w", encoding="utf8")
for word in stats:
    f_stats.write(str(str(word) + " - " + str(stats[word])))
f_stats.close()

# print(heapq.nlargest(10, negative_count, key=negative_count.get))
# print(heapq.nlargest(10, positive_count, key=positive_count.get))

