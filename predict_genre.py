import numpy as np
import elasticsearch
import elasticsearch_dsl
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier


def get_elasticsearch_data():
    es = elasticsearch.Elasticsearch(['http://localhost:9200/'])

    request = elasticsearch_dsl.Search(using=es, index='images',
                                       doc_type='img')
    request = request.source(['categories.category',
                              'categories.score',
                              'chart'])

    query_count = request.count()
    request = request[0:query_count]
    response = request.execute()

    return response, query_count


def format_elasticsearch_data(response, query_count):
    cat_count = len(response[0].categories)
    x_cat = np.zeros((query_count, cat_count), dtype=object)
    x_score = np.zeros((query_count, cat_count), dtype=object)
    y_label = np.zeros((query_count, 1), dtype=object)

    for i, img in enumerate(response):
        for j, item in enumerate(img.categories):
            x_cat[i, j] = item.category
            x_score[i, j] = item.score
            y_label[i] = [img.chart]

    x_train = tokenize(x_cat, x_score)

    y_label = np.array(y_label)
    label_dict, y_index = np.unique(y_label, return_inverse=True)

    return x_train, y_index, label_dict


def tokenize(x_cat, x_score):
    cv = CountVectorizer(input='content',
                         lowercase=False,
                         tokenizer=lambda text: text)

    v_train = cv.fit_transform(x_cat)
    x_train = v_train.astype('float64')

    feature_names = cv.get_feature_names()

    for img, (cat_img, score_img) in enumerate(zip(x_cat, x_score)):
        img_dict = dict(zip(cat_img, score_img))
        for cat in cat_img:
            cat_index = feature_names.index(cat)
            x_train[img, cat_index] = img_dict[cat]

    return x_train.toarray()


def main():
    response, query_count = get_elasticsearch_data()
    x_train, y_train, y_dict = format_elasticsearch_data(response, query_count)

    skf = StratifiedKFold(n_splits=5, random_state=None, shuffle=True)

    cf = GaussianNB()
    scores = cross_val_score(cf, x_train, y_train, cv=skf, scoring='f1_micro')
    print(scores)


main()
