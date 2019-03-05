import sys
import json
import numpy as np
import elasticsearch
import elasticsearch_dsl
import itertools
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import KFold, cross_validate, train_test_split, GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

with open('config.json', 'r') as f:
    config = json.load(f)


def get_elasticsearch_data(service):
    es = elasticsearch.Elasticsearch(['http://localhost:9200/'])
    index_name = config['REPO'][service]['INDEX']

    request = elasticsearch_dsl.Search(using=es, index=index_name,
                                       doc_type='img')
    request = request.source(['categories.category',
                              'categories.score',
                              'chart'])

    # request = request.query('match', chart='(r-b-hip-hop) OR (k-pop)')

    query_count = request.count()
    request = request[0:query_count]
    response = request.execute()

    return response, query_count


def format_elasticsearch_data(response, query_count):
    cat_count = len(response[0].categories)
    # cat_count = 5
    x_cat = np.zeros((query_count, cat_count), dtype=object)
    x_score = np.zeros((query_count, cat_count), dtype=object)
    y_label = np.zeros((query_count, 1), dtype=object)

    for i, img in enumerate(response):
        for j, item in enumerate(img.categories):
            x_cat[i, j] = item.category
            x_score[i, j] = item.score
            y_label[i] = [img.chart]

        # for manual category count
        # for j in range(0, cat_count):
        #     x_cat[i, j] = img.categories[j].category
        #     x_score[i, j] = img.categories[j].score
        #     y_label[i] = [img.chart]

    x_train = tokenize(x_cat, x_score)

    y_label = np.array(y_label)
    label_dict, y_index = np.unique(y_label, return_inverse=True)

    return x_train, y_index, label_dict


def remove_stop_items(doc):
    stop_items = ['ensemble', 'clothing, article of clothing, vesture, wear, wearable, habiliment']
    for item in stop_items:
        doc = list(filter(lambda x: x != item, doc))

    return doc


def tokenize(x_cat, x_score):
    # stop_items = ['ensemble', 'clothing, article of clothing, vesture, wear, wearable, habiliment']

    # cv = CountVectorizer(input='content',
    #                      lowercase=False,
    #                      tokenizer=lambda text: text)

    cv = TfidfVectorizer(input='content',
                         lowercase=False,
                         tokenizer=lambda text: text)

    v_train = cv.fit_transform(x_cat)

    print(v_train * 10.0)

    x_train = v_train.astype('float64')

    # Scores as features
    # feature_names = cv.get_feature_names()
    # for img, (cat_img, score_img) in enumerate(zip(x_cat, x_score)):
    #     img_dict = dict(zip(cat_img, score_img))
    #     for cat in cat_img:
    #         cat_index = feature_names.index(cat)
    #         x_train[img, cat_index] = x_train[img, cat_index] * img_dict[cat]

    x_train = x_train.toarray()

    return x_train


def create_cf(cf_name):

    if cf_name == 'nb':
        cf = GaussianNB()
    elif cf_name == 'svm':
        cf = LinearSVC()
        cf.set_params(penalty='l2',
                      loss='squared_hinge',
                      dual=False,
                      C=0.05)
    elif cf_name == 'dt':
        cf = DecisionTreeClassifier()
        cf.set_params(criterion='gini',
                      max_depth=190,
                      min_samples_split=190)
    else:
        exit()

    return cf


def tune_params(cf, params, x_data, y_data):
    cv = KFold(n_splits=5, shuffle=True, random_state=None)

    grid_clf = GridSearchCV(cf, params, scoring='f1_macro', cv=cv, refit=False)
    grid_clf.fit(x_data, y_data)

    print('Best Param: ', grid_clf.best_params_)

    means = grid_clf.cv_results_['mean_test_score']
    stds = grid_clf.cv_results_['std_test_score']

    for mean, std, params in zip(means, stds, grid_clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))


def predict_cross_valid(cf, x_train, y_train):
    scoring = ['accuracy',
               'precision_macro',
               'recall_macro',
               'f1_macro']

    cv = KFold(n_splits=5, shuffle=True, random_state=None)

    scores = cross_validate(cf,
                            x_train,
                            y_train,
                            scoring=scoring,
                            cv=cv,
                            return_train_score=False)

    output_metrics(scores, scoring)


def predict_split(cf, x_data, y_data, y_dict):
    x_train, x_test, y_train, y_test = train_test_split(
        x_data, y_data, test_size=0.25)

    cf.fit(x_train, y_train)
    y_pred = cf.predict(x_test)

    score = accuracy_score(y_test, y_pred)
    print(score)

    cmatrix = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(cmatrix, y_dict)


def output_metrics(scores, scoring):
    for score in scoring:
        print(score, ': ', round(np.mean(scores['test_' + score]), 3))


def plot_confusion_matrix(cmatrix, y_dict):
    classes = range(len(y_dict))
    classes = y_dict
    plt.figure(figsize=(10, 10))
    plt.imshow(cmatrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar(shrink=0.75, pad=0.01)
    plt.grid(False)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cmatrix.max() / 2.
    for i, j in itertools.product(range(cmatrix.shape[0]),
                                  range(cmatrix.shape[1])):
        plt.text(j, i, cmatrix[i, j],
                 horizontalalignment="center",
                 verticalalignment="center",
                 color="white" if cmatrix[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.show()


def main():
    arg_count = len(sys.argv)

    if arg_count == 3:
        classifiers = ['nb', 'svm', 'dt']
        if sys.argv[2] in classifiers:
            service = sys.argv[1]
            cf = create_cf(sys.argv[2])
        else:
            print('Classifier must be nb, svm, or dt')
    else:
        print('Usage: predict_genre service classifier')

    response, query_count = get_elasticsearch_data(service)
    x_data, y_data, y_dict = format_elasticsearch_data(
        response, query_count)

    # params = {'criterion': ['gini', 'entropy'],
    #           'max_depth': [150, 170, 190, 210],
    #           'min_samples_split': [170, 190, 210, 230, 250]}
    # tune_params(cf, params, x_data, y_data)

    # params = {'criterion': ['gini', 'entropy'],
    #           'max_depth': [10, 20, 30, 40, 50],
    #           'min_samples_split': [10, 20, 30, 40, 50]}
    # tune_params(cf, params, x_data, y_data)

    predict_cross_valid(cf, x_data, y_data)
    # predict_split(cf, x_data, y_data, y_dict)


main()
