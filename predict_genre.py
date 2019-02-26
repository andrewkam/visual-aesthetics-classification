import sys
import numpy as np
import elasticsearch
import elasticsearch_dsl
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import KFold, StratifiedKFold, cross_validate, cross_val_score, train_test_split, GridSearchCV
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


def create_cf(cf_name):

    if cf_name == 'nb':
        cf = GaussianNB()
    elif cf_name == 'svm':
        cf = LinearSVC()
        cf.set_params(penalty='l2',
                      loss='squared_hinge',
                      dual=False,
                      C=40)
    elif cf_name == 'dt':
        cf = DecisionTreeClassifier()
        cf.set_params(criterion='entropy',
                      max_depth=180,
                      min_samples_split=120)
    else:
        exit()

    return cf


def tune_params(cf, params, cv, x_data, y_data):
    grid_clf = GridSearchCV(cf, params, scoring='f1_macro', cv=cv, refit=False)
    grid_clf.fit(x_data, y_data)

    print('Best Param: ', grid_clf.best_params_)

    means = grid_clf.cv_results_['mean_test_score']
    stds = grid_clf.cv_results_['std_test_score']

    for mean, std, params in zip(means, stds, grid_clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))


def predict(cf, cv, x_train, y_train):
    scoring = ['accuracy',
               'precision_macro',
               'recall_macro',
               'f1_macro']

    scores = cross_validate(cf,
                            x_train,
                            y_train,
                            scoring=scoring,
                            cv=cv,
                            return_train_score=False)

    output_metrics(scores, scoring)


def output_metrics(scores, scoring):
    for score in scoring:
        print(score, ': ', np.mean(scores['test_' + score]))


def main():
    arg_count = len(sys.argv)

    if arg_count == 2:
        classifiers = ['nb', 'svm', 'dt']
        if sys.argv[1] in classifiers:
            cf = create_cf(sys.argv[1])
        else:
            print('Classifier must be nb, svm, or dt')
    else:
        print('Usage: predict_genre classifier')

    response, query_count = get_elasticsearch_data()
    x_train, y_train, y_dict = format_elasticsearch_data(response, query_count)

    cv = KFold(n_splits=5, shuffle=True, random_state=None)

    # params = {'criterion': ['gini', 'entropy'],
    #           'max_depth': [180, 200, 250],
    #           'min_samples_split': [120, 140, 160]}
    # tune_params(cf, params, cv, x_train, y_train)

    predict(cf, cv, x_train, y_train)


main()
