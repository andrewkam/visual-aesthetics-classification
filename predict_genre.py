import sys
import json
import random
import cv2
import numpy as np
import elasticsearch
import elasticsearch_dsl
import itertools
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import KFold, cross_validate, train_test_split, GridSearchCV
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, TruncatedSVD

with open('config.json', 'r') as f:
    config = json.load(f)


def get_elasticsearch_data(service):
    es = elasticsearch.Elasticsearch(['http://localhost:9200/'])
    index_name = config['REPO'][service]['INDEX']

    request = elasticsearch_dsl.Search(using=es, index=index_name,
                                       doc_type='img')
    request = request.source(['categories.category',
                              'categories.score',
                              'chart',
                              'uri'])

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
    image_paths = np.zeros((query_count, 1), dtype=object)

    for i, img in enumerate(response):
        for j, item in enumerate(img.categories):
            x_cat[i, j] = item.category
            x_score[i, j] = item.score
            y_label[i] = [img.chart]
            image_paths[i] = [img.uri]

        # for manual category count
        # for j in range(0, cat_count):
        #     x_cat[i, j] = img.categories[j].category
        #     x_score[i, j] = img.categories[j].score
        #     y_label[i] = [img.chart]

    x_train = tokenize(x_cat, x_score)

    y_label = np.array(y_label)
    label_dict, y_index = np.unique(y_label, return_inverse=True)

    return x_train, y_index, label_dict, image_paths


def remove_stop_items(doc):
    stop_items = [#'ensemble',
                  'clothing, article of clothing, vesture, wear, wearable, habiliment',
                  # 'black',
                  # 'street clothes',
                  # 'man&apos;s clothing',
                  # 'garment'
                  ]
    for item in stop_items:
        doc = list(filter(lambda x: x != item, doc))

    return doc


def tokenize(x_cat, x_score):
    cv = CountVectorizer(input='content',
                         lowercase=False,
                         preprocessor=remove_stop_items,
                         tokenizer=lambda text: text)

    # cv = TfidfVectorizer(input='content',
    #                      lowercase=False,
    #                      binary=True,
    #                      tokenizer=lambda text: text)

    v_train = cv.fit_transform(x_cat)

    x_train = v_train.astype('float64')

    # Scores as features
    # feature_names = cv.get_feature_names()
    # for img, (cat_img, score_img) in enumerate(zip(x_cat, x_score)):
    #     score_sum = sum(score_img)
    #     img_dict = dict(zip(cat_img, score_img/score_sum))
    #     for cat in cat_img:
    #         cat_index = feature_names.index(cat)
    #         x_train[img, cat_index] = x_train[img, cat_index] * img_dict[cat]

    x_train = x_train.toarray()

    return x_train


def create_cf(cf_name):

    if cf_name == 'nb':
        # cf = GaussianNB()
        cf = BernoulliNB()
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

    cv = KFold(n_splits=8, shuffle=True, random_state=None)

    scores = cross_validate(cf,
                            x_train,
                            y_train,
                            scoring=scoring,
                            cv=cv,
                            return_train_score=False)

    output_metrics(scores, scoring)


def predict_split(cf, x_data, y_data, y_dict):
    x_train, x_test, y_train, y_test = train_test_split(
        x_data, y_data, test_size=0.15)

    cf.fit(x_train, y_train)
    y_pred = cf.predict(x_test)

    # score = f1_score(y_test, y_pred, average='macro')
    score = accuracy_score(y_test, y_pred)
    print(score)
    print(classification_report(y_test, y_pred, target_names=y_dict, digits=3))

    cmatrix = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(cmatrix, y_dict)


def output_metrics(scores, scoring):
    for score in scoring:
        print(score, ': ', round(np.mean(scores['test_' + score]), 3))


def format_labels(y_dict):
    matrix_labels = {
        'country-albums': 'Country',
        'r-b-hip-hop-albums': 'R&B/Hip-Hop',
        'rock-albums': 'Rock',
        'k-pop': 'K-Pop'
    }

    classes = []
    for chart in y_dict:
        classes.append(matrix_labels[chart])

    return classes


def plot_confusion_matrix(cmatrix, y_dict):
    # matrix_labels = {
    #     'country-albums': 'Country',
    #     'r-b-hip-hop-albums': 'R&B/Hip-Hop',
    #     'rock-albums': 'Rock',
    #     'k-pop': 'K-Pop'
    # }

    # classes = []
    # for chart in y_dict:
    #     classes.append(matrix_labels[chart])
    classes = format_labels(y_dict)

    plt.figure(figsize=(6, 6))
    plt.imshow(cmatrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Actual vs Predicted Genres', fontweight='bold')
    plt.colorbar(shrink=0.75, pad=0.01)
    plt.grid(False)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    thresh = cmatrix.max() / 2.
    for i, j in itertools.product(range(cmatrix.shape[0]),
                                  range(cmatrix.shape[1])):
        plt.text(j, i, cmatrix[i, j],
                 horizontalalignment="center",
                 verticalalignment="center",
                 color="white" if cmatrix[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('Actual Genre', fontweight='bold')
    plt.xlabel('Predicted Genre', fontweight='bold')
    plt.show()


def visualize_scatter(data_2d, label_ids, y_dict, figsize=(10, 10)):
    classes = format_labels(y_dict)

    plt.figure(figsize=figsize)
    plt.grid()

    # nb_classes = len(np.unique(label_ids))

    for label_id in np.unique(label_ids):
        plt.scatter(data_2d[np.where(label_ids == label_id), 0],
                    data_2d[np.where(label_ids == label_id), 1],
                    marker='o',
                    # color=plt.cm.Set1(label_id / float(nb_classes)),
                    linewidth='1',
                    alpha=0.6,
                    label=classes[label_id])
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()


def visualize_scatter_with_images(X_2d_data, image_paths, figsize=(15, 15), image_zoom=1):

    images = []

    for image_path in image_paths:
        image_path = image_path[0].replace(config['IMAGEDIR']['DEEPDETECT'], config['IMAGEDIR']['LOCAL'])
        image_path = image_path.replace('&apos;', '\'')
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (100, 100))
        images.append(image)

    fig, ax = plt.subplots(figsize=figsize)
    artists = []
    for xy, i in zip(X_2d_data, images):
        x0, y0 = xy
        img = OffsetImage(i, zoom=image_zoom)
        ab = AnnotationBbox(img, (x0, y0), xycoords='data', frameon=False)
        artists.append(ax.add_artist(ab))
    ax.update_datalim(X_2d_data)
    ax.autoscale()
    plt.tight_layout()
    plt.show()


def calc_tsne(x_data, y_data, y_dict, image_paths):
    item_count = 2000

    xy_data = list(zip(x_data, y_data, image_paths))
    random.shuffle(xy_data)
    x_data, y_data, image_paths = zip(*xy_data)

    # pca = PCA()
    # pca_result = pca.fit_transform(x_data)
    x_reduced = TruncatedSVD(n_components=50, random_state=0).fit_transform(x_data)

    tsne = TSNE(n_components=2, verbose=1, learning_rate=200, perplexity=50, n_iter=2000)
    tsne_result = tsne.fit_transform(x_reduced[:item_count])
    tsne_result_scaled = StandardScaler().fit_transform(tsne_result)

    visualize_scatter(tsne_result_scaled, y_data[:item_count], y_dict)
    visualize_scatter_with_images(tsne_result_scaled, image_paths[:item_count])


def main():
    arg_count = len(sys.argv)

    if arg_count == 3:
        classifiers = ['nb', 'svm', 'dt', 'tsne']
        if sys.argv[2] in classifiers:
            service = sys.argv[1]
            if sys.argv[2] != 'tsne':
                cf = create_cf(sys.argv[2])
        else:
            print('Classifier must be nb, svm, or dt')
    else:
        print('Usage: predict_genre service classifier')

    response, query_count = get_elasticsearch_data(service)
    x_data, y_data, y_dict, image_paths = format_elasticsearch_data(
        response, query_count)

    # params = {'criterion': ['gini', 'entropy'],
    #           'max_depth': [150, 170, 190, 210],
    #           'min_samples_split': [170, 190, 210, 230, 250]}
    # tune_params(cf, params, x_data, y_data)

    # params = {'criterion': ['gini', 'entropy'],
    #           'max_depth': [10, 20, 30, 40, 50],
    #           'min_samples_split': [10, 20, 30, 40, 50]}
    # tune_params(cf, params, x_data, y_data)

    # params = {'penalty': ['l2'],
    #           'loss': ['hinge'],
    #           'dual': [True],
    #           'C': [0.0001, 0.001, 0.01, 0.1, 1, 5, 10, 20]}
    # tune_params(cf, params, x_data, y_data)

    if sys.argv[2] != 'tsne':
        predict_cross_valid(cf, x_data, y_data)
        predict_split(cf, x_data, y_data, y_dict)
    else:
        calc_tsne(x_data, y_data, y_dict, image_paths)


main()
