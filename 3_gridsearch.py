import logging
import pickle
import time
from pprint import pprint

import numpy as np
import pandas as pd
from sklearn import model_selection, preprocessing
from sklearn.externals import joblib
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.feature_selection import (SelectKBest, chi2, f_classif, f_regression)
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline

# Display progress logs on stdout
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')


def encode_data(d, catname):
    """Encode category with LabelEncoder

    Arguments:
        d {DataFrame}   -- data
        catname {text}  -- primary category name
    Returns:
        [list] -- Class name and list of class id
    """
    le = preprocessing.LabelEncoder()
    le.fit(d[catname])
    return list(le.classes_), le.transform(d[catname])


def open_dataset(filename, catname, nb_element_per_cat, sub_cat_filter=None):
    """Open dataset and filter by category and sub-category

    Arguments:
        filename {text}             -- pipeline
        cat_name {text}             -- primary category name
        nb_element_per_cat {text}   -- number of element per category. Choose randomly
        sub_cat_filter {text}       -- name of the sub category
    Returns:
        [DataFrame] -- data
    """

    # Open dataset
    d = pickle.load(open(filename, "rb"))
    d = pd.DataFrame(d)

    df = pd.DataFrame(columns=d.columns)
    for cat in d[catname].value_counts().index:

        print("category : " + cat + ' has: ' + str(d[catname].value_counts()[cat]))

        # If no filter -> get all categories
        if not sub_cat_filter:
            if d[catname].value_counts()[cat] >= nb_element_per_cat:
                cat1 = d[(d[catname] == cat)].sample(n=nb_element_per_cat)
                df = pd.concat([df, cat1], axis=0)
        else:
            # Get specific categories with filter
            if cat.startswith(sub_cat_filter) and d[catname].value_counts()[cat] >= nb_element_per_cat:
                cat1 = d[(d.cat == cat)].sample(n=nb_element_per_cat)
                df = pd.concat([df, cat1], axis=0)

    print("Total data: " + str(len(d)))
    return df


def perform_grid_search(pipeline, data_in, data_out, catname, models_path):
    """Gridsearch for Pipeline

    Arguments:
        pipeline {object}   -- pipeline
        data_in {list}      -- dataset text input
        data_out {list}     -- dataset category output
        cat_name {text}     -- primary category name
        models_path {text}  -- models path

    Returns:
        [string] -- clean text
    """

    parameters = {
        'vect__max_df': np.arange(0.7, 1, 0.05),
        'vect__ngram_range': ((1, 1), (1, 2)),  # unigrams or bigrams

        'tfidf__use_idf': (True, False),
        'tfidf__norm': ('l1', 'l2'),

        'kbest__k': np.arange(3000, 15000, 500),
        'kbest__score_func': (chi2, f_classif, f_regression),

        'SGD__loss': ('log', 'modified_huber'),
        'SGD__penalty': ['l2'],
        'SGD__alpha': [0.001, 0.0001, 0.00001, 0.000001, 0.0000001, 0.00000001]
    }

    # Gridsearch
    grid_search = RandomizedSearchCV(pipeline, parameters, n_jobs=4, verbose=5, n_iter=200)

    print("Parameters:")
    pprint(parameters)

    # Fitting
    t0 = time.time()
    grid_search.fit(data_in, data_out)
    print("done in %0.3fs" % (time.time() - t0))

    # Results
    print("Best score: %0.3f" % grid_search.best_score_)
    print("Best parameters set:")
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))

    # Save best_params, estimator and classlist in Pickle
    current_time = time.strftime("%Y%m%d-%H%M%S")

    best_params_filename_ = models_path + 'best_parameters_' + catname + '_' + str(current_time) + '.pkl'
    joblib.dump(best_parameters, best_params_filename_, compress=1)

    best_estimator_filename_ = models_path + 'estimator_' + catname + '_' + str(current_time) + '.pkl'
    joblib.dump(grid_search.best_estimator_, best_estimator_filename_, compress=1)

    classlist_filename_ = models_path + 'classlist_' + catname + '_' + str(current_time) + '.pkl'
    joblib.dump(classList, classlist_filename_, compress=1)

    return best_params_filename_, classlist_filename_


if __name__ == "__main__":

    # -------------------------------------------------------------------------
    # PARAMETERS
    # -------------------------------------------------------------------------

    cat_name = 'cat_main'  # cat_main: main category (ex: Computer Science) | cat_sub : sub category (ex:cs.CL)
    element_per_cat = 8300
    test_size = 0.33
    seed = 7
    models_path = './files/models/'

    # -------------------------------------------------------------------------
    # PIPELINE
    # -------------------------------------------------------------------------

    pipeline = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('kbest', SelectKBest()),
        ("SGD", SGDClassifier()),
    ])

    # -------------------------------------------------------------------------
    # OPEN DATASET AND ENCODE OUTPUT
    # -------------------------------------------------------------------------

    # create dataset from file
    data = open_dataset('files/processed/dataset.p', cat_name, element_per_cat)

    # encode output with labelencoder
    classList, encoded_output = encode_data(data, cat_name)
    out = encoded_output.tolist()
    inp = data.input.tolist()

    # -------------------------------------------------------------------------
    # GRIDSEARCH
    # -------------------------------------------------------------------------

    best_params_filename, classlist_filename = perform_grid_search(
        pipeline, inp, out, cat_name, models_path)

    # -------------------------------------------------------------------------
    # MODEL AND CONFUSION MATRIX - CLASSIFICATION REPORT
    # -------------------------------------------------------------------------

    # Split dataset
    X_train, X_test, y_train, y_test = model_selection.train_test_split(inp, out, test_size=test_size,
                                                                        random_state=seed)
    # Load best_params and class list
    best_params = joblib.load(best_params_filename)
    classList = joblib.load(classlist_filename)

    # Pipeline - Set previous parameters
    pipeline.set_params(**best_params)
    pipeline.fit(X_train, y_train)

    # predict test instances
    y_preds = pipeline.predict(X_test)

    # confusion matrix
    matrix = confusion_matrix(y_test, y_preds)
    print(matrix)

    # classification report
    print(classification_report(y_test, y_preds, target_names=classList))
