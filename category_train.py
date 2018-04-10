from __future__ import print_function

from pprint import pprint
from time import time
import logging, pickle, string
import pandas as pd
import numpy as np
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_selection import SelectKBest, chi2, f_classif, f_regression
from sklearn.metrics import confusion_matrix
from sklearn import model_selection
from sklearn.preprocessing import OneHotEncoder
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import RandomizedSearchCV

# Display progress logs on stdout
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')

wordnet = WordNetLemmatizer()

def cleanText(text):
    """Clean raw text using different methods :
       1. tokenize text
       2. lower text
       3. remove punctuation
       4. remove non-alphabetics char
       5. remove stopwords
       6. lemmatize

    Arguments:
        text {string} -- raw text

    Returns:
        [string] -- clean text
    """

    # split into words
    tokens = word_tokenize(text)
    # convert to lower case
    tokens = [w.lower() for w in tokens]
    # remove punctuation from each word
    table = str.maketrans('', '', string.punctuation)
    stripped = [w.translate(table) for w in tokens]
    # remove remaining tokens that are not alphabetic
    words = [word for word in stripped if word.isalpha()]
    # filter out stop words
    stop_words = set(stopwords.words('english'))
    words = [w for w in words if not w in stop_words]

    stemmed = [wordnet.lemmatize(word) for word in words]

    return ' '.join(stemmed)


# #############################################################################
# Load some categories from the training set
# #############################################################################

def encodeData(data):
    le = preprocessing.LabelEncoder()
    le.fit(data[cat_name])
    return list(le.classes_), le.transform(data[cat_name])

def openDataset(filename, cat_name, nb_element_per_cat, sub_cat_filter=None):

    # Open dataset
    data = pickle.load(open(filename, "rb"))
    data = pd.DataFrame(data)

    df = pd.DataFrame(columns=data.columns)
    for cat in data[cat_name].value_counts().index:

        print("category : " + cat + ' has: ' + str(data[cat_name].value_counts()[cat]))

        # If no filter -> get all categories
        if(not sub_cat_filter):
            if data[cat_name].value_counts()[cat] > nb_element_per_cat:
                cat1 = data[(data[cat_name] == cat)].sample(n=nb_element_per_cat)
                df = pd.concat([df, cat1], axis=0)
        else:
            # Get specific categories with filter
            if cat.startswith(sub_cat_filter) and data[cat_name].value_counts()[cat] > nb_element_per_cat:
                cat1 = data[(data.cat == cat)].sample(n=nb_element_per_cat)
                df = pd.concat([df, cat1], axis=0)

    print("Total data: " + str(len(data)))
    return df

def getClassNameFromProba(probaArray, enc):
    print(probaArray)
    print(enc)
    idx = probaArray.flatten().argmax(axis=0)
    return enc[idx]


def performGridSearch(pipeline, data, cat_name):
    
    parameters = {
        'vect__max_df': np.arange(0.5, 1, 0.05),
        #'vect__max_features': (None, 5000, 10000, 50000),
        'vect__ngram_range': ((1, 1), (1, 2)),  # unigrams or bigrams
        'tfidf__use_idf': (True, False),
        'tfidf__norm': ('l1', 'l2', None),

        'kbest__k': np.arange(500, 15000, 500),
        'kbest__score_func': (chi2, f_classif, f_regression),
        #'nb__alpha': (1e-3, 1e-4),

        #'clf__alpha': (0.00001, 0.000001, 0.0000001),
        #'clf__penalty': ('l2', 'elasticnet'),
        ##'clf__n_iter': (10, 50, 80),

        #'tree__n_estimators': (500,1000,2000),
        #'tree__max_features': (32, 64)

        #'SGD__seed': [0],
        'SGD__loss': ('log', 'hinge'),
        'SGD__penalty': ['l1', 'l2', 'elasticnet'],
        'SGD__alpha': [0.001, 0.0001, 0.00001, 0.000001, 0.0000001, 0.00000001]
    }
    
    grid_search = RandomizedSearchCV(pipeline, parameters, n_jobs=-1, verbose=10)

    print("Performing grid search...")
    print("pipeline:", [name for name, _ in pipeline.steps])
    print("parameters:")
    pprint(parameters)
    t0 = time()
    print(data[cat_name])
    grid_search.fit(data.input, data[cat_name])
    print("done in %0.3fs" % (time() - t0))
    print()
    print("Best score: %0.3f" % grid_search.best_score_)
    print("Best parameters set:")
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))
    

if __name__ == "__main__":

    # Define
    cat_name = 'cat_main'
    element_per_cat = 8300
    test_size = 0.33
    seed = 7

    pipeline = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),

        # Dimention reduction
        # Find the 4000 most informative columns
        ('kbest', SelectKBest()),

        #('clf', SGDClassifier()),
        #('tree', ExtraTreesClassifier())

        #('SVC', SVC(probability=True)))
        #('clf', DecisionTreeClassifier(max_depth=10),)
        ("SGD", SGDClassifier(loss='modified_huber')),
        #("ASGD", SGDClassifier(average=True)),
        #("Passive-Aggressive I", PassiveAggressiveClassifier(loss='hinge', C=1.0)),
        #("Passive-Aggressive II", PassiveAggressiveClassifier(loss='squared_hinge', C=1.0)),
        #('ridge', RidgeClassifier(tol=1e-2, solver="lsqr"))
        #('nb', MultinomialNB('''fit_prior=False'''))
    ])

    # create dataset from file
    data = openDataset('files/processed/dataset.p', cat_name, element_per_cat)

    performGridSearch(pipeline, data, cat_name)

    # encode output with labelencoder
    classList, encoded_output = encodeData(data)
    out = encoded_output.tolist()
    inp = data.input.tolist()

    # split dataset
    X_train, X_test, y_train, y_test = model_selection.train_test_split(inp, out, test_size=test_size, random_state=seed)

    # pipeline
    pipeline.fit(X_train, y_train)

    # predict test instances
    y_preds = pipeline.predict(X_test)

    # confusion matrix
    matrix = confusion_matrix(y_test, y_preds)
    print(matrix)

    # classification report
    print(classification_report(y_test, y_preds, target_names=classList))
    
    # TEST

    t = [cleanText("At some point you just need to stop looking and be blissfully ignorant...this was not one of those days. In and update to my previously updated blog article, I have found another instance where the plaintext password was written to system logs. This time I found it in more persistent log. This is actually a worse problem than the one I previously reported on. The previous examples were found in the unified logs which can hang around for a few weeks, this new example stores the exact same information in the system's / var / log / install.log. I have found that the install.log will only get wiped out upon major re - installation(ie: 10.11 -> 10.12 -> 10.13), therefore these plaintext passwords will hang around for quite a bit longer than a few weeks! I had entries dating back to when I originally installed High Sierra on this system back in November of 2017! Twitter user @sirkkalap, was unable to re - create what I previously reported on. I finally got some time this afternoon to re - test. As it turns out, I was unable to re - create my results from 03 / 24. I assumed that at some point in the past few days a silent security update was pushed out. I went to my install.log file to investigate further. As far as updates go - the only thing that has potential to be the cause of the fix is a GateKeeper ConfigData update v138(com.apple.pkg.GatekeeperConfigData.16U1432). I have not investigated if this was the true cause. I have not updated to 10.13.4 yet, this was on 10.13.3. During this investigations I was VERY surprised to see the same diskmanagementd logs that I had found in the unified logs. Why are they logged in the software installation log at all, I have no clue. It makes absolutely no sense to me.")]

    r = pipeline.predict_proba(t)
    className = getClassNameFromProba(r, classList)
    print(className)

    t = [cleanText('International Union for the Conservation of Nature, the body that administers the world’s official endangered species list, announced yesterday that it was moving the giraffe from a species of Least Concern to Vulnerable status in its Red List of Threatened Species report. That means the animal faces extinction in the wild in the medium - term future if nothing is done to minimize the threats to its life or habitat. The next steps are endangered, critically endangered, extinct in the wild and extinct. RELATED CONTENT \
                   How America Can Help Save a Non - American Species: The Mighty Giraffe\
                   Poaching of elephants and rhinoceros and the illegal trade in pangolins has overshadowed the problems with giraffes in the last decade. But Damian Carrington at The Guardian reports that giraffe numbers have dropped precipitously in the last 31 years, from 157, 000 individuals in 1985 to 97, 500 at last count.\
                   “Whilst giraffes are commonly seen on safari, in the media and in zoos, people—including conservationists—are unaware that these majestic animals are undergoing a silent extinction, ” Julian Fennessy, the co - chair of the IUCN’s Species Survival Commission’s Giraffe and Okapi Specialist Group says in a press release. “With a decline of almost 40 percent in the last three decades alone, the world’s tallest animal is under severe pressure in some of its core ranges across East, Central and West Africa. As one of the world’s most iconic animals, it is timely that we stick our neck out for the giraffe before it is too late.”\
                   The giraffes face two main threats, encroachment from cities and towns into their habitat and poaching. Poaching has become increasingly problematic. Some food insecure villagers kill the animals for their meat, but Jani Actman at National Geographic reports many giraffes are slaughtered just for their tails, which are considered a status symbol and have been used as a dowry when asking a bride’s father for his daughters hand in marriage in some cultures.\
                   Patrick Healy at The New York Times reports that red list divides the giraffe into nine subspecies. Five of those subspecies are decreasing in numbers while two populations are increasing and one is stable. West African giraffes, the smallest subspecies, have grown from 50 individuals in the 1990s to about 400 today. But that success took a massive amount of effort from the government of Niger and conservation groups.\
                   It will take similar efforts throughout the giraffe’s wide range to arrest its plummeting numbers. Derek Lee, founder of the Wild Nature Institute who contributed to the IUCN update tells Healy that both poaching and habitat encroachment need to be stopped to save the giraffe. “These are problems everywhere for giraffes, ” he says. “You need to stop both threats.”\
                   While increasing funding for anti - poaching efforts can do some good, Lee thinks stopping habitat encroachment is a much more difficult prospect, since it would mean interfering with land development, mining and other economic activities and livelihoods.\
                   The biggest problem for giraffes, though, may be the lack of attention over the years. “I am absolutely amazed that no one has a clue, ” Julian Fennessy, executive director of Giraffe Conservation Foundation tells Sarah Knapton at The Telegraph. “This silent extinction. Some populations less than 400. That is more endangered than any gorilla, or almost any large mammal in the world.”\
                   “There’s a strong tendency to think that familiar species(such as giraffes, chimps, etc.) must be OK because they are familiar and we see them in zoos, ” Duke University conservation biologist Stuart Pimm, tells the Associated Press. In fact, giraffes have silently been going extinct across Africa over the last century. The animal is already gone from seven countries, Burkina Faso, Eritrea, Guinea, Malawi, Mauritania, Nigeria and Senegal.\
                   Read more: https: // www.smithsonianmag.com / smart - news / giraffes - silently - slip - endangered - species - list - 180961372 0GE7SgHgQRRboBTT.99 Give the gift of Smithsonian magazine for only $12! http: // bit.ly / 1cGUiGv Follow us: @SmithsonianMag on Twitter')]

    r = pipeline.predict_proba(t)
    className = getClassNameFromProba(r, classList)
    print(className)

