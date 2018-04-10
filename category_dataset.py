import os, pickle, string, sys

from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


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

def getFilesFromPath(path):
    """return files from folder
    
    Arguments:
        path {string} -- folder path
    
    Returns:
        [array] -- array of file's name
    """
    
    files = []
    for file in os.listdir(path):
        # if file.endswith(".p") and "cs." in file: a corriger car physics est pris en compte
        files.append(os.path.join(path, file))

    return files

    
if __name__ == '__main__':
    
    # Output folder
    path = './files/raw/'

    files_path = getFilesFromPath(path)
    wordnet = WordNetLemmatizer()

    print("Folder name     : " + path)
    print("Number of files : " + str(len(files_path)))

    # Store dataset
    fulldataset = []

    # Open each pickle and start pre-processing
    for file in files_path:
        try:
            with open(file, "rb") as f:
                data = pickle.load(f)
                print(file)

                for record in data:
                    record['sum'] = cleanText(record['sum'])
                    record['title'] = cleanText(record['title'])
                    record['input'] = record['sum'] + ' ' + record['title']
                    record['sum'] = record['sum'].split(';')[0]
                    fulldataset.append(record)
        except:
            print(sys.exc_info())
            print("Error with file : " + file)
            pass

    # Store in Pickle
    pickle.dump(fulldataset, open('dataset.p', "wb"))
    print("File contains " + str(len(fulldataset)) + " records")
