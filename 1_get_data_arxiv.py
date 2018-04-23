import os.path
from collections import OrderedDict
from xml.etree import ElementTree

import pickle
import requests
import sys

sys.setrecursionlimit(1000000)


def get_entries(xml_root, cat_main):
    """Get entries from XML

    Arguments:
        xml_root {object}   -- xml
        cat_name {text}     -- primary category name

    Returns:
        [array] -- Entry - extracted element from xml
    """

    entries = []

    for r in xml_root.findall('{http://www.w3.org/2005/Atom}entry'):
        id = r.find('{http://www.w3.org/2005/Atom}id').text.replace('http://arxiv.org/abs/', '')
        url = r.find('{http://www.w3.org/2005/Atom}id').text
        title = r.find('{http://www.w3.org/2005/Atom}title').text
        summary = r.find('{http://www.w3.org/2005/Atom}summary').text.replace('\n', ' ')
        cat_sub = r.find('{http://arxiv.org/schemas/atom}primary_category').attrib['term']

        entries.append({'id': id,
                        'url': url,
                        'title': title,
                        'sum': summary,
                        'cat_main': cat_main,
                        'cat_sub': cat_sub})
    return entries


if __name__ == '__main__':

    # -------------------------------------------------------------------------
    # PARAMETERS
    # -------------------------------------------------------------------------

    start = 0
    max_results = 1000
    models_path = './files/models/'

    CATEGORIES = OrderedDict([
        ("cs*", "Computer Science"),
        ("astro-ph*", "Physics"),
        ("cond-mat*", "Physics"),
        ("gr-qc*", "Physics"),
        ("hep-ex*", "Physics"),
        ("hep-lat*", "Physics"),
        ("hep-ph*", "Physics"),
        ("hep-th*", "Physics"),
        ("math-ph*", "Physics"),
        ("nlin*", "Physics"),
        ("nucl-ex*", "Physics"),
        ("nucl-th*", "Physics"),
        ("physics*", "Physics"),
        ("quant-ph*", "Physics"),
        ("math*", "Mathematics"),
        ("q-bio*", "Quantitative Biology"),
        ("q-fin*", "Quantitative Finance"),
        ("stat*", "Statistics"),
        ("econ*", "Economics"),
        ("eess*", "Electrical Engineering and Systems Science")
    ])

    # -------------------------------------------------------------------------
    # ARXIV QUERIES
    # -------------------------------------------------------------------------

    for key, val in CATEGORIES.items():

        # True until query return results
        more_data = True

        while more_data:
            url = 'http://export.arxiv.org/api/query?search_query=cat:' + \
                  str(key) + '&start=' + str(start) + '&max_results=' + str(max_results)

            filename = './files/raw/' + val.lower().replace(' ', '_') + '_' + str(start) + '.p'

            if not os.path.exists(filename):

                print(url)
                data = requests.get(url, timeout=None)
                try:

                    root = ElementTree.fromstring(data.content)
                    entries = get_entries(root, val)

                    print(len(entries))
                    more_data = True if len(entries) > 0 else False
                    start = (start + max_results) if len(entries) > 0 else 0

                    # Store in Pickle
                    if start != 0:
                        pickle.dump(entries, open(models_path + filename, "wb"))
                except:
                    print(sys.exc_info())
                    pass
            else:
                print("file already exist")
                start = start + max_results
