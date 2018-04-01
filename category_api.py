import requests, sys, pickle
from time import sleep
from xml.etree import ElementTree
from collections import OrderedDict
import os.path

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

sys.setrecursionlimit(1000000)  # 10000 is an example, try with different values

def get_entries(xml_root, cat_main):
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
                        'cat_sub': cat_sub
                        })                        
    return entries

if __name__ == '__main__':
    
    start = 0
    max_results = 1000
    
    for key, val in CATEGORIES.items():
        
        more_data = True
        while(more_data):
            url = 'http://export.arxiv.org/api/query?search_query=cat:' + \
                str(key) + '&start=' + str(start) + '&max_results=' + str(max_results)

            filename = './files/raw2/' + val.lower().replace(' ', '_') + '_' + str(start) + '.p'

            print(filename)

            if(not os.path.exists(filename)):
                
                print(url)
                data = requests.get(url, timeout=None)
                try:
                    print(val)
                    root = ElementTree.fromstring(data.content)

                    entries = get_entries(root, val)
                    #start += max_results
                    print(len(entries))
                    more_data = True if len(entries) > 0 else False
                    start = (start + max_results) if len(entries) > 0 else 0

                    # Store in Pickle
                    if start != 0:
                        pickle.dump(entries, open(filename, "wb"))
                except:
                    print(sys.exc_info())
                    pass
            else:
                print("file already exist")
                start = start + max_results

            
