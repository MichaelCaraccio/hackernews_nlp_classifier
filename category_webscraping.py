import requests
from bs4 import BeautifulSoup
import pickle
from time import sleep
import string
import sys
from nltk.tokenize import word_tokenize
import nltk
from nltk.corpus import stopwords

sys.setrecursionlimit(100000)  # 10000 is an example, try with different values

if __name__ == '__main__':
    
    scrapdate = [
                 #['1701', '2050'],
                 #['1702', 2200],
                 #['1703', 2700],
                 #['1704', 2295],
                 #['1705', 2805],
                 #['1706', 2482],
                 ['1707', 2644],
                 ['1708', 2496],
                 ['1709', 2750],
                 ['1710', 2669],
                 ['1711', 3056],
                 ['1712', 2416]
                 ]
    elementPerPage = 25
    
    def cleanText(text):
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
        return ';'.join(words)

    for month in range(len(scrapdate)):
        for page in range(0, scrapdate[month][1], elementPerPage):
            save_path = './' + str(scrapdate[month][0]) + "_" + str(page) + ".p"
            main_url = "https://arxiv.org/list/cs/" + \
                str(scrapdate[month][0]) + "?skip=" + \
                str(page) + "&show=" + str(elementPerPage)
            print(main_url)
            print(save_path)

            page_url = requests.get(main_url)
            soup = BeautifulSoup(page_url.text, 'html.parser')

            tds = soup.find_all('dt')
            dds = soup.find_all('dd')

            results = []

            # get infos
            for i in range(len(tds)):   
                td = tds[i]
                dd = dds[i]

                url = td.find('span').find('a')['href']
                id = url.replace('/abs/', '')
            
                main_class = dd.find('div', class_="meta")

                title = main_class.find('div', class_="list-title").contents[2]
                category = main_class.find('div', class_='list-subjects').find('span', class_='primary-subject').text

                results.append({'id': id, 'url': url, 'title':title, 'cat': category})
            
            count = 1

            # get abstract
            
            for result in results:
                sleep(6)
                main_url = 'https://arxiv.org' + result['url']
                print(str(count) + " : " + main_url)
                url = requests.get(main_url)
                soup = BeautifulSoup(url.text, 'html.parser')

                abstract = soup.find('blockquote', class_='abstract').contents[2].replace('\n', ' ').strip()

                result['abs'] = abstract
                count+=1
                print(result)

            # Store in Pickle
            pickle.dump(results, open(save_path, "wb"))

            #favorite_color = pickle.load(open(str(scrapdate[1]) + "_" + str(page) + ".p", "rb"))
            #print("-------------------------------------------------")
            #print(favorite_color)
            #print("-------------------------------------------------")
