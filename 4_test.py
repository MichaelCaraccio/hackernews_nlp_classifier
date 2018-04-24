import string

from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.externals import joblib

wordnet = WordNetLemmatizer()


def clean_text(text):
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


def get_class_name_from_proba(proba_array, enc):
    """Get class name from labelencoder

    Arguments:
        proba_array {array} -- list of propability
        enc                 -- array with labelencoder's values

    Returns:
        [string] -- class name
    """
    idx = proba_array.flatten().argmax(axis=0)
    return enc[idx]


if __name__ == '__main__':
    models_path = './files/models/'

    classList = joblib.load(models_path + 'classlist_cat_main_20180423-234137.pkl')
    clf = joblib.load(models_path + 'estimator_cat_main_20180423-234137.pkl')

    t = [clean_text(
        "At some point you just need to stop looking and be blissfully ignorant...this was not one of those days. In "
        "and update to my previously updated blog article, I have found another instance where the plaintext password "
        "was written to system logs. This time I found it in more persistent log. This is actually a worse problem "
        "than the one I previously reported on. The previous examples were found in the unified logs which can hang "
        "around for a few weeks, this new example stores the exact same information in the system's / var / log / "
        "install.log. I have found that the install.log will only get wiped out upon major re - installation(ie: "
        "10.11 -> 10.12 -> 10.13), therefore these plaintext passwords will hang around for quite a bit longer than a "
        "few weeks! I had entries dating back to when I originally installed High Sierra on this system back in "
        "November of 2017! Twitter user @sirkkalap, was unable to re - create what I previously reported on. I "
        "finally got some time this afternoon to re - test. As it turns out, I was unable to re - create my results "
        "from 03 / 24. I assumed that at some point in the past few days a silent security update was pushed out. I "
        "went to my install.log file to investigate further. As far as updates go - the only thing that has potential "
        "to be the cause of the fix is a GateKeeper ConfigData update v138("
        "com.apple.pkg.GatekeeperConfigData.16U1432). I have not investigated if this was the true cause. I have not "
        "updated to 10.13.4 yet, this was on 10.13.3. During this investigations I was VERY surprised to see the same "
        "diskmanagementd logs that I had found in the unified logs. Why are they logged in the software installation "
        "log at all, I have no clue. It makes absolutely no sense to me.")]

    pred = clf.predict_proba(t)
    print(pred)
    className = get_class_name_from_proba(pred, classList)
    print(className)

    t = [clean_text('International Union for the Conservation of Nature, the body that administers the world’s '
                    'official endangered species list, announced yesterday that it was moving the giraffe from a '
                    'species of Least Concern to Vulnerable status in its Red List of Threatened Species report. That '
                    'means the animal faces extinction in the wild in the medium - term future if nothing is done to '
                    'minimize the threats to its life or habitat. The next steps are endangered, critically '
                    'endangered, extinct in the wild and extinct. RELATED CONTENT \ How America Can Help Save a Non - '
                    'American Species: The Mighty Giraffe\ Poaching of elephants and rhinoceros and the illegal trade '
                    'in pangolins has overshadowed the problems with giraffes in the last decade. But Damian '
                    'Carrington at The Guardian reports that giraffe numbers have dropped precipitously in the last '
                    '31 years, from 157, 000 individuals in 1985 to 97, 500 at last count.\ “Whilst giraffes are '
                    'commonly seen on safari, in the media and in zoos, people—including conservationists—are unaware '
                    'that these majestic animals are undergoing a silent extinction, ” Julian Fennessy, '
                    'the co - chair of the IUCN’s Species Survival Commission’s Giraffe and Okapi Specialist Group '
                    'says in a press release. “With a decline of almost 40 percent in the last three decades alone, '
                    'the world’s tallest animal is under severe pressure in some of its core ranges across East, '
                    'Central and West Africa. As one of the world’s most iconic animals, it is timely that we stick '
                    'our neck out for the giraffe before it is too late.”\ The giraffes face two main threats, '
                    'encroachment from cities and towns into their habitat and poaching. Poaching has become '
                    'increasingly problematic. Some food insecure villagers kill the animals for their meat, '
                    'but Jani Actman at National Geographic reports many giraffes are slaughtered just for their '
                    'tails, which are considered a status symbol and have been used as a dowry when asking a bride’s '
                    'father for his daughters hand in marriage in some cultures.\ Patrick Healy at The New York Times '
                    'reports that red list divides the giraffe into nine subspecies. Five of those subspecies are '
                    'decreasing in numbers while two populations are increasing and one is stable. West African '
                    'giraffes, the smallest subspecies, have grown from 50 individuals in the 1990s to about 400 '
                    'today. But that success took a massive amount of effort from the government of Niger and '
                    'conservation groups.\ It will take similar efforts throughout the giraffe’s wide range to arrest '
                    'its plummeting numbers. Derek Lee, founder of the Wild Nature Institute who contributed to the '
                    'IUCN update tells Healy that both poaching and habitat encroachment need to be stopped to save '
                    'the giraffe. “These are problems everywhere for giraffes, ” he says. “You need to stop both '
                    'threats.”\ While increasing funding for anti - poaching efforts can do some good, Lee thinks '
                    'stopping habitat encroachment is a much more difficult prospect, since it would mean interfering '
                    'with land development, mining and other economic activities and livelihoods.\ The biggest '
                    'problem for giraffes, though, may be the lack of attention over the years. “I am absolutely '
                    'amazed that no one has a clue, ” Julian Fennessy, executive director of Giraffe Conservation '
                    'Foundation tells Sarah Knapton at The Telegraph. “This silent extinction. Some populations less '
                    'than 400. That is more endangered than any gorilla, or almost any large mammal in the world.”\ '
                    '“There’s a strong tendency to think that familiar species(such as giraffes, chimps, etc.) must '
                    'be OK because they are familiar and we see them in zoos, ” Duke University conservation '
                    'biologist Stuart Pimm, tells the Associated Press. In fact, giraffes have silently been going '
                    'extinct across Africa over the last century. The animal is already gone from seven countries, '
                    'Burkina Faso, Eritrea, Guinea, Malawi, Mauritania, Nigeria and Senegal.\ Read more: https: // '
                    'www.smithsonianmag.com / smart - news / giraffes - silently - slip - endangered - species - list '
                    '- 180961372 0GE7SgHgQRRboBTT.99 Give the gift of Smithsonian magazine for only $12! http: // '
                    'bit.ly / 1cGUiGv Follow us: @SmithsonianMag on Twitter')]

    pred = clf.predict_proba(t)
    print(pred)
    className = get_class_name_from_proba(pred, classList)
    print(className)
