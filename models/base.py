import string
import html
import nltk
from nltk.corpus import stopwords
from nltk import word_tokenize
import pickle
import string

class BaseModel:
    def __init__(self):
        self.lemmatizer = nltk.WordNetLemmatizer()
        self.stop = stopwords.words('english')
        self.stop += ['not_' + w for w in self.stop]
        self.model = None
        self.vec = None
        self.punctuation = string.punctuation.replace("'","")
        self.translation = str.maketrans(self.punctuation,' '*len(self.punctuation))

    # Load Vec
    def load_vec(self, vec_path, mode='rb'):
        with open(vec_path, mode) as pkl_file:
            self.vec = pickle.load(pkl_file)

    # Load Model
    def load_model(self, model_path, mode='rb'):
        with open(model_path, mode) as pkl_file:
            self.model = pickle.load(pkl_file)

    # Preprocessing
    def preprocessing(self, line: str) -> str:
        line = html.unescape(str(line))
        line = str(line).replace("can't", "cann't")
        line = str(line).translate(self.translation)
        line = word_tokenize(line.lower())

        tokens = []
        negated = False
        for t in line:
            if t in ['not', "n't", 'no']:
                negated = not negated
            if t in string.punctuation or not t.isalpha():
                negated = False
            if negated == True:
                tokens.append('not_' + t)
            else:
                tokens.append(t)

        tokens = [self.lemmatizer.lemmatize(t, 'v') for t in tokens if t not in self.stop]

        return ' '.join(tokens)

    # Predict
    def predict(self, line):
        if self.model is None or self.vec is None:
            print('Modle / Vec is not loaded')
            return ""

        line = self.preprocessing(line)
        features = self.vec.transform([line])

        return self.model.predict(features)[0]
