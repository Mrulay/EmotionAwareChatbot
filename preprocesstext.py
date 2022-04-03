from nltk.corpus import stopwords
import contractions
import string

stop = stopwords.words('english')

def preProcessText(text):
    text = text.lower() # lowercasing everything
    text = contractions.fix(text) # fixing contractions (e.g. isn't -> is not)
    text = text.translate(str.maketrans('', '', string.punctuation)) # removing punctuations
    text = text.translate(str.maketrans('', '', string.punctuation)) # removing punctuations
    text = text.replace('@', '') 
    return text