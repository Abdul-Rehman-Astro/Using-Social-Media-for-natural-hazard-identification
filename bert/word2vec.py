import gensim
import gensim.models as g
import gensim.downloader
from spacy.tokenizer import Tokenizer
from spacy.lang.en import English

def vectorize_sentence(sentence,model):
    nlp = English()
    tokenizer = Tokenizer(nlp.vocab)
    a = []
    for i in tokenizer(sentence):
        try:
            a.append(model.get_vector(str(i)))
        except:
            pass
        
    a=np.array(a).mean(axis=0)
    a = np.zeros(300) if np.all(a!=a) else a
    return a
word2vec = gensim.downloader.load('word2vec-google-news-300') #1.66 gb
# vectorize the data
X_train_vec = pd.DataFrame(np.vstack(X_train['text'].apply(vectorize_sentence, model=word2vec)))
X_test_vec = pd.DataFrame(np.vstack(X_test['text'].apply(vectorize_sentence, model=word2vec)))
