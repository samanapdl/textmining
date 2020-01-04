import glob
import string
import os
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from yellowbrick.text import FreqDistVisualizer
from wordcloud import WordCloud
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.cluster.hierarchy import ward, dendrogram



files = glob.glob(os.path.join(os.getcwd(),"texts", "*.txt"))
data = []
for text in files:
    with open(text) as f:
        txt = f.read()
    data.append(txt)

def data_processing(text, stem = True) :
    table = str.maketrans(dict.fromkeys(string.punctuation))
    text = text.translate(table)
    tokens = word_tokenize(text)
    
    tokens= [word for word in tokens if word.isalpha()]
    stopwords = nltk.corpus.stopwords.words('english')
    tokens = [w for w in tokens if not w in stopwords]
    if stem:
        stemmer = SnowballStemmer("english")
        tokens = [stemmer.stem(word) for word in tokens]
    return tokens

#define vectorizer parameters
tfidf_vectorizer = TfidfVectorizer( tokenizer= data_processing,
                                    max_df=0.5,
                                    min_df=0.1, stop_words='english',
                                    use_idf=True)
vectorizer = CountVectorizer(tokenizer=data_processing,
                            max_df = 0.5,
                            min_df = 0.1, 
                            stop_words = 'english')
tfidf_matrix = tfidf_vectorizer.fit_transform(data)

tf_model = vectorizer.fit_transform(data)
total_tf = [sum(x) for x in zip(*tf_model.toarray())]
terms = tfidf_vectorizer.get_feature_names()
freq = []
term_list = []
popTerm = []
pt = []
cloud = {}
for i,v in enumerate(total_tf):
    if v > 35:
        freq.append(v)
        popTerm.append(i)
        term_list.append(terms[i])
        cloud[terms[i]] = v
        pt.append(tf_model.transpose().toarray()[i])
print(freq)
print(term_list)

#word cloud
visualizer = FreqDistVisualizer(features=terms, orient='v')
visualizer.fit(tf_model)
visualizer.show()
wordcloud = WordCloud(normalize_plurals= False).generate_from_frequencies(cloud)    

plt.imshow(wordcloud)
plt.axis("off")
plt.savefig('word_cloud.png', dpi=200) 
plt.show()


km = KMeans(n_clusters=2)
km = km.fit(tf_model.transpose())
print(tf_model.shape)
clusters = km.labels_.tolist()

color = [ '#d95f02' if x == 0 else '#7570b3' for x in clusters]
figure, ax = plt.subplots(figsize=(20,15))
for i in popTerm:
    ax.scatter(tf_model.toarray()[2][i], tf_model.toarray()[3][i], c =color[i])
    ax.annotate(terms[i], (tf_model.toarray()[2][i], tf_model.toarray()[3][i]))
    print(terms[i])
plt.savefig('kmeans.png')
plt.show()

dist = euclidean_distances(pt)
link_met = ward(dist)
figure, ax = plt.subplots(figsize=(20,15))
ax = dendrogram(link_met, orientation='top', labels= term_list)
plt.tick_params(\
    axis= 'x',
    which='both',
    bottom='off',
    top='off',
    labelbottom='off')
plt.tight_layout()
plt.savefig('word_clusters.png', dpi=200)
plt.show()

