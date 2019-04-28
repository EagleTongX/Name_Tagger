import sys
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import treebank
import numpy as np
from sklearn.cluster import KMeans
from gensim.models import Word2Vec

lemmatizer = WordNetLemmatizer()
stopWords = set(stopwords.words('english'))

have_tag = False

token = []
pos = []
chunk = []
tag = []
embeds = {}



#get embeded word from Glove, set dimension 
dimension = 50
with open("glove.6B.50d.txt")as f:
	for line in f.readlines():
		line = line[:-1]
		line = line.split(" ")    
		embeds[line[0]] = [float(v) for v in line[1:]]
'''
a = [v for v in embeds.values()]
#a = np.array(a)
cluster = True
k = 1024
kmeans = KMeans(n_clusters = k).fit(a)
'''

#define cluster
#nltk.download('treebank')
wordvec = Word2Vec(treebank.sents(), size = 50).wv
k = 500
kmeans = KMeans(n_clusters = k, random_state = 0).fit(wordvec[wordvec.vocab])


with open(sys.argv[1]) as f:
	if sys.argv[1].endswith('name'):
		have_tag = True

	cur_token, cur_pos, cur_chunk, cur_tag = [], [], [], []
	while True:
		line = f.readline()
		if line == None or line == '':
			break
		if line == '\n':
			token.append(cur_token)
			pos.append(cur_pos)
			chunk.append(cur_chunk)

			if have_tag:
				tag.append(cur_tag)

			cur_token, cur_pos, cur_chunk, cur_tag = [], [], [], []

		else:
			l = line[:-1].split('\t')
			cur_token.append(l[0])
			cur_pos.append(l[1])
			cur_chunk.append(l[2])

			if have_tag:
				cur_tag.append(l[3])

# create dic of word embedding
vec = {}
binar = []
for tok in token:
	for t in tok:
		if t.lower() in embeds:
			vec[t] = embeds[t.lower()]
		else:
			vec[t] = [0] * dimension

		binar.append(vec[t])

#binarization get mean(C_i+) and mean(C_i-)
b = np.array(binar)
mean_pos = b[b>0].mean()
mean_neg = b[b<0].mean()
#mean_pos = np.nanmean(np.where(b >0, b, np.nan), axis = 0)
#mean_neg = np.nanmean(np.where(b <0, b, np.nan), axis = 0)
pre_token = []
pre_pos = []
pre_chunk = []
pre_tag = []
nex_token = []
nex_pos = []
nex_chunk = []
nex_tag = []


for sent_tok, sent_pos, sent_chu in zip(token, pos, chunk):
	pre_token.append(["<BOS>",] + sent_tok[:-1])
	pre_pos.append(["<BOS>",] + sent_pos[:-1])
	pre_chunk.append(["<BOS>",] + sent_chu[:-1])

	nex_token.append(sent_tok[1:] + ["<EOS>",])
	nex_pos.append(sent_pos[1:] + ["<EOS>",])
	nex_chunk.append(sent_chu[1:] + ["<EOS>",])

if have_tag:
	for sent_tag in tag:
		pre_tag.append(["<BOS>",] + sent_tag[:-1])
		nex_tag.append(sent_tag[1:] + ["<EOS>",])
else:
	for sent in token:
		pre_tag.append(["@@", ] * len(sent))
		nex_tag.append(["@@", ] * len(sent))


feature_name = ['cur_pos', 'cur_chunk', 'pre_token', 'nex_token', 'pre_pos', 'next_pos', 'pre_chunk', 'pre_tag', 'nex_tag']
feature_list = [pos, chunk, pre_token, pre_pos, nex_token, pre_pos, nex_pos, pre_chunk, pre_tag, nex_tag]



with open(sys.argv[1]+'-feature-enhanced', 'w') as f:
	for sent_idx in range(len(token)):
		for word_idx in range(len(token[sent_idx])):

			# features from corpus.
			features = [f[sent_idx][word_idx] for f in feature_list]
			feats = '\t'.join([n+"="+v for n, v in zip(feature_name, features)])
			
			# extra features I used.
			feats += '\tisalpha=%s' % token[sent_idx][word_idx].isalpha()
			feats += '\tislower=%s' % token[sent_idx][word_idx].islower()
			feats += '\tistitle=%s' % token[sent_idx][word_idx].istitle()
			feats += '\tlemm=%s' % lemmatizer.lemmatize(token[sent_idx][word_idx])
			feats += '\tlower=%s' % token[sent_idx][word_idx].lower()
			feats += '\tisdigit=%s' % token[sent_idx][word_idx].isdigit()
			feats += "\tprefix=%s" % token[sent_idx][word_idx][:3]
			feats += "\tsuffix=%s" % token[sent_idx][word_idx][-3:]


			t = token[sent_idx][word_idx]
			#word embedding
			'''
			if t.lower() in embeds:
				for idx in range(dimension):
					feats += "\twordVector%s=%s" % (idx, vec[t][idx])
			else:
				for idx in range(dimension):
					feats += "\twordVector%s=%s" % (idx, float('0'))
		


			'''
			#word embedding (binarization)
			if t.lower() in embeds:
			

				for idx in range(dimension):
					if vec[t][idx] > mean_pos:
						feats += '\tbinar%s=%s' % (idx, 'POSITIVE')
					elif vec[t][idx] < mean_neg:
						feats += '\tbinar%s=%s' % (idx, 'NEG')
					else:
						feats += '\tbinar%s=%s' % (idx, 'ZERO')
			else:
				for idx in range(dimension):
					feats += '\tbinar%s=%s' % (idx, 'OOV')
	
			'''
			#add k-means clustering
			if t.lower() in wordvec.vocab:
				label = kmeans.predict([wordvec[t.lower()]])[0]
				feats += '\tcluster_label%s' % str(label)
			else:
				feats += '\tcluster_label%s' % '00V'
			'''


			# add token at the beginning of the line.
			line = token[sent_idx][word_idx]+'\t'+feats

			if have_tag:
				line += '\t'+tag[sent_idx][word_idx]
			
			f.write(line+'\n')
		f.write('\n')














