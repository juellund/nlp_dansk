import PyPDF2
import os
import re
import nltk
import string

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
import numpy as np

from spacy.lang.da.stop_words import STOP_WORDS
from sentida import Sentida
from collections import Counter
from operator import itemgetter
from polyglot.text import Text
from sklearn.decomposition import LatentDirichletAllocation as LDA
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer
from collections import defaultdict

plt.rcParams["figure.figsize"] = (15,7.5)
nltk.download('stopwords')
nltk.download('punkt')

def read_pdfs(sti, tekster):
  files = os.listdir(sti)
  alle_tekster = []
  for file in files:
    if file in tekster:
      pdf = open(sti + '/' + file, 'rb')
      read_pdf = PyPDF2.PdfFileReader(pdf)
      tekst = []
      for page in range(read_pdf.getNumPages()):
        tekst.append(read_pdf.getPage(page).extractText())
      alle_tekster.extend(tekst)
  tekst_raw = ''
  for side in alle_tekster:
    side_raw = side.split('\n')
    tekst_raw += ' '.join(side_raw)
  return tekst_raw

def normalize(text, term):
  return re.sub(r'{}'.format(term), ' ', text)

def normalizer(text, regex_terms):
  text = re.sub(r'[\s]{2,}', ' ', text)
  for term in regex_terms + regex_terms:
    text = normalize(text, term)
  for letter in ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'x', 'y', 'z', 'æ', 'ø', 'å']:
    text = re.sub(r'\s{}\s'.format(letter), '{}'.format(letter), text)
  return text

def word_freq(tokens, stopord=True, save_fig=None):
  nltk_stopwords = nltk.corpus.stopwords.words('danish')
  spacy_stopwords = STOP_WORDS
  stopwords = set(list(nltk_stopwords) + list(spacy_stopwords) + ['Så', 'se', 'måtte', 'lod', 'lå', 'lagde', 'Hvis', 'går', 'En', 'gerne', 'fået', 'Når', 'helt', 'kommet', 'Om', 'får'] + ['saa', 'paa', 'sagde', 'kunde', 'Du', 'saae', 'gik', 'see', 'Det', 'skulde', 'naar', 'stod', 'ogsaa', 'igjen', 'Og', 'hele', 'Men', 'I', 'Jeg', 'sad', 'De', 'laae', 'komme', 'maatte', 'Den', 'fik', 'Dig', 'faae', 'maa', 'Der', 'holdt', 'Da', 'gaae', 'een', 'satte', 'O', 'bare', 'bleve', 'seet', 'saadan', 'veed', 'Saa', 'Hvad', 'seer', 'vist', 'Have', 'gjerne', 'saaledes', 'hvert', 'gjør', 'vel', 'heel', 'begyndte', 'staae', 'lade', 'Her', 'Haar', 'Hvor', 'Side', 'al'])
  tokens = [token for token in tokens if token.isalpha()]
  if stopord:
    tokens = [token for token in tokens if token not in stopwords]
  hyppighed = Counter(tokens)
  hyppighed = sorted(hyppighed.items(), key=itemgetter(1), reverse=True)
  plt.bar([x[0] for x in hyppighed[:100]], [x[1] for x in hyppighed[:100]])
  plt.xticks(rotation=90)
  if save_fig != None:
    plt.title('{}'.format(save_fig))
    plt.savefig('/content/gdrive/My Drive/studenterprojekt/folkeskolelærer/{}.png'.format(save_fig))
  plt.show()

def update_ticks(x, pos):
  return round(x*100,1)

def sentida_analysis(sents, percentage=True, plot=False, save_fig=None, version=1):
  SV = Sentida()
  sentiment = []
  sentiment_val = []
  positive = []
  pos_neu = []
  neutral = []
  neg_neu = []
  negative = []
  for sent in sents:
    if len(re.sub('[^A-Za-z0-9]+', '', sent)) == 0:
      None
    else:
      sent_val = SV.sentida(text = sent, output = 'mean', normal = False)
      sentiment_val.append(sent_val)
      if version = 1:
        if sent_val > 0.0:
            sentiment.append('positiv')
            positive.append(sent)
        elif sent_val < 0.0:
            sentiment.append('negativ')
            negative.append(sent)
        else:
            sentiment.append('neutral')
            neutral.append(sent)
      elif version == 2:
        if sent_val > 0.0 and sent_val <= 1.0:
          sentiment.append('positiv/neutral')
          pos_neu.append(sent)
        elif sent_val > 1.0:
          sentiment.append('positiv')
          positive.append(sent)
        elif sent_val < 0.0 and sent_val >= -1.0:
          sentiment.append('negativ/neutral')
          neg_neu.append(sent)
        elif sent_val < -1.0:
          sentiment.append('negativ')
          negative.append(sent)
        else:
          sentiment.append('neutral')
          neutral.append(sent)
  if plot:
    if percentage:
        hyppighed = Counter(sentiment)
        hyppighed = sorted(hyppighed.items(), key=itemgetter(1), reverse=True)
        norm_hyp = [(x[0], x[1]/len(sents)) for x in hyppighed]
        fig, ax = plt.subplots()
        ax.bar([x[0] for x in norm_hyp], [x[1] for x in norm_hyp])
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(update_ticks))
        for p in ax.patches:
            ax.annotate(str(round(p.get_height()*100,2)), (p.get_x() * 1.005, p.get_height() * 1.005))
    else:
        hyppighed = Counter(sentiment)
        hyppighed = sorted(hyppighed.items(), key=itemgetter(1), reverse=True)
        fig, ax = plt.subplots()
        ax.bar([x[0] for x in hyppighed], [x[1] for x in hyppighed])
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(update_ticks))
        for p in ax.patches:
            ax.annotate(str(round(p.get_height(),2)), (p.get_x() * 1.005, p.get_height() * 1.005))
    if save_fig != None:
        plt.title('{}'.format(save_fig))
        plt.savefig('/content/gdrive/My Drive/studenterprojekt/folkeskolelærer/{}.png'.format(save_fig))
    plt.show()
  else:
    if version == 1:
      return positive, neutral, negative #positive, pos_neu, neutral, neg_neu, negative
    elif version == 2:
      return positive, pos_neu, neutral, neg_neu, negative


def pos(sents, percentage=True, plot=False, save_fig=None):
  pos = []
  pos_words = defaultdict(list)
  pos_dic = {'ADJ': 'tillægsord', 'ADP': 'forholdsord', 'ADV': 'biord', 'AUX': 'hjælpeudsagnsord', 'CONJ': 'sideordningsbindeord', 'DET': 'bestemmerled', 'INTJ': 'udråbsord', 
             'NOUN': 'navneord', 'NUM': 'talord', 'PART': 'particle', 'PRON': 'stedord', 'PROPN': 'egenavn', 'PUNCT': 'tegnsætning', 'SCONJ': 'underordningsbindeord',
             'SYM': 'symbol', 'VERB': 'udsagnsord', 'X': 'andet'}
  for sent in sents:
    text = Text(sent, hint_language_code='da')
    pos_tags = text.pos_tags
    pos_tags = [(x[0], pos_dic[x[1]]) for x in pos_tags]
    for elm in pos_tags:
      pos_words[elm[1]].append(elm[0])
    pos.extend(pos_tags)
  if plot:
    if percentage:
      data = pd.Series([x[1] for x in pos]).value_counts(normalize=True).sort_values()[::-1].to_dict()
      fig, ax = plt.subplots()
      ax.bar(data.keys(), data.values())
      ax.yaxis.set_major_formatter(mticker.FuncFormatter(update_ticks))
      plt.xticks(rotation=90)
      for p in ax.patches:
        ax.annotate(str(round(p.get_height()*100,2)), (p.get_x() * 1.005, p.get_height() * 1.005))
    else:
      data = pd.Series([x[1] for x in pos]).value_counts(normalize=False).sort_values()[::-1].to_dict()
      fig, ax = plt.subplots()
      ax.bar(data.keys(), data.values())
      ax.yaxis.set_major_formatter(mticker.FuncFormatter(update_ticks))
      plt.xticks(rotation=90)
      for p in ax.patches:
        ax.annotate(str(round(p.get_height(),2)), (p.get_x() * 1.005, p.get_height() * 1.005))
    if save_fig != None:
        plt.title('{}'.format(save_fig))
        plt.savefig('/content/gdrive/My Drive/studenterprojekt/folkeskolelærer/{}.png'.format(save_fig))
    plt.show()
  else:
    return pos_words

def word_length_freq(tokens, save_fig=None):
  tokens = ([token for token in tokens if any(c.isalpha() for c in token)])
  token_lengths = [len(token) for token in tokens]
  fig = plt.figure()
  length_freq = nltk.FreqDist(token_lengths)
  length_freq.plot()
  if save_fig != None:
    fig.suptitle('{}'.format(save_fig))
    fig.savefig('/content/gdrive/My Drive/studenterprojekt/folkeskolelærer/{}.png'.format(save_fig))

def chi_square_method(candidate_sti, candidate_tekster, candidate_regex_terms, base_sti, base_tekster, base_regex_terms):
  candidate = read_pdfs(candidate_sti, candidate_tekster)
  candiate = normalizer(candidate, candidate_regex_terms)
  candidate = nltk.word_tokenize(candidate, language='danish')
  candidate = ([token for token in candidate if any(c.isalpha() for c in token)])
  candidate = ([token.lower() for token in candidate])
  base = read_pdfs(base_sti, base_tekster)
  base = normalizer(base, base_regex_terms)
  base = nltk.word_tokenize(base, language='danish')
  base = ([token for token in base if any(c.isalpha() for c in token)])
  base = ([token.lower() for token in base])
  joint_corpus = (candidate + base)
  joint_freq_dist = nltk.FreqDist(joint_corpus)
  most_common = list(joint_freq_dist.most_common(500))
  author_share = (len(candidate) / len(joint_corpus))
  chisquared = 0
  for word,joint_count in most_common:
      candidate_count = candidate.count(word)
      base_count = base.count(word)
      expected_candidate_count = joint_count * author_share
      expected_base_count = joint_count * (1-author_share)
      chisquared += ((candidate_count-expected_candidate_count) *
                      (candidate_count-expected_candidate_count) /
                      expected_candidate_count)
      chisquared += ((base_count-expected_base_count) *
                      (base_count-expected_base_count)
                      / expected_base_count)
  print("Chi-i-anden for {} er".format(candidate_tekster), chisquared)

def word_clean(x):
  nltk_stopwords = nltk.corpus.stopwords.words('danish')
  spacy_stopwords = STOP_WORDS
  stopwords = set(list(nltk_stopwords) + list(spacy_stopwords) + ['Så', 'se', 'måtte', 'lod', 'lå', 'lagde', 'Hvis', 'går', 'En', 'gerne', 'fået', 'Når', 'helt', 'kommet', 'Om', 'får'] + ['saa', 'paa', 'sagde', 'kunde', 'Du', 'saae', 'gik', 'see', 'Det', 'skulde', 'naar', 'stod', 'ogsaa', 'igjen', 'Og', 'hele', 'Men', 'I', 'Jeg', 'sad', 'De', 'laae', 'komme', 'maatte', 'Den', 'fik', 'Dig', 'faae', 'maa', 'Der', 'holdt', 'Da', 'gaae', 'een', 'satte', 'O', 'bare', 'bleve', 'seet', 'saadan', 'veed', 'Saa', 'Hvad', 'seer', 'vist', 'Have', 'gjerne', 'saaledes', 'hvert', 'gjør', 'vel', 'heel', 'begyndte', 'staae', 'lade', 'Her', 'Haar', 'Hvor', 'Side', 'al'])
  nopunc = ''.join([char.lower() for char in x if char not in string.punctuation])
  clean_mess = ' '.join([word for word in nopunc.split() if word.lower() not in stopwords])
  return clean_mess

def print_topics(model, count_vectorizer, n_top_words):
  words = count_vectorizer.get_feature_names()
  for topic_idx, topic in enumerate(model.components_):
      print("\nTopic {}:".format(topic_idx))
      print(" ".join([words[i]
                      for i in topic.argsort()[:-n_top_words - 1:-1]]))

def topic_analysis(tekster_sti, tekster, regex_terms, num_topics=2, num_words=50, seed=0):
  nltk_stopwords = nltk.corpus.stopwords.words('danish')
  spacy_stopwords = STOP_WORDS
  stopwords = set(list(nltk_stopwords) + list(spacy_stopwords) + ['Så', 'se', 'måtte', 'lod', 'lå', 'lagde', 'Hvis', 'går', 'En', 'gerne', 'fået', 'Når', 'helt', 'kommet', 'Om', 'får'] + ['saa', 'paa', 'sagde', 'kunde', 'Du', 'saae', 'gik', 'see', 'Det', 'skulde', 'naar', 'stod', 'ogsaa', 'igjen', 'Og', 'hele', 'Men', 'I', 'Jeg', 'sad', 'De', 'laae', 'komme', 'maatte', 'Den', 'fik', 'Dig', 'faae', 'maa', 'Der', 'holdt', 'Da', 'gaae', 'een', 'satte', 'O', 'bare', 'bleve', 'seet', 'saadan', 'veed', 'Saa', 'Hvad', 'seer', 'vist', 'Have', 'gjerne', 'saaledes', 'hvert', 'gjør', 'vel', 'heel', 'begyndte', 'staae', 'lade', 'Her', 'Haar', 'Hvor', 'Side', 'al'])
  
  tekster_all = []

  for tekst in tekster:
    curr_tekst = read_pdfs(tekster_sti, tekst)
    curr_tekst = normalizer(curr_tekst, regex_terms)
    tekster_all.append(curr_tekst)

  data = pd.Series(tekster_all).apply(word_clean)

  count_vectorizer = CountVectorizer(stop_words=stopwords, ngram_range=(1, 4), min_df = 4, max_df = 0.8)
  count_data = count_vectorizer.fit_transform(data)

  search_params = {'n_components': [2]} #, 3, 4, 5, 6,7,8,9,10, 15, 20, 25
  lda = LDA(random_state=seed)
  model = GridSearchCV(lda, search_params)
  model.fit(count_data)
  best_lda_model = model.best_estimator_

  #print("Best model's params: ", model.best_params_)
  #print("Best log likelihood score: ", model.best_score_)
  #print("Model perplexity: ", best_lda_model.perplexity(count_data))
          
  number_topics = num_topics
  number_words = num_words
  print_topics(best_lda_model, count_vectorizer, number_words)