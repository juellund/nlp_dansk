import pypdf
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
from collections import defaultdict

plt.rcParams["figure.figsize"] = (15,7.5)
nltk.download('stopwords')
nltk.download('punkt')

def read_pdfs(sti, tekster, newline=False):
  """Read a PDF-file and return it as a string.

  Keywords arguments:
  sti -- the location of the PDF-file
  tekster -- a list of PDF-names
  """
  files = os.listdir(sti)
  alle_tekster = []
  for file in files:
    if file in tekster:
      pdf = open(sti + '/' + file, 'rb')
      read_pdf = pypdf.PdfReader(pdf)
      tekst = []
      for page in range(len(read_pdf.pages)):
        tekst.append(read_pdf.getPage(page).extractText())
      alle_tekster.extend(tekst)
  tekst_raw = ''
  for side in alle_tekster:
    if newline:
      tekst_raw += ''.join(side)
    else:
      side_raw = side.split('\n')
      tekst_raw += ' '.join(side_raw)
  return tekst_raw

def normalize(text, term):
  """Takes a string and normalizes based on regex term.

  Keywords arguments:
  text -- the text as a string
  term -- a single regex
  """
  return re.sub(r'{}'.format(term), ' ', text)

def normalizer(text, regex_terms):
  """Take a string and remove double spaces and stand alone letters (except i).

  Keywords arguments:
  text -- the text as a string
  regex_terms -- a list of regex
  """
  text = re.sub(r'[\s]{2,}', ' ', text)
  for term in regex_terms + regex_terms:
    text = normalize(text, term)
  for letter in ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'x', 'y', 'z', 'æ', 'ø', 'å']:
    text = re.sub(r'\s{}\s'.format(letter), '{}'.format(letter), text)
  return text

def word_freq(tokens, stopord=True, fig_title=None):
  """Takes a list of tokens and compute/visualize word frequency.

  Keywords arguments:
  tokens -- a list of tokens
  stopword -- remove stopwords if true
  fig_title -- add title to figure if not None
  """
  nltk_stopwords = nltk.corpus.stopwords.words('danish')
  spacy_stopwords = STOP_WORDS
  stopwords = set(list(nltk_stopwords) + list(spacy_stopwords) + ['Q', 'QQ', 'QQQ', 'A', 'Så', 'se', 'måtte', 'lod', 'lå', 'lagde', 'Hvis', 'går', 'En', 'gerne', 'fået', 'Når', 'helt', 'kommet', 'Om', 'får'] + ['saa', 'paa', 'sagde', 'kunde', 'Du', 'saae', 'gik', 'see', 'Det', 'skulde', 'naar', 'stod', 'ogsaa', 'igjen', 'Og', 'hele', 'Men', 'I', 'Jeg', 'sad', 'De', 'laae', 'komme', 'maatte', 'Den', 'fik', 'Dig', 'faae', 'maa', 'Der', 'holdt', 'Da', 'gaae', 'een', 'satte', 'O', 'bare', 'bleve', 'seet', 'saadan', 'veed', 'Saa', 'Hvad', 'seer', 'vist', 'Have', 'gjerne', 'saaledes', 'hvert', 'gjør', 'vel', 'heel', 'begyndte', 'staae', 'lade', 'Her', 'Haar', 'Hvor', 'Side', 'al'])
  tokens = [token for token in tokens if token.isalpha()]
  if stopord:
    tokens = [token for token in tokens if token not in stopwords]
  hyppighed = Counter(tokens)
  hyppighed = sorted(hyppighed.items(), key=itemgetter(1), reverse=True)
  plt.bar([x[0] for x in hyppighed[:100]], [x[1] for x in hyppighed[:100]])
  plt.xticks(rotation=90)
  if fig_title != None:
    plt.title('{}'.format(fig_title))
    #plt.savefig('/content/gdrive/My Drive/studenterprojekt/folkeskolelærer/{}.png'.format(save_fig))
  plt.show()

def update_ticks(x, pos):
  return round(x*100,1)

def sentida_analysis(sents, percentage=True, plot=False, fig_title=None, version=1):
  """Takes a list of sentences and compute a sentiment value.

  Keywords arguments:
  sents -- a list of sentences
  percentage -- return the result as percentage if true
  plot -- return a visualization if true (instead of lists of pos/neu/neg sentences)
  fig_title -- add title to figure if not None
  version -- a parameter deciding how to categorize the sentiment value into a sentiment group
  """
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
      if version == 1:
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
    if fig_title != None:
        plt.title('{}'.format(fig_title))
        #plt.savefig('/content/gdrive/My Drive/studenterprojekt/folkeskolelærer/{}.png'.format(save_fig))
    plt.show()
  else:
    if version == 1:
      return positive, neutral, negative
    elif version == 2:
      return positive, pos_neu, neutral, neg_neu, negative

def pos(sents, percentage=True, plot=False, fig_titel=None):
  """Takes a list of sentences and compute part-of-speech tagging.

  Keywords arguments:
  sents -- a list of sentences
  percentage -- return the result as percentage if true
  plot -- return a visualization if true (instead of a dict of part-of-speech tags and corresponding tokens)
  fig_title -- add title to figure if not None
  """
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
    if fig_titel != None:
        plt.title('{}'.format(fig_titel))
        #plt.savefig('/content/gdrive/My Drive/studenterprojekt/folkeskolelærer/{}.png'.format(save_fig))
    plt.show()
  else:
    return pos_words

def word_length_freq(tokens, fig_titel=None):
  """Takes a list of tokens and reurn word length frequency for that list.

  Keywords arguments:
  tokens -- a list of tokens
  fig_titel -- add title to figure if not None
  """
  tokens = ([token for token in tokens if any(c.isalpha() for c in token)])
  token_lengths = [len(token) for token in tokens]
  fig = plt.figure()
  length_freq = nltk.FreqDist(token_lengths)
  length_freq.plot()
  if fig_titel != None:
    fig.suptitle('{}'.format(fig_titel))
    #fig.savefig('/content/gdrive/My Drive/studenterprojekt/folkeskolelærer/{}.png'.format(save_fig))

def chi_square_method(candidate_sti, candidate_tekster, candidate_regex_terms, base_sti, base_tekster, base_regex_terms):
  """Takes candidate texts plus base texts and use the chi-squared statistic to find a similarity score between them.

  Keywords arguments:
  candidate_sti -- the location of the PDF-file for the candidate texts
  candidate_tekster -- a list of PDF-names for the candidate texts
  candidate_regex_terms -- a list of regex for the candidate texts
  base_sti -- the location of the PDF-file for the base texts
  base_tekster -- a list of PDF-names for the base texts
  base_regex_terms -- a list of regex for the base texts
  """
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
