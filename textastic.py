"""
filename: textastic.py
description an extensible reusable library for text analysis and comparison
"""

from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import pandas as pd
from textblob import TextBlob
import re
import sankey as sk
import numpy as np
from check_txt import TextChecker
import seaborn as sns

# load in stop words
STOP_WORDS = 'stop_words.txt'


def sentiments(sentences):
    """ Calculates sentences' polarity and sentiments
    Args:
        sentences (list): sentences to calculate polarity and sentiments
    Returns:
        tups (list): list of tuples of each sentence's polarity and sentiment
    """
    tups = []
    for i in range(len(sentences)):
        tups.append((TextBlob(sentences[i]).sentiment.polarity, TextBlob(sentences[i]).sentiment.subjectivity))

    return tups


def filter_word_from_dct(dct, words):
    """ Removes desired words from a dictionary of words
    Args:
        dct (dictionary): contains words and their relevant statistics
        words (list): list of words to take out of dictionary values
    Returns:
        filtered_dict (dict): modified dictionary with removed words
    """
    filtered_dct = {}
    words = [word.strip() for word in words]
    for key, value in dct.items():
        if key not in words:
            filtered_dct[key] = value

    return filtered_dct


def read_txt(file, words=True):
    """ Reads in a text file and cleans it
    Args:
        file (str): name of text file
        words (boolean, default value = True): determines if sentences should be split
    Returns:
        txt (list): list of sentences or words from text file
    """
    txt = []

    # open text file
    with open(file, 'r', encoding='UTF8') as infile:
        for line in infile:
            line = line.strip()  # splits line into words
            line = re.sub('[^\w\s]', '', line)  # remove punctuation
            line = line.lower()  # convert to lowercase

            if not words:
                # appends the sentence
                txt.append(line)
            else:
                # appends each word
                txt.extend([word for word in line.split(" ") if word != ''])

    return txt


def n_grams(sentences):
    """ Reads in a dictionary and iterates each values looking for all valid tri-grams
    Args:
        sentences (list of strings): list of sentences
    Returns:
        n_gram (list): list of n_grams
    """
    n_gram = []
    for line in sentences:
        line = TextBlob(line)
        n_gram.append(line.ngrams(2))
    return n_gram


def count_ngrams(lst):
    """ Counts number of occurrences of given n_grams
    Args:
        lst (list): list n_grams in question
    Returns:
        counter (dict): contains how many times each n_gram occurred
    """
    counter = {}
    for row in lst:
        for ngram in row:
            ngram = tuple(ngram)
            if ngram in counter:
                counter[ngram] += 1
            else:
                counter[ngram] = 1
    return counter


class Textastic:

    def __init__(self):
        self.data = defaultdict(dict)  # keys are strings maps to dictionary value
        # second dictionary is a filename or label and value is statistics

    @staticmethod
    def load_stop_words(stopfile=STOP_WORDS):
        """Returns a list of all stop words"""
        return read_txt(stopfile)

    @staticmethod
    @TextChecker.check_txt
    def _default_parser(filename):
        """ Finds sentences, words, ngrams, and sentiment values for a txt file
        Args:
            filename (str): filename, should end in .txt or an error will be raised
        Returns:
            results (dict): resulting word count, number of words, number of n_grams, polarity, and subjectivity of file
        """
        # get a list of sentences and words then find wordcount
        sentences = read_txt(filename, words=False)
        words = read_txt(filename, words=True)
        word_dct = Counter(words)

        results = {
            'wordcount': word_dct,
            'numwords': len(words),
            'ngrams': count_ngrams(n_grams(sentences)),
            'sent_sent': sentiments(sentences)  # returns list of tuples (polarity,subjectivity)
        }
        return results

    def _save_results(self, label, results):
        """ Saves the results into the main class data dictionary
        Args:
            self (Textastic): class object
            label (str): desired label
            results (dict): contains items only of given dictionary
        """
        for k, v in results.items():
            self.data[k][label] = v

    def load_text(self, filename, label=None, parser=None):
        """ Registers a text document with the framework, extracts and stores data to be used in later visualizations
        Args:
            self (Textastic): class objecgt
            filename (str): name of file
            label (str, default = None): desired label
            parser (str, default = None): determines type of parser used to load the texts
        Returns:
            None
        """
        if parser is None:
            try:
                # parse the results
                results = Textastic._default_parser(filename)
            except Exception as e:
                # this Exception will occur if a txt file is not inputted
                print(str(e))
                results = None
        else:
            results = parser(filename)

        if label is None:
            label = filename

        # store the results of processing one file in the internal state
        self._save_results(label, results)

    def sankey_ngrams(self, k=10):
        """ Creates a sankey diagram to show wordflow of the documents
        Args:
            k (int): number of ngrams to use (it will choose the most frequent)
        Returns:
            None
        """
        ngram_count = {}

        for value in self.data['ngrams'].values():
            # finds the k most common ngrams of each file
            temp = Counter(value).most_common(k)

            # finds how often those words occur throughout all files
            for item in temp:
                if item[0] in ngram_count:
                    ngram_count[item[0]] += item[1]
                else:
                    ngram_count[item[0]] = item[1]

        # formats the ngrams into a dictionary (assumes ngrams are bigrams although code can be later altered to
        # allow more user customization and other length n grams
        dct = {'word1': [k[0] for k in ngram_count.keys()], 'word2': [k[1] for k in ngram_count.keys()],
               'value': ngram_count.values()}

        # create a sankey diagram
        sk.make_sankey(pd.DataFrame(dct), 'word1', 'word2', values='value', threshold=0)

    def wordcount_sankey(self, wordlist=None, k=5, wordcount=True):
        """ Creates a sankey diagram showing most common words in each file
        Args:
            wordlist (list, default = None): optional list of words to include in sankey
            k (int): top k words from each file
            wordcount (bool): if true looks at word count if false looks at ngrams
        Returns:
             None
        """
        df_dct = {'name': [], 'word': [], 'value': []}
        dct = self.data
        processed_words = set()

        if wordcount:
            dct_key = 'wordcount'
        else:
            dct_key = 'ngrams'

        # if the user has not selected their own custom word list
        if wordlist is None:

            # find all the words that are not stop words and add the first k amount of them in the file to the list
            for key, value in dct[dct_key].items():

                # filters out bigrams of only stopwords if looking at ngrams
                if not wordcount:
                    temp_dct = {(w[0], w[1]): k for w, k in dct['ngrams'][key].items()
                                          if w[0] not in self.load_stop_words(STOP_WORDS) and
                                          w[1] not in self.load_stop_words(STOP_WORDS)}

                else:
                    temp_dct = filter_word_from_dct(dct[dct_key][key], self.load_stop_words(STOP_WORDS))
                df_dct['name'].extend([key] * k)
                df_dct['word'].extend(w[0] for w in list(Counter(temp_dct).most_common(k)))
                df_dct['value'].extend(v[1] for v in list(Counter(temp_dct).most_common(k)))

        else:

            # if the user does have their own custom word list create a tuple of key, word, value for each word in the
            # list
            for key, value in dct[dct_key].items():
                key_word_value = [(key, list(dct[dct_key][key].keys())[i], list(dct[dct_key][key].values())[i])
                                  for i in range(len(list(dct[dct_key][key].values())))
                                  if list(dct[dct_key][key].keys())[i] in wordlist]

                # separate key word tuple
                df_dct['name'].extend([key[0] for key in key_word_value])
                df_dct['word'].extend([key[1] for key in key_word_value])
                df_dct['value'].extend([key[2] for key in key_word_value])

        if wordlist is None:

            # find all the words across all texts
            for i, word in enumerate(df_dct['word']):

                # Check if this word is present in other texts
                if word not in processed_words:

                    for other_key, other_value in dct[dct_key].items():

                        # add the necessary info to the dictionary if the word was in another texts top k
                        if other_key != df_dct['name'][i] and word in other_value:
                            df_dct['name'].append(other_key)
                            df_dct['word'].append(word)
                            df_dct['value'].append(other_value[word])
                    processed_words.add(word)

        # create the sankey diagram
        df = pd.DataFrame(df_dct)

        # makes ngram tuple into string
        if not wordcount:
            df['word'] = df['word'].apply(lambda x: ' '.join(x))

        sk.make_sankey(df, 'name', 'word', values='value', threshold=0)

        return

    def sentence_sentiment(self, columns=5, rows=2, hist=True):
        """ Creates a subplot that compares polarity and subjectivity of each text
        Args:
            self (Textastic): class object
            columns (int, default value = 5): number of desired columns of subplots
            rows (int, default value = 2): number of desired rows of subplots
            hist (bool): if true does histogram if false does a kde
        Returns:
            None
        """

        # Create figure and axis for plots
        fig, axs = plt.subplots(rows, columns, figsize=(columns, rows), sharey=True, num='SENTIMENT SUBPLOTS')
        fig.subplots_adjust(hspace=.5, wspace=1)

        count = 0
        for key, value in self.data['sent_sent'].items():
            # Extract polarity and subjectivity values
            polarities = [x[0] for x in self.data['sent_sent'][key]]
            subjectivities = [y[1] for y in self.data['sent_sent'][key]]

            ax = axs[count // columns, count % columns]

            # Set y-axis label only for the first subplot in each row
            if count % columns == 0:
                ax.set_ylabel("Subjectivity")

            # Calculate 2D histogram for density
            if hist:
                h = ax.hist2d(polarities, subjectivities, bins=15, cmap='viridis', cmin=1)

                # create colorbar
                cbar = fig.colorbar(h[3], ax=ax)
                cbar.set_label('Density')
            else:
                # create kde
                plt.subplot(rows, columns, count + 1)
                sns.kdeplot(x=polarities, y=subjectivities)


            count += 1
            ax.set_title(key)

            # set x-axis labels for each subplot
            ax.set_xlabel("Polarity")
            x_ticks = [-1, 0, 1]
            ax.set_xticks(x_ticks)

        plt.show()

    @staticmethod
    def mag(v):
        """ Calculates magnitude of a vector
        Args:
            v (list): vector in question
        Returns:
            magnitude of v (float)
         """
        return (sum([i ** 2 for i in v])) ** .5

    @staticmethod
    def dot(u, v):
        """ Calulates dot product of two vectors
        Args:
            u (list): first vector
            v (list): second vector
        Returns:
            dot product of u and v (int)
        """
        return sum([i * j for i, j in zip(u, v)])

    @staticmethod
    def vectorize(words, unique):
        """ Converts words into vectors
        Args:
            words (list): a user's words
            unique (list): all unique words
        Returns:
            vector with counter values that give a value to how many of the certain unique word was in the user list (list)
        """
        return [Counter(words)[word] for word in unique]

    def cosine_similarity(self, u, v):
        """ Calculates cosine similarity between two vectors
        Args:
            self (Textastic): class object
            u (list): first vector
            v (list): second vector
        Returns:
            None (if vectors' magnitudes are not 0) or their cosine similarity
        """
        if self.mag(u) != 0 and self.mag(v) != 0:
            return self.dot(u, v) / (self.mag(u) * self.mag(v))
        else:
            return

    def cosine_similarity_array(self):
        """ Creates a heatmap of similarity of words in each text
        Args:
            self (Textastic): class object
        Returns:
            None
        """
        dct = self.data['wordcount']
        unique = set()

        # Unique words from all documents not including stop words
        for value in dct.values():
            unique.update([key for key in value.keys() if key not in self.load_stop_words(STOP_WORDS)])
        unique = list(unique)

        # Collect sentiments for words
        lst = list(dct.items())

        arr = np.ones((len(lst), len(lst)), dtype=float)
        x_labels = []
        for i in range(len(lst)):
            vi = self.vectorize(lst[i][1], unique)
            x_labels.append(lst[i][0])
            for j in range(i + 1, len(lst)):
                vj = self.vectorize(lst[j][1], unique)

                arr[i, j] = self.cosine_similarity(vi, vj)
                arr[j, i] = arr[i, j]

        # Set figure size
        plt.figure(figsize=(10, 8), num='COSINE SIMILARITY HEATMAP')

        # Specify the size and location of the axes
        ax = plt.axes([0.25, 0.25, 0.6, 0.6])

        # Create heatmap
        sns.heatmap(arr, xticklabels=x_labels, yticklabels=x_labels, ax=ax, cmap='viridis')

        # Label and show
        plt.show(block=False)
