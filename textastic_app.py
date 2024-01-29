"""Jeremiah Payeur, Madelyn Redick, Sahana Dhar

textastic_app.py

Visualizing Cold War Speeches based on words sentiment and cosine similarity"""

from textastic import Textastic
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

Eisenhower = 'Dwight D. Eisenhower - Atoms for Peace (1953).txt'
JFK = 'John F. Kennedy - Ich bin ein Berliner (1963).txt'
Truman = 'Harry S. Truman - Truman Doctrine (1947).txt'
Stalin = 'Joseph Stalin - Election Speech (1946).txt'
Marx = 'Karl Marx - Extracts from the Communist Manifesto (1848).txt'
Zhivkov_Brezhnev = 'Meeting between Comrades Leonid Ilyich Brezhnev and Todor Zhivkov.txt'
Khrushchev = 'Nikita Khrushchev - Secret Speech to 20th Party Congress (1956).txt'
Reagan_Empire = 'Ronald Reagan - Evil Empire (1983).txt'
Reagan_Wall = 'Ronald Reagan - Tear Down This Wall_ (1987).txt'
Lenin = 'Vladimir Ilyich Lenin - Power to the Soviets (1917).txt'


def main():
    tt = Textastic()
    tt.load_text(Eisenhower, 'Eisenhower')
    tt.load_text(JFK, 'JFK')
    tt.load_text(Truman, 'Truman')
    tt.load_text(Stalin, 'Joseph Stalin')
    tt.load_text(Marx, 'Marx')
    tt.load_text(Zhivkov_Brezhnev, 'Zhivkov_Brezhnev')
    tt.load_text(Khrushchev, 'Khrushchev')
    tt.load_text(Reagan_Empire, 'Reagan_Empire')
    tt.load_text(Reagan_Wall, 'Reagan_Wall')
    tt.load_text(Lenin, 'Lenin')

    tt.sankey_ngrams(k=6)
    tt.wordcount_sankey(k=3)
    tt.wordcount_sankey(k=3, wordcount=False)
    tt.cosine_similarity_array()
    tt.sentence_sentiment()
    tt.sentence_sentiment(hist=False)


if __name__ == "__main__":
    main()
