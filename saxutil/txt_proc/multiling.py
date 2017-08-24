import unicodedata
from collections import Counter
from langdetect import detect_langs
import nltk


def strip_accents(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


def get_langdetect_proportions(doclist):
    d = {}
    for doc in doclist:
        langlist = detect_langs(doc.lower())
        for lang in langlist:
            cur_prob = d.get(lang.lang, 0.0)
            d[lang.lang] = cur_prob + lang.prob
    for lang in list(d.keys()):
        d[lang] /= float(len(doclist))
    return d


class BowLangTracker(object):
    def __init__(self):
        self.wordset_on_lang = {}

    def add_vocab(self, langcode, vocab):
        newvocab = set([w.lower() for w in vocab])
        curvocab = self.wordset_on_lang.get(langcode, set())
        self.wordset_on_lang[langcode] = newvocab.union(curvocab)

    def langcodes(self):
        return self.wordset_on_lang.keys()

    def langwords(self, langcode):
        return self.wordset_on_lang[langcode]

    def distinct_lang_words(self):
        words = []
        for lc in self.langcodes():
            words.extend(self.langwords(lc))
        cntr = Counter(words)
        dwords = set()
        for w, cnt in cntr.items():
            if cnt == 1:
                dwords.add(w)
        return dwords

    def rm_non_distinct_words_from_lang(self, langcode):
        dwords = self.distinct_lang_words()
        lwords = set()
        for w in self.langwords(langcode):
            if w in dwords:
                lwords.add(w)
        self.wordset_on_lang[langcode] = lwords

    def load_en_words(self):
        self.add_vocab('en', nltk.corpus.words.words())

    def load_floresta(self):
        self.add_vocab('pt', nltk.corpus.floresta.words())

    def load_cess_esp(self):
        self.add_vocab('es', nltk.corpus.cess_esp.words())

    def load_udhr_de(self):
        self.add_vocab('de', nltk.corpus.udhr.words('German_Deutsch-Latin1'))

    def has_word(self, tkns, langcode):
        langwords = self.langwords(langcode)
        for t in tkns:
            if t in langwords:
                return True
        return False
