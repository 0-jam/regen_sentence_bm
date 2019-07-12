import MeCab

m = MeCab.Tagger('-Owakati')


# Split sentence by word
def divide_word(sentence):
    return m.parse(sentence).strip().split()
