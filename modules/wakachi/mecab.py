import MeCab

m = MeCab.Tagger('-Owakati')


# Split sentence by word
def divide_word(sentence):
    return m.parse(sentence).strip().split()


def divide_text(text):
    sentences = []

    for line in text.split('\n'):
        sentences += divide_word(line)

    return sentences
