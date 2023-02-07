# This program will show the results of how closely nlp calculates words and sentences to be similar to each other.

import spacy

nlp = spacy.load('en_core_web_md')

word1 = nlp("cat")
word2 = nlp("monkey")
word3 = nlp("banana")

# Cat and monkey are almost 60% similar which makes sense as they are both mammals but from different species.
print(f"The similarity between {word1} and {word2} is: {word1.similarity(word2)}")

# Banana and monkey have a 40% result of similarity and as they are often associated with each other,
# but not technically part of the same category in a sense, I found this is interesting.
print(f"The similarity between {word3} and {word2} is: {word3.similarity(word2)}")

# That Banana and cat are not considered very similar is not that surprising.
print(f"The similarity between {word3} and {word1} is: {word3.similarity(word1)}")

print("------------------------------")


# Below is my own example
w1 = nlp("wood")
w2 = nlp("violin")
w3 = nlp("guitar")

# It's interesting that 'wood' and 'violin' are considered less similar or related than 'monkey' and 'banana'.
print(f"This is the result of comparing {w1} to {w2}: {w1.similarity(w2)}")

# It's not surprising that 'violin' and 'guitar' are similar as they are both instruments of the string class.
print(f"This is the result of comparing {w2} to {w3}: {w2.similarity(w3)}")

# It's curious that 'guitar' is considered closer to 'wood' than 'violin' is!
# Perhaps the percentage of the instrument that is made from wood is higher in a guitar.
print(f"This is the result of comparing {w3} to {w1}: {w3.similarity(w1)}")

print("-------------------------------")

tokens = nlp('cat apple monkey banana')

for token1 in tokens:
    for token2 in tokens:
        print(token1.text, token2.text, token1.similarity(token2))

print("------------------------------")

sentence_to_compare = "Why is my cat on the car"

sentences = ["where did my dog go",
             "Hello, there is my car",
             "I\'ve lost my car in my car",
             "I\'d like my boat back",
             "I will name my dog Diana"]

model_sentence = nlp(sentence_to_compare)

for sentence in sentences:
    similarity = nlp(sentence).similarity(model_sentence)
    print(sentence + " - ", similarity)

# When running the example program with 'en_core_web_sm', the results of the similarities are generally lower.
# The following message was also printed with the results: "The model you're using has no word vectors loaded,
# so the result of the Doc.similarity method will be based on the tagger, parser and NER,
# which may not give useful similarity judgements.
# This may happen if you're using one of the small models, e.g. `en_core_web_sm`, which don't ship with word vectors
# and only use context-sensitive tensors. You can always add your own word vectors, or use one of the larger models
# instead if available."
# Which suggests less accuracy.
