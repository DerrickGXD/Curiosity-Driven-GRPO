import sys
sys.path.append('/home/derrick/deepscaler/')

from curiosity_redteam.incontext_sentence_embed import InContextCosineSentenceEmbeddingReward


with open("output.txt", "r") as f:
    output = f.read()


list_of_sentences = output.split("</think>")[0].replace("\\n\\n", " ").split(". ")



curiosity_penalty_fn = InContextCosineSentenceEmbeddingReward()

for i, sentence in enumerate(list_of_sentences):

    penalty = curiosity_penalty_fn([sentence])
    print(i, sentence, penalty)
    curiosity_penalty_fn.append_reference(sentence)
 