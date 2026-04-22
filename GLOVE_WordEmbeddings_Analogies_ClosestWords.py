import torch
import torchtext.vocab
from torchtext.vocab import GloVe

glove = GloVe(name='6B', dim=100)
print(glove.itos[:15])
print(glove.stoi['the'])

def get_vector(embeddings, word):
    assert word in embeddings.stoi, f'{word}* is not in the vocab!'
    return embeddings.vectors[embeddings.stoi[word]]

get_vector(glove, 'paper')

def closest(embeddings, vector, n = 6):
    
    distances = []

    for neighbor in embeddings.itos:
        distances.append((neighbor, torch.dist(vector, get_vector(embeddings, neighbor))))

    return sorted(distances, key = lambda x: x[1])[:n]

closest(glove, get_vector(glove, 'paper'))
closest(glove, get_vector(glove, 'shenanigans'))

def print_tuples(tuples):

    for t in tuples:
        print('(%.4f) %s' % (t[1], t[0]))

print_tuples(closest(glove, get_vector(glove, 'stupendous')))

def analogy(embeddings, w1, w2, w3, n = 6):
    
    print('\n[%s : %s :: %s : ?]' % (w1, w2, w3))

    closest_words = closest(embeddings, \
                            get_vector(embeddings, w2)
                           -get_vector(embeddings, w1) \
                           +get_vector(embeddings, w3), \
                            n + 3)

    closest_words = [x for x in closest_words if x[0] not in [w1, w2, w3]][:n]
    return closest_words

print_tuples(analogy(glove, 'moon', 'night', 'sun'))
print_tuples(analogy(glove, 'fly', 'bird', 'swim'))
print_tuples(analogy(glove, 'earth', 'moon', 'sun'))