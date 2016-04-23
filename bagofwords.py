
import random
import theano
import numpy
import theano.tensor as TT

spamemails = ['this email is spam', 'get lots of traffic', 'easy cash one click', 'nigerian prince', 'free traffic for your site', 'buy gold get cash', 'get nigerian prince inheritance', 'free whatever'] * 4
hamemails = ['hey how you doing', 'lets work on that assignment', 'what time does the library open'] * 8

dictionary = []
word_histogram = {}

wholecorpus = spamemails + hamemails
for email in wholecorpus:
    for word in email.split():
        if word not in word_histogram:
            word_histogram[word] = 1
        else:
            word_histogram[word] += 1

inverse_histogram = [[] for z in xrange(100)]
for word, count in word_histogram.items():
    inverse_histogram[count].append(word)


dictsize = 30
while len(dictionary) < dictsize:
    while len(inverse_histogram[-1]) == 0:
        inverse_histogram.pop()
    dictionary.append(inverse_histogram[-1][-1])
    inverse_histogram[-1].pop()

def getBagOfWords(email, dictionary):
    bag = [0] * len(dictionary)
    for word in email.split():
        if word in dictionary:
            bag[dictionary[word]] += 1
    return bag

inverse_dictionary = dict(enumerate(dictionary))
dictionary = dict(zip(inverse_dictionary.values(), inverse_dictionary.keys()))

inputNodes = dictsize
hiddenNodes = 10
W1 = theano.shared(numpy.random.rand(inputNodes, hiddenNodes))
W2 = theano.shared(numpy.random.rand(hiddenNodes, 1))

x = TT.lvector('x')
y = TT.scalar('y')
hidden_output = TT.nnet.sigmoid(TT.dot(x, W1))
output = TT.nnet.sigmoid(TT.dot(hidden_output, W2))

corpus = zip(spamemails, [1] * len(spamemails)) + \
         zip(hamemails, [0] * len(hamemails))


error = TT.sum((y - output) ** 2)

weights = [W1, W2]
gradients = TT.grad(error, weights)
learning_rate = 0.001
updates = [(weight, weight - learning_rate * gradient)
           for weight, gradient in zip(weights, gradients)]

update_function = theano.function([x, y], [error], updates = updates)

iterations = 1000000
for i in xrange(iterations):
    sample = random.choice(corpus)
    bagofwords = getBagOfWords(sample[0], dictionary)
    label = sample[1]
    update_function(bagofwords, label)

def test(email):
    outputval = output.eval({ x: getBagOfWords(email[0], dictionary) })
    print "%50s %4d %4f" % (email[0], email[1], outputval)

for email in corpus:
    test(email)

sentence = raw_input()
while sentence != 'quit':
    test((sentence, -1))
    sentence = raw_input()

