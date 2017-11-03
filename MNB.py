import math
import os
import re
import sys
from collections import Counter
from collections import defaultdict

import nltk
import numpy as np
from nltk.corpus import stopwords


def training_mnb(class_list, training_set):
    #count = 1
    vocabulary = list()
    total_docs = 0
    count_docs_in_class = list()
    prior_list = list()
    text_of_each_class = list()
    for name in sorted(os.listdir(training_set)):
        if os.path.isdir(os.path.join(training_set, name)) and '''count <= len(class_list)''':
            local_docs = 0
            test_class = ""
            for file_list in os.listdir(os.path.join(training_set, name)):
                total_docs += 1
                local_docs += 1
                f = open(os.path.join(training_set, name, file_list))
                read_file_class = str(f.readlines())
                tokens = nltk.word_tokenize(read_file_class)
                stop_list = stopwords.words('english')
                read_file_class = str([word.lower() for word in tokens if
                                       word.isalpha() and not word in stop_list and len(word) > 2])
                read_file_class = re.sub(r'[<|>|?|_|,|!|:|;|(|)|\"|=|-|$|\\|/|*|\'|+|\[|\]|#|$|%|^|?|~|`]', r'',
                                         read_file_class)
                test_class = test_class + " " + read_file_class
            vocabulary.extend(test_class.split())
            text_of_each_class.append(test_class)
            count_docs_in_class.append(local_docs)
            #count += 1
    vocabulary = set(vocabulary)
    i = 0
    conditional_prob_class = list()

    for class_name in class_list:
        tct = list()
        conditional_prob_term = defaultdict(list)
        prior_list.append(count_docs_in_class[i] / total_docs)
        counterText = Counter(text_of_each_class[i].split())
        for term in vocabulary:
            tct.append(counterText[term])
        j = 0
        tct_sum = sum(tct)
        length_tct = len(tct)
        for term in vocabulary:
            conditional_prob_term[term].append((tct[j] + 1) / (tct_sum + length_tct))
            j += 1
        conditional_prob_class.append(conditional_prob_term)
        i += 1
    return vocabulary, prior_list, conditional_prob_class


def apply_mnb(class_list, prior, conditional_prob_class, document, test_set):
    score = list()
    f = open(os.path.join(test_set, name, document))
    token_string = " ".join(f.readlines())
    token_list = re.sub(r'[<|>|?|_|,|!|:|;|(|)|\"|=|-|$|\\|/|*|\'|+|\[|\]|#|$|%|^|?|~|`]', r'', token_string)
    tokens = nltk.word_tokenize(token_list)
    stop_list = stopwords.words('english')
    w_vocabulary = set([word.lower() for word in tokens if word.isalpha() and not word in stop_list and len(word) > 2])
    m = 0
    for class_name in class_list:
        score.append(math.log(prior[m]))
        for term in w_vocabulary:
            if not conditional_prob_class[m][term]:
                continue
            score[m] += math.log(conditional_prob_class[m][term][0])
        m += 1
    # score = np.array(score)
    return np.argmax(np.array(score))


if __name__ == "__main__":
    training_set = sys.argv[1]
    test_set = sys.argv[2]

    class_list = list()

    dir_name = os.path.split(training_set)[1]
    #count = 1
    #folders = 5
    for name in sorted(os.listdir(training_set)):
        if os.path.isdir(os.path.join(training_set, name)) and '''count <= folders''':
            class_list.append(name)
            #count += 1

    #count = 1
    vocabulary, prior_list, conditional_prob_class = training_mnb(class_list, training_set)
    success = 0
    failure = 0
    for name in sorted(os.listdir(test_set)):
        if os.path.isdir(os.path.join(test_set, name)) and '''count <= folders''':
            for doc in os.listdir(os.path.join(test_set, name)):
                folder_predicted = apply_mnb(class_list, prior_list, conditional_prob_class, doc, test_set)
                if class_list[int(folder_predicted)] == name:
                    success += 1
                else:
                    failure += 1
        #count += 1
    print("Success : " + str(success))
    print("Failure : " + str(failure))
    print("Accuracy : " + str(success * 100 / (success + failure)))
