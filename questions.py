import nltk
import sys
import os
import string
import math

FILE_MATCHES = 1
SENTENCE_MATCHES = 1


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python questions.py corpus")

    # Calculate IDF values across files
    files = load_files(sys.argv[1])
    file_words = {
        filename: tokenize(files[filename])
        for filename in files
    }
    file_idfs = compute_idfs(file_words)

    # Prompt user for query
    query = set(tokenize(input("Query: ")))

    # Determine top file matches according to TF-IDF
    filenames = top_files(query, file_words, file_idfs, n=FILE_MATCHES)

    # Extract sentences from top files
    sentences = dict()
    for filename in filenames:
        for passage in files[filename].split("\n"):
            for sentence in nltk.sent_tokenize(passage):
                tokens = tokenize(sentence)
                if tokens:
                    sentences[sentence] = tokens

    # Compute IDF values across sentences
    idfs = compute_idfs(sentences)

    # Determine top sentence matches
    matches = top_sentences(query, sentences, idfs, n=SENTENCE_MATCHES)
    for match in matches:
        print(match)


def load_files(directory): # successfully implemented
    """
    Given a directory name, return a dictionary mapping the filename of each
    `.txt` file inside that directory to the file's contents as a string.
    """
    files = os.listdir(directory)
    file_data_list = list()

    for file in files:
        with open(os.path.join(directory, file), encoding='utf-8') as f:
            text = f.read()
            file_data_list.append((file, text))
        print(f"loaded {file}")

    file_data = dict(file_data_list)
    
    return file_data

def tokenize(document): # successfully implemented
    """
    Given a document (represented as a string), return a list of all of the
    words in that document, in order.

    Process document by coverting all words to lowercase, and removing any
    punctuation or English stopwords.
    """
    tokens = nltk.word_tokenize(document)
    PerfectTokens = list()
    for token in tokens:
        lowerToken = token.lower()
        if lowerToken in nltk.corpus.stopwords.words("english"):
            continue

        PerfectToken = lowerToken

        index = 0
        for char in string.punctuation:
            PerfectToken = PerfectToken.replace(char, "")
        if PerfectToken == '' or PerfectToken in string.punctuation:
            continue
        PerfectTokens.append(PerfectToken)
    return PerfectTokens


def compute_idfs(documents): # successfully implemented
    """
    Given a dictionary of `documents` that maps names of documents to a list
    of words, return a dictionary that maps words to their IDF values.

    Any word that appears in at least one of the documents should be in the
    resulting dictionary.
    """
    word_bank = list()
    word_idfs = list()
    unique_word_bank = list()

    TotalDocuments = len(documents)
    for document in documents:
        word_bank.extend(documents[document])

    for word in word_bank:
        if word in unique_word_bank:
            continue
        unique_word_bank.append(word)

    for word in unique_word_bank:
        NumDocumentsContaining = 0

        for document in documents:
            if word in documents[document]:
                NumDocumentsContaining += 1

        idf = math.log(TotalDocuments * 1.0 / NumDocumentsContaining)
        word_idfs.append((word, idf))

    word_idfs = dict(word_idfs)
    return word_idfs

def TF(word, word_bank):
    freq = 0
    for item in word_bank:
        if item == word:
            freq += 1

    return freq

def top_files(query, files, idfs, n): # successfully implemented
    """
    Given a `query` (a set of words), `files` (a dictionary mapping names of
    files to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the filenames of the the `n` top
    files that match the query, ranked according to tf-idf.
    """
    files_score = list()

    for file in files:
        netscore = 0
        for word in query:
            try:
                tfidf = TF(word, files[file]) * idfs[word]
                netscore += tfidf
            except KeyError:
                pass
        files_score.append((file, netscore))

    desc_files_score = sorted(files_score, key =lambda value: value[1])
    l = len(desc_files_score)

    n_top_files = [desc_files_score[l-1-k][0] for k in range(0, n)]

    return n_top_files


def top_sentences(query, sentences, idfs, n):
    """
    Given a `query` (a set of words), `sentences` (a dictionary mapping
    sentences to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the `n` top sentences that match
    the query, ranked according to idf. If there are ties, preference should
    be given to sentences that have a higher query term density.
    """
    sentence_score = list()
    for sentence in sentences:
        S = sentences[sentence]

        QTD = 0 # Query Term Density
        MC = 0 # Match Word Counter
        SumIDF = 0
        for word in query:
            if word in S:
                MC += 1
                try:
                    SumIDF += idfs[word]
                except:
                    pass
        QTD = MC * 1.0 / len(S)

        sentence_score.append((sentence, SumIDF, QTD))

    # Ranking based on score
    desc_sentence_score = sorted(sentence_score, key =lambda value: value[1])

    l = len(desc_sentence_score)
    n_top_sentences = [desc_sentence_score[l-1-k] for k in range(0, n)]
    n_top_sentences.append(("", 0, 0))
    # Ranking based on QTD
    equal_idf_sentences = list()
    already_added = list()
    for index in range(0, len(n_top_sentences)-1):
        S_base = n_top_sentences[index]

        similarIdfs = set()
        similarIdfs.add(S_base)

        for rem in range(index, len(n_top_sentences)-1):
            if S_base[1] == n_top_sentences[rem][1]:
                similarIdfs.add(n_top_sentences[index])

        equal_idf_sentences.append(list(similarIdfs))
    best_sentences = list()
    for similarIdf in equal_idf_sentences:
        ordered = sorted(similarIdf, key =lambda value: value[2])
        for each in ordered:
            best_sentences.append(each[0])

    return best_sentences

if __name__ == "__main__":
    main()
