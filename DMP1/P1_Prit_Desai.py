import os
import math
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from collections import Counter, defaultdict

# Download necessary NLTK data
nltk.download('stopwords')

# Initialization
corpusroot = './US_Inaugural_Addresses'
tokenizer = RegexpTokenizer(r'[a-zA-Z]+')  # Tokenizer for words
stopWordsFilter = set(stopwords.words('english'))  # Stopwords list
stemmer = PorterStemmer()

# To store preprocessed content of each document
preprocessedDocuments = {}

# Preprocessing function for tokenization, stopword removal, and stemming
def preprocessData(text):
    tokens = tokenizer.tokenize(text.lower())  # Tokenize and lowercase
    tokens_noStopwords = [word for word in tokens if word not in stopWordsFilter]  # Remove stopwords
    return [stemmer.stem(word) for word in tokens_noStopwords]  # Stemming

# Normalize vector for cosine similarity calculation
def normalizeVector(vector):
    normalized = math.sqrt(sum([val ** 2 for val in vector.values()]))
    return {term: (val / normalized) for term, val in vector.items()}

# Cosine similarity calculation
def cosineSimilarity(vectorA, vectorB):
    common = set(vectorA.keys()) & set(vectorB.keys())
    numerator = sum([vectorA[x] * vectorB[x] for x in common])
    sum1 = sum([val ** 2 for val in vectorA.values()])
    sum2 = sum([val ** 2 for val in vectorB.values()])
    denominator = math.sqrt(sum1) * math.sqrt(sum2)
    return float(numerator) / denominator if denominator else 0.0

# Preprocessing the documents
for filename in os.listdir(corpusroot):
    if filename.endswith('.txt'):
        with open(os.path.join(corpusroot, filename), "r", encoding='windows-1252') as file:
            doc = file.read()
            preprocessedDocuments[filename] = preprocessData(doc)

# TF-IDF calculation for documents
# Calculate document frequency (DF)
documentFrequency = Counter()
for tokens in preprocessedDocuments.values():
    documentFrequency.update(set(tokens))

# Number of documents
N = len(preprocessedDocuments)

# Calculate IDF
idf = {term: math.log(N / (dfCount + 1)) + 1 for term, dfCount in documentFrequency.items()}

# Calculate TF-IDF for each document
tf_idf = {}
for doc, tokens in preprocessedDocuments.items():
    termFrequency = Counter(tokens)

    # Calculate logarithmic TF
    log_tf = {term: (1 + math.log(tfCount)) for term, tfCount in termFrequency.items() if tfCount > 0}

    # Calculate TF-IDF and normalize
    tf_idf_temp = {term: log_tf.get(term, 0) * idf[term] for term in tokens}
    tf_idf[doc] = normalizeVector(tf_idf_temp)

# Function to calculate the query vector (without IDF)
def calculate_query_vector(qstring):
    query_tokens = preprocessData(qstring)  # Preprocess the query string
    query_tf = Counter(query_tokens)

    # Calculate logarithmic TF for the query
    query_vector = {token: 1 + math.log(query_tf[token]) for token in query_tf if query_tf[token] > 0}

    # Normalize the query vector
    return normalizeVector(query_vector)

# Function to get IDF of a token
def getidf(token):
    preprocessedToken = preprocessData(token)
    if not preprocessedToken:
        return -1  # If token is filtered out by preprocessing
    token = preprocessedToken[0]
    dfCount = documentFrequency.get(token, 0)
    return math.log10(N / dfCount) if dfCount > 0 else -1

# Function to get normalized TF-IDF weight of a token in a document
def getweight(filename, token):
    preprocessedToken = preprocessData(token)
    if not preprocessedToken:
        return 0  # If token is filtered out
    token = preprocessedToken[0]
    doc_tf_idf = tf_idf.get(filename, {})
    return doc_tf_idf.get(token, 0.0)

# Create postings list for each token
def create_postings_list():
    postings_list = defaultdict(list)
    for doc, tokens in preprocessedDocuments.items():
        for token in set(tokens):
            weight = tf_idf[doc].get(token, 0)
            postings_list[token].append((doc, weight))

    # Sort postings list by weight in descending order
    for token in postings_list:
        postings_list[token].sort(key=lambda x: x[1], reverse=True)

    return postings_list

# Calculate cosine similarity based on query tokens and document weights
def calculate_cosine_similarity(query_vector, doc, postings_list, query_tokens):
    actual_score = 0
    upper_bound_score = 0

    # Query tokens that have the document in the top-10 list (T1)
    for token in query_tokens:
        if token in postings_list:
            top_10_list = postings_list[token][:10]
            doc_weight = next((w for d, w in top_10_list if d == doc), 0)
            if doc_weight > 0:
                actual_score += query_vector[token] * doc_weight

    # Query tokens that do not have the document in the top-10 list (T2)
    for token in query_tokens:
        if token in postings_list:
            top_10_list = postings_list[token][:10]
            if not any(d == doc for d, w in top_10_list):
                upper_bound_weight = top_10_list[-1][1] if len(top_10_list) == 10 else 0
                upper_bound_score += query_vector[token] * upper_bound_weight

    return actual_score, upper_bound_score

# Process a query and return the document with the highest similarity
def query(qstring):
    query_vector = calculate_query_vector(qstring)  # Calculate the query vector
    query_tokens = list(query_vector.keys())  # Get the list of query tokens
    postings_list = create_postings_list()  # Create postings list for all documents

    candidate_docs = defaultdict(lambda: [0, 0])  # Store actual and upper-bound scores for each document
    all_docs = set(preprocessedDocuments.keys())  # Set of all documents in the corpus

    # Track documents appearing in top-10 of each token's posting list
    docs_in_top_10_for_all_tokens = set(all_docs)  # Start with all documents and narrow down based on top-10 results

    # For each query token, get the top-10 elements from its postings list
    for token in query_tokens:
        if token in postings_list:
            top_10_list = postings_list[token][:10]
            top_10_docs = {doc for doc, weight in top_10_list}

            # Narrow down the list of candidate documents by intersecting with the top-10 results for this token
            docs_in_top_10_for_all_tokens &= top_10_docs

            # Add the actual score for each document in the top-10
            for doc, weight in top_10_list:
                candidate_docs[doc][0] += query_vector[token] * weight

    # If no document appears in the top-10 for all query tokens, return 'fetch more'
    if not docs_in_top_10_for_all_tokens:
        return ("fetch more", 0)

    # Calculate upper-bound scores for documents not in the top-10 list for some tokens
    for doc in all_docs:
        actual_score, upper_bound_score = calculate_cosine_similarity(query_vector, doc, postings_list, query_tokens)
        candidate_docs[doc][0] = actual_score
        candidate_docs[doc][1] = actual_score + upper_bound_score

    # Find the document with the highest actual score
    best_doc = None
    best_actual_score = 0
    best_upper_bound_score = 0
    for doc, (actual_score, upper_bound_score) in candidate_docs.items():
        if actual_score >= best_actual_score:  # Update the best document based on actual score
            best_doc = doc
            best_actual_score = actual_score
            best_upper_bound_score = upper_bound_score

    # If no document is found or more than 10 elements are needed, return 'fetch more'
    if best_doc is None or best_upper_bound_score > best_actual_score:
        return ("fetch more", 0)

    # Handle the case where no document matches any query tokens
    if best_actual_score == 0:
        return ("None", 0)

    # Return the document with the highest similarity score
    return (best_doc, best_actual_score)

# Test the functions
print("%.12f" % getidf('democracy'))
print("%.12f" % getidf('foreign'))
print("%.12f" % getidf('states'))
print("%.12f" % getidf('honor'))
print("%.12f" % getidf('great'))
print("--------------")
print("%.12f" % getweight('19_lincoln_1861.txt', 'constitution'))
print("%.12f" % getweight('23_hayes_1877.txt', 'public'))
print("%.12f" % getweight('25_cleveland_1885.txt', 'citizen'))
print("%.12f" % getweight('09_monroe_1821.txt', 'revenue'))
print("%.12f" % getweight('37_roosevelt_franklin_1933.txt', 'leadership'))
print("--------------")
print("(%s, %.12f)" % query("states laws"))
print("(%s, %.12f)" % query("war offenses"))
print("(%s, %.12f)" % query("british war"))
print("(%s, %.12f)" % query("texas government"))
print("(%s, %.12f)" % query("world civilization"))
