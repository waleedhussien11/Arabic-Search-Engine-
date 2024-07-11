
import math
import pandas as pd
from collections import defaultdict
from pyarabic import araby
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


class ArabicTfidfVectorizer:
    def __init__(self):
        self.term_document_frequency = defaultdict(set)
        self.documents = pd.DataFrame(columns=["docno", "content", "titles"])

    def _calculate_query_tfidf(self, docs, term_document_frequency):
        tfidf_matrix = []

        for doc in docs:
            tfidf_vector = {}
            tokens = araby.tokenize(doc)
            token_count = len(tokens)
            term_counts = defaultdict(int)

            for term in tokens:
                term_counts[term] += 1

            for term, count in term_counts.items():
                if term in term_document_frequency:
                    tf = count / token_count
                    idf = math.log(len(self.documents) / len(term_document_frequency[term]))
                    tfidf = tf * idf
                    tfidf_vector[term] = tfidf

            tfidf_matrix.append(tfidf_vector)

        return tfidf_matrix

    def transform(self, docs):
        return self._calculate_query_tfidf(docs, self.term_document_frequency)
    
    def fit_transform(self, docs):
        self.documents = docs
        self.term_document_frequency = self._calculate_term_document_frequency()
        return self._calculate_tfidf(self.documents, self.term_document_frequency)

    def _calculate_term_document_frequency(self, docs=None):
        if docs is None:
            docs = self.documents
        term_document_frequency = defaultdict(set)
        for doc_id, doc in docs.iterrows():
            tokens = araby.tokenize(doc['content'])
            unique_terms = set(tokens)
            for term in unique_terms:
                term_document_frequency[term].add(doc_id)
        return term_document_frequency

    def _calculate_tfidf(self, docs, term_document_frequency):
        tfidf_matrix = []

        for doc_id, doc in docs.iterrows():
            tfidf_vector = {}
            tokens = araby.tokenize(doc['content'])
            token_count = len(tokens)
            term_counts = defaultdict(int)

            for term in tokens:
                term_counts[term] += 1

            for term, count in term_counts.items():
                tf = count / token_count
                idf = math.log(len(docs) / len(term_document_frequency[term]))
                tfidf = tf * idf
                tfidf_vector[term] = tfidf

            tfidf_matrix.append(tfidf_vector)

        return tfidf_matrix
    

class ArabicIndexer(ArabicTfidfVectorizer):  
    def __init__(self):
        super().__init__()  
        self.index = defaultdict(list)
        self.docno_to_title = {}

    def add_documents(self, docs):
        for doc_id, content, title in docs:
            new_doc = pd.DataFrame({"docno": [doc_id], "content": [content], "titles": [title]})
            self.documents = pd.concat([self.documents, new_doc], ignore_index=True)
            self.docno_to_title[doc_id] = title
        self._create_index()

    def _create_index(self):
        tfidf_matrix = self.fit_transform(self.documents)
        for doc_id, doc_vector in enumerate(tfidf_matrix):
            docno = self.documents.iloc[doc_id]['docno']
            for term, tfidf_weight in doc_vector.items():
                self.index[term].append((docno, tfidf_weight))

    def search(self, query, top_n=10):
        query_vector = self.transform([query])[0]  
        scores = defaultdict(float)

        for term, query_weight in query_vector.items():
            if term in self.index:
                for doc_id, doc_weight in self.index[term]:
                    scores[doc_id] += query_weight * doc_weight

        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_scores[:top_n]
    from collections import defaultdict

    def search_cosine(self, query, top_n=10):
        query_vector = self.transform([query])[0]  # Calculate the TF-IDF vector for the query

        scores = defaultdict(float)
        
        # Calculate cosine similarity between the query vector and each document vector
        for doc_id, doc_vector in self.document_vectors.items():
            cosine_sim = np.dot(query_vector, doc_vector) / (np.linalg.norm(query_vector) * np.linalg.norm(doc_vector))
            scores[doc_id] = cosine_sim

        # Sort the documents based on cosine similarity scores
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        return sorted_scores[:top_n]

    
    def _query_tfidf_to_array(self, query_tfidf):
        terms = list(query_tfidf.keys())
        tfidf_values = list(query_tfidf.values())
        return [tfidf_values[terms.index(term)] if term in terms else 0 for term in self.documents.columns]

    def _doc_tfidf_to_array(self, doc_tfidf):
        return [doc_tfidf[term] if term in doc_tfidf else 0 for term in self.documents.columns]

     

    @staticmethod
    def load(filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)