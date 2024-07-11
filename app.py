import streamlit as st
import pandas as pd
from methods import preprocess
from utils import ArabicIndexer
import base64

# Define ArabicIndexer class

# Load the DataFrame
dataframe = pd.read_csv('Information-Retrieval-Arabic-main\data.csv')

# Define the search function
def get_the_docs(query, df, ground_truth=None):
    indexer = ArabicIndexer.load('Information-Retrieval-Arabic-main\indexer\indexer.pkl')
    query = preprocess(query)
    cosine_similarities = indexer.search(query)
    
    # List to store the updated tuples with titles
    updated_cosine_similarities = []
    
    # Iterate through the cosine similarities
    for docno, cosine_sim in cosine_similarities:
        title = df.loc[df["docno"] == docno, "titles"].iloc[0]
        # Append the updated tuple with title to the list
        updated_cosine_similarities.append((docno, cosine_sim, title))

    # Calculate precision and recall if ground truth is provided
    precision, recall = None, None
    if ground_truth:
        retrieved_docs = [docno for docno, _, _ in updated_cosine_similarities]
        relevant_docs = ground_truth.get(query, [])
        precision, recall = calculate_precision_recall(retrieved_docs, relevant_docs)

    return updated_cosine_similarities, precision, recall

def calculate_precision_recall(retrieved_docs, relevant_docs):
    relevant_retrieved = len(set(retrieved_docs) & set(relevant_docs))
    precision = relevant_retrieved / len(retrieved_docs) if len(retrieved_docs) > 0 else 0
    recall = relevant_retrieved / len(relevant_docs) if len(relevant_docs) > 0 else 0
    return precision, recall




st.title(' Arabic Search Engine :sunglasses:')

# User input for the query
query = st.text_input('Enter your query in Arabic:')


# Load ground truth data
ground_truth = {
    "query1": ["relevant_doc1", "relevant_doc2"],
    "query2": ["relevant_doc3", "relevant_doc4"],
    # Add more queries and relevant documents as needed
}

if st.button('Search'):
    # Perform the search
    results, precision, recall = get_the_docs(query, dataframe, ground_truth)
    
    # Display the results
    if results:
        st.header('Search Results:')
        for i, (docno, cosine_sim, title) in enumerate(results, start=1):
            # Display the title in a smaller font size with a number
            st.subheader(f'{i}. {title}')
            # Display the document ID and cosine similarity
            st.write(f'Document ID: {docno}, Cosine Similarity: {cosine_sim:.2f}')

        # Display precision and recall if available
        if precision is not None and recall is not None:
            st.write(f'Precision: {precision:.2f}')
            st.write(f'Recall: {recall:.2f}')
    else:
        st.write('No matching documents found.')
