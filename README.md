# Arabic Search Engine

This project is an Arabic search engine that preprocesses data, applies normalization, lemmatization, and removes stop words. It then indexes the data and retrieves matching documents based on cosine similarity. The deployment is done using Streamlit to provide visuals for the search engine.

## Table of Contents
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Preprocessing Steps](#preprocessing-steps)
  - [Data Cleaning](#data-cleaning)
  - [Normalization](#normalization)
  - [Lemmatization](#lemmatization)
  - [Stop Words Removal](#stop-words-removal)
- [Indexing](#indexing)
- [Retrieving and Ranking](#retrieving-and-ranking)
- [Deployment](#deployment)
- [Contributing](#contributing)
- [License](#license)

## Features
- Arabic text preprocessing
  - Data cleaning
  - Normalization
  - Lemmatization
  - Stop words removal
- Indexing of processed data
- Retrieval of matching documents based on cosine similarity
- Deployment using Streamlit for visual interface

## Installation
1. Clone the repository:
    ```bash
    git clone https://github.com/waleedhussien11/Arabic-Search-Engine-.git
    ```
2. Change into the project directory:
    ```bash
    cd arabic-search-engine
    ```
3. Create a virtual environment:
    ```bash
    python -m venv venv
    ```
4. Activate the virtual environment:
    - On Windows:
      ```bash
      venv\Scripts\activate
   
```

## Usage
1. Prepare your data by placing it in the `data` directory.
2. Run the preprocessing script:
    ```bash
    python preprocess.py
    ```
3. Run the indexing script:
    ```bash
    python index.py
    ```
4. Start the Streamlit app:
    ```bash
    streamlit run app.py
    ```

## Preprocessing Steps

### Data Cleaning
- Remove unnecessary characters, symbols, and punctuation.

### Normalization
- Normalize Arabic text to a consistent form.

### Lemmatization
- Convert words to their base form.

### Stop Words Removal
- Remove common Arabic stop words that do not contribute to the meaning.

## Indexing
- Index the processed data to facilitate efficient retrieval.

## Retrieving and Ranking
- Retrieve documents that match the query.
- Rank the documents based on cosine similarity to the query.

## Deployment
- The application is deployed using Streamlit, providing a visual interface for the search engine.

## Contributing
Contributions are welcome! Please fork the repository and create a pull request.

## License
This project is licensed under the MIT License.
