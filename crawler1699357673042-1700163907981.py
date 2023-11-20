#Start by importing the libraries
import json
import math
import os
import time
import requests
from bs4 import BeautifulSoup
import nltk
from nltk.tokenize import RegexpTokenizer
import schedule
import urllib.robotparser
import tkinter as tk
from tkinter import scrolledtext

tokenizer = RegexpTokenizer(r'[a-zA-Z]+')

# stop_words = set(stopwords.words('english'))  # This is not working since there is some download error

# So, defining a list of stopwords
stopwords = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]

# This function pre processes a textual data
def preprocess(text):
    text = text.lower()    # Lowercase the text
    text = ''.join([c for c in text if c.isalpha() or c.isspace()])   # Remove any punctuations, numbers or symbols 

    # tokenize, remove stopwords, and stemming
    tokens = tokenizer.tokenize(text)
    stop_words = set(stopwords)
    words = [w for w in tokens if w not in stop_words]
    stemmer = nltk.stem.PorterStemmer()
    words = [stemmer.stem(w) for w in words]

    return words

# Function to create inverted index
def create_index(information):

    # Inverted index dict
    index = {}

    # Unique doc ID for each record
    doc_id = 0

    for page in information:
        for record in page:
            doc_id = doc_id + 1

            # Preprocess the crawled data
            title_tokens = preprocess(record['title'])
            journal_tokens = preprocess(record['journal'])

            #Build inverted index
            for token in title_tokens + journal_tokens:
                if token in index:
                    index[token].append(doc_id)
                else:
                    index[token] = [doc_id]
    return index

# Function to calculate tf-idf
def calculate_tfidf(information, index):

    # TF-IDF (Term Frequency - Inverse Document Frequency) dict
    tfidf = {}

    # Unique doc ID for each record
    doc_id = 0

    for page in information:
        for record in page:
            doc_id = doc_id + 1

            # PReprocess the data
            title_tokens = preprocess(record['title'])
            journal_tokens = preprocess(record['journal'])

            #Get the required tokens 
            doc_tokens = title_tokens + journal_tokens

            # Find out max word count in a record
            max_freq = max(doc_tokens.count(w) for w in doc_tokens)

            # For each record, we store the tfidf for each term in another dict
            tfidf[doc_id] = {}

            for token in doc_tokens:
                tf = doc_tokens.count(token) / max_freq
                idf = math.log(len(information) / len(index[token]))
                tfidf[doc_id][token] = tf * idf

    return tfidf

# Function which ranks the records based on tfidf  score
def rank_documents(query, information, index, tfidf):

    # Preprocess user query
    query_tokens = preprocess(query)

    # Compute the score based on tfidf
    scores = {}
    for token in query_tokens:
        if token in index:
            for doc_id in index[token]:
                if doc_id in scores:
                    scores[doc_id] += tfidf[doc_id][token]
                else:
                    scores[doc_id] = tfidf[doc_id][token]
    
    # Return the sorted (descending) list of scores
    return sorted(scores, key=scores.get, reverse=True)


def can_fetch(url, user_agent='*'):
    rp = urllib.robotparser.RobotFileParser()
    rp.set_url(urllib.parse.urljoin(url, '/robots.txt'))
    rp.read()
    return rp.can_fetch(user_agent, url)

# Web crawler
def crawler(url):

    response = requests.get(url)

    # Must be polite by preserving robots.txt rules and not hitting the servers too fast
    time.sleep(5)

    soup = BeautifulSoup(response.text, "html.parser") # Parse data

    # Find all the publications related data
    publications = soup.find_all("div", class_="result-container")

    info = []
    for publication in publications:

        #Get title and publication link
        title = publication.find("h3", class_="title").text
        publication_link = publication.find("a", class_="link")["href"]

        # Get authors and their profiles
        authors = publication.find_all("a", class_="link person")
        authors = [author.text for author in authors]
        author_links = publication.find_all("a", class_="link person", rel="Person")
        author_links = [author["href"] for author in author_links]

        # GEt date, journal, volume, number of pages, and article id
        date = publication.find("span", class_="date").text
        journal = publication.find("span", class_="journal")
        if journal is not None:
            journal = journal.get_text()
        else:
            journal = ""
        volume = publication.find("span", class_="volume")
        if volume is not None:
            volume = volume.get_text()
        else:
            volume = ""
        numberofpages = publication.find("span", class_="numberofpages")
        if numberofpages is not None:
            numberofpages = numberofpages.get_text()
        else:
            numberofpages = ""
        article_id = publication.find("p", class_="type").text.split()[-1]

        # Build a dict storing all info and append it to info list
        publication_info = {
            "title": title,
            "publication_link": publication_link,
            "authors": authors,
            "authors_profiles": author_links,
            "date": date,
            "journal": journal,
            "volume": volume,
            "numberofpages": numberofpages,
            "article_id": article_id
        }
        info.append(publication_info)

    return info

# Define the get_document_by_id function
def get_document_by_id(doc_id, information):
    doc_id_counter = 0
    for page in information:
        for record in page:
            doc_id_counter += 1
            if doc_id_counter == doc_id:
                return record
    return None

def gui_search_engine(information, index, tfidf):
    # GUI setup
    window = tk.Tk()
    window.title("Search Engine")

    def search():
        query = entry.get()
        ranked_docs = rank_documents(query, information, index, tfidf)[:10]
        result_text.delete(1.0, tk.END)
        for doc_id in ranked_docs:
            doc_info = get_document_by_id(doc_id, information)
            if doc_info:
                result_text.insert(tk.END, f"Document ID: {doc_id}\nTitle: {doc_info['title']}\nLink: {doc_info['publication_link']}\n\n")

    # Create the label, entry, button, and text widgets
    label = tk.Label(window, text="Enter your query:")
    label.pack()

    entry = tk.Entry(window, width=50)
    entry.pack()

    search_button = tk.Button(window, text="Search", command=search)
    search_button.pack()

    result_text = scrolledtext.ScrolledText(window, wrap=tk.WORD, width=80, height=40)
    result_text.pack()

    # Start the GUI event loop
    window.mainloop()
# Used to crawl the data and store it in  JSON FILE
def crawl_data(json_file):
    information = []
    for i in range (9):
        pub = crawler(f"https://pureportal.coventry.ac.uk/en/organisations/ics-research-centre-for-fluid-and-complex-systems-fcs/publications/?page={i}")
        information.append(pub)
    with open(json_file, "w") as f:
        json.dump(information, f)
    return information

def main():

    # This file will store all our crawled data
    # This is to avoid multiple crawlings everytime code is run
    json_file = "information.json"

    # Check if the JSON file exists or not
    if os.path.exists(json_file) and os.path.isfile(json_file):

        # If exists, then just open the file and load all data to information list
        with open(json_file, "r") as f:
            information = json.load(f)
    else:
        # Else, perform web crawling and populate information
        # The 10 web pages are crawled to get all publication info
        information = crawl_data(json_file)



    # Information list contains records in the following format:
    # information[i][j] is a dictionary containing data of j'th publication present in i'th page
    # For example, information[0][0] returns the first record present in first page

    # Call relevant function to create index and calculate tf-idf
    index = create_index(information)
    tfidf = calculate_tfidf(information, index)

    # For user query, get the ranked list of docs
    query = input("Enter your query: ")

    # Only the first 10 most relevant searches are displayed
    ranked_docs = rank_documents(query, information, index, tfidf)[:10]

    # Display relevant information of retrieved docs
    for doc_id in ranked_docs:
        doc_id_comp = 0
        for page in information:
            for record in page:
                doc_id_comp = doc_id_comp + 1
                if doc_id_comp == doc_id:
                    print("\nDocument ID: ", doc_id)
                    print("Title: ", record['title'])
                    print("Publication link: ", record['publication_link'])
                    print("Author: ", record['authors'])
                    print("Author's Profile: ", record['authors_profiles'])
                    print("Date: ", record['date'])
                    print("Journal: ", record['journal'])
                    print("Volume: ", record['volume'])
                    print("Article ID: ", record['article_id'])

if __name__ == "__main__":
    # Assuming json_file is defined somewhere above in your script
    json_file = "information.json"  # or your designated JSON file name

    # Check if the JSON file exists or not
    if os.path.exists(json_file) and os.path.isfile(json_file):
        # If exists, then just open the file and load all data to information list
        with open(json_file, "r") as f:
            information = json.load(f)
    else:
        # Else, perform web crawling and populate information
        information = crawl_data(json_file)

    # Create index and calculate tf-idf based on the crawled or loaded information
    index = create_index(information)
    tfidf = calculate_tfidf(information, index)

    # Call the CLI search engine function with the necessary parameters
    gui_search_engine(information, index, tfidf)