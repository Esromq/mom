import os
import openai
from openai import OpenAI
import logging
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.document_loaders import TextLoader
from langchain.vectorstores import Pinecone
from pinecone import Pinecone as PineconeClient, ServerlessSpec
import time
import json
from transformers import pipeline
import docx
import pyttsx3
from serpapi import GoogleSearch
import tensorflow as tf
from flask import Flask, request, jsonify
from datasets import load_dataset
import getpass
import hashlib
from textblob import TextBlob
import atexit


# Configure logging
logging.basicConfig(level=logging.INFO)

# Suppress TensorFlow informational messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

# Flask app initialization
app = Flask(__name__, static_url_path='/static')

# Load environment variables
load_dotenv()

# API keys and configurations
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")
SERP_API_KEY = os.getenv("SERP_API_KEY")

if not all([OPENAI_API_KEY, PINECONE_API_KEY, SERP_API_KEY, PINECONE_ENV]):
    raise ValueError("Missing one or more required API keys or environment variables.")

# Initialize OpenAI
client = OpenAI(api_key = OPENAI_API_KEY)
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
MODEL = "text-embedding-3-small"
user_name = "Creator"



# Define the directory where your files are located
directory_path = "C:/Users/esrom/Code Version 1/static/files"

# List all text files in the directory
file_paths = [os.path.join(directory_path, file) for file in os.listdir(directory_path) if file.endswith(".txt")]

# Load all documents
documents = []
for file_path in file_paths:
    loader = TextLoader(file_path)
    documents.extend(loader.load())
    


# Initialize Pinecone
pc = PineconeClient(api_key=PINECONE_API_KEY)
spec = ServerlessSpec(cloud="aws", region="us-east-1")
index_name = 'core-knowledge'

# Check if index already exists
if index_name not in pc.list_indexes().names():
    # Create index if not exists
    pc.create_index(
        index_name,
        dimension=1536,  # The dimension of the OpenAI embeddings
        metric='cosine',
        spec=spec
    )
    # Wait for index to be initialized
    while not pc.describe_index(index_name).status['ready']:
        time.sleep(1)
        
def embed_documents(self, texts):
    # Ensure all inputs are strings
    texts = [str(text) for text in texts if isinstance(text, (str, bytes))]
    return self._get_len_safe_embeddings(texts, engine=engine)

# Create embeddings for documents
def get_embeddings_from_documents(documents):
    embeddings = []
    for document in documents:
        doc_content = document("content", "")  # Ensure "content" exists
        if not isinstance(doc_content, str):
            print(f"Skipping invalid content: {doc_content}")
            continue  # Skip non-string content
        embedding = embeddings.embed_query(doc_content)
        embeddings.append(embedding)
    return embeddings


# Function to generate a valid Pinecone ID based on the document content or filename
def generate_valid_id(doc_content, max_length=512):
    # Generate a hash of the document content (you can also use filenames or other unique identifiers)
    doc_hash = hashlib.sha256(doc_content.encode('utf-8')).hexdigest()
    
    # Truncate the hash to ensure it's no longer than the maximum allowed length
    return doc_hash[:max_length]
# Embed documents and upsert to Pinecone
def upsert_documents_to_pinecone(documents, embeddings, index):
    embeddings_batch = get_embeddings_from_documents(documents)

    # Prepare metadata for upsert
    upsert_data = []
    for doc_content, embedding in zip(documents, embeddings_batch):
        valid_id = generate_valid_id(doc_content)  # Generate a valid ID for each document
        meta = {'text': doc_content}
        
        # Add the vector with the valid ID to the upsert data
        upsert_data.append((valid_id, embedding, meta))
    
    # Upsert to Pinecone
    index.upsert(vectors=list(upsert_data))
    print(f"Upserted {len(documents)} documents to Pinecone.")
    if len(documents) < 0:
        print(f"No Documents upserted to Pinecone.")

def chunk_documents(documents, chunk_size=500):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size)
    return [chunk for doc in documents for chunk in splitter.split_text(doc)]

# Cache for summaries
summary_cache = {}

# Load existing summaries from a JSON file (if using persistent storage)
def load_summary_cache(cache_file="summary_cache.json"):
    if os.path.exists(cache_file):
        with open(cache_file, "r") as f:
            return json.load(f)
    return {}

# Save summaries to a JSON file
def save_summary_cache(cache, cache_file="summary_cache.json"):
    with open(cache_file, "w") as f:
        json.dump(cache, f, indent=4)

# Initialize cache
summary_cache = load_summary_cache()

# Split long texts into chunks and summarize
def summarize_text_with_cache(doc_id, text, max_length=100):
    if doc_id in summary_cache:
        # Return cached summary if available
        return summary_cache[doc_id]
    
    # Set max_length dynamically if not provided
    max_length = max_length or min(len(text) // 2, 500)

    
    # Otherwise, summarize and cache the result
    chunk_size = 512  # Adjust chunk size as needed
    text_chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
    
    summaries = []
    for chunk in text_chunks:
        summary = summarizer(chunk, max_length=max_length, min_length=30, do_sample=False)
        summaries.append(summary[0]['summary_text'])
    
    combined_summary = " ".join(summaries)
    summary_cache[doc_id] = combined_summary  # Cache the summary
    return combined_summary

# Initialize the summarization pipeline
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Load .docx files from a folder
def load_documents(folder_path):
    loaded_docs = {}
    for filename in os.listdir(folder_path):
        if filename.endswith(".docx"):
            doc = docx.Document(os.path.join(folder_path, filename))
            content = "\n".join([para.text for para in doc.paragraphs if para.text])
            loaded_docs[filename] = content
    return loaded_docs

# Preprocess and cache summaries
folder_path = "./static/files"  # Define the folder path
doc = load_documents(folder_path)


# Load the dataset
dataset = load_dataset("cnn_dailymail", "3.0.0")    

# Initialize Pinecone and upsert documents
index = pc.Index(index_name)

# Test query to check Pinecone retrieval
def test_pinecone_query(index, query, embeddings):
    try:
        query_embedding = embeddings.embed_query(query)
        result = index.query(vector=query_embedding, top_k=5, include_metadata=True)
        if not result['matches']:
            logging.warning("No matches found for the query!")
        else:
            logging.info(f"Matches: {result['matches']}")
    except Exception as e:
        logging.error(f"Pinecone query failed: {e}")


# Define a function for querying Pinecone and returning relevant documents
def search_documents(query, index, embeddings, top_k=5):
    query_embedding = embeddings.embed_query(query)
    result = index.query(vector=query_embedding, top_k=top_k, include_metadata=True)
    return result['matches']

# Perform sentiment analysis on input
def analyze_sentiment(user_input):
    blob = TextBlob(user_input)
    return blob.sentiment.polarity

# Adjust the response tone based on sentiment
def adjust_response_tone(response, sentiment):
    if sentiment < -0.3:
        return f"I'm so sorry dear. {response}"
    elif sentiment > 0.3:
        return f"You are a blessing! {response}"
    return response

# AI chat function
def chat_with_mom(user_input, documents):
    matching_documents = search_documents(user_input, index, embeddings)
    
    # Combine matching documents into a single string
   # Combine relevant summaries or document content
    document_context = ""
    for match in matching_documents:
        doc_id = match['id']
        doc_text = match['metadata']['text']
        summary = summarize_text_with_cache(doc_id, doc_text)
        document_context += summary + "\n"
    
       # Perform sentiment analysis
    sentiment = analyze_sentiment(user_input)
    
    
    response = client.chat.completions.create(
    model="gpt-4",
    max_tokens=800,
    temperature=0.5,
    messages=[
        {"role": "system", "content": "You are Theresa, Esrom's mother, a wise and nurturing advisor, referencing her writings, through your son's (Esrom) quantum programing. You once said (I give honor to this moment. I appreciate that these words maybe an influence on someone. I wish Peace to the reader. I am at 'Peace' in my life and growing moment by moment. Thank you for being apart of my journey. I am becoming more aware of the love that brings us together. This is a 'Miracle'. be blessed!). You have written and have access to the following writings: {document_references}."},
        {"role": "user", "content": f"Context:\n{document_context}\n\nUser: {user_input}"},
        {"role": "system", "content": document_context}  # Added context from documents
    ],
    )
    
    response_text = response.choices[0].message.content

    logging.debug(f"User Input: {user_input}")
    logging.debug(f"Matching Documents: {matching_documents}")
    logging.debug(f"Document Context: {document_context}")
    logging.debug(f"AI Response: {response_text}")

    return adjust_response_tone(response_text, sentiment)

# Text-to-Speech function using pyttsx3
def text_to_speech(text):
    engine = pyttsx3.init()
    voices = engine.getProperty('voices')
    # Select a male voice
    for voice in voices:
        if 'female' in voice.name.lower():
            engine.setProperty('voice', voice.id)
            break
    engine.say(text)
    engine.runAndWait()

user_input = "Hey Momma!"
response_text = chat_with_mom(user_input, doc)
text = "Hi Son, this is Momma-AI, Also Known as Theresa-AI speaking!"
text_to_speech(response_text)


# Google search using SERP API
def search_google(query):
    params = {
        "engine": "google",
        "q": query,
        "location": "Atlanta, GA, United States",
        "hl": "en",
        "gl": "us",
        "google_domain": "google.com",
        "num": "10",
        "start": "10",
        "safe": "off",
        "api_key": SERP_API_KEY
    }

    search = GoogleSearch(params)
    results = search.get_dict()
    return results["organic_results"]

@app.route("/chat", methods=["POST"])
def chat():

    """
    Endpoint to interact with SeanAI via API.
    Expects JSON input with a "user_input" key.
    """
    data = request.json
    user_input = data.get("user_input", "")
    if not user_input:
        return jsonify({"error": "No input provided."}), 400

    # Respond to user input using all document content
    response = chat_with_mom(user_input, documents)
    return jsonify({"response": response})

if __name__ == "__main__":
    mode = input("Enter mode (flask/terminal): ").strip().lower()
    if mode == 'flask':
        app.run(host='0.0.0.0', port=5001)
    elif mode == 'terminal':
        print("You are now chatting with TheresaAI. Type 'exit' to quit.")
        while True:
            user_input = input("You: ")
            if user_input.lower() == "exit":
                print("Goodbye!")
                # Save cache on exit
                atexit.register(lambda: save_summary_cache(summary_cache))
                break
        
            # Simulate SeanAI's response
            response = chat_with_mom(user_input, documents)
            print(f"SeanAI: {response}")
            text_to_speech(response)  # Optional: Convert Theresa's response to speech

    else:
        print("Invalid mode. Exiting.")
        app.run(debug=False)

    

