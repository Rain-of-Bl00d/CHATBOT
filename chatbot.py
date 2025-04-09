import numpy as np
import random
import nltk
import json
import os
import gradio as gr
import traceback
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from pymongo import MongoClient

nltk.download('punkt_tab')  # Tokenizes text into words (it is showing error for many cases! as insecure or some security reassion)
nltk.download('wordnet')  # Lexical database for lemmatization

lemmatizer = nltk.WordNetLemmatizer()

DATASET_FILE = "chatbot_data.json"

def load_dataset():
    if os.path.exists(DATASET_FILE):
        with open(DATASET_FILE, "r") as file:
            return json.load(file)
    return {"questions": [], "answers": []}

dataset = load_dataset()

if not dataset["questions"]:
    dataset["questions"] = [
        "hello", 
        "hi", 
        "how are you", 
        "what is your name", 
        "what can you do", 
        "who created you", 
        "tell me a joke",
        "what is AR", 
        "what is VR", 
        "difference between AR and VR", 
        "how does AR work", 
        "how does VR work",
        "best AR devices", 
        "best VR headsets", 
        "uses of AR", 
        "uses of VR", 
        "what is the future of AR/VR",
        "recommend an AR app", 
        "recommend a VR game", 
        "best AR frameworks", 
        "best VR frameworks", 
        "goodbye",
        "thank you"
    ]
    dataset["answers"] =  [
        "Hello! How can I help you?", 
        "Hi there! What's on your mind?", 
        "I'm just a bot, but I'm doing great! How about you?", 
        "I am an AI chatbot built to assist you with AR and VR information.", 
        "I can answer your questions about AR, VR, and general topics!", 
        "I was created by a developer to help with AR/VR-related queries.", 
        "Why don't skeletons fight each other? Because they don't have the guts!",
        "AR (Augmented Reality) overlays digital content onto the real world.", 
        "VR (Virtual Reality) immerses users in a completely virtual environment.", 
        "AR adds digital elements to real-world views, while VR replaces the real world entirely.", 
        "AR works by using cameras and sensors to map real environments and overlay digital objects.", 
        "VR works by using headsets that create a fully digital world for the user.", 
        "Some of the best AR devices are Microsoft HoloLens, Magic Leap, and AR-supported smartphones.", 
        "Top VR headsets include Oculus Quest, HTC Vive, and PlayStation VR.", 
        "AR is used in gaming, retail, education, healthcare, and navigation.", 
        "VR is used in gaming, training, simulations, and virtual tourism.", 
        "AR and VR are rapidly evolving, with applications in gaming, business, and the metaverse.", 
        "A great AR app is Pokémon GO, which blends digital objects with the real world.", 
        "A popular VR game is Half-Life: Alyx, offering a fully immersive experience.", 
        "Some of the best AR frameworks are Google ARCore, Apple ARKit, and Vuforia.", 
        "Some of the best VR frameworks include OpenVR, SteamVR, and Oculus SDK.", 
        "Goodbye! Have a great day!", 
        "You're welcome! Let me know if you need more help!"
    ]
    
# # Replace with your own URI or use localhost for local MongoDB
# MONGO_URI = "mongodb://localhost:27017"  # or your Atlas URI
# client = MongoClient(MONGO_URI)

# # Create/use a database
# db = client["chatbot_db"]

# # Create/use a collection
# chat_collection = db["chat_logs"]
# qa_collection = db["qa_pairs"]

def clean_text(text):
    tokens = nltk.word_tokenize(text.lower())
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

cleaned_questions = [clean_text(q) for q in dataset["questions"]]

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(cleaned_questions)
y = dataset["answers"]

model = LogisticRegression()
model.fit(X, y)

def chatbot_response(user_input):
    cleaned_input = clean_text(user_input)
    user_vector = vectorizer.transform([cleaned_input])
    prediction = model.predict(user_vector)[0]
    return prediction

def debug_code(code):
    try:
        exec(code, {})  # Execute code in an empty dictionary
        return "No errors found. The code executed successfully."
    except SyntaxError as e:
        return f"Syntax Error: {str(e).split('(')[0].strip()}"
    except IndentationError as e:
        return f"Indentation Error: {str(e).split('(')[0].strip()}"
    except NameError as e:
        return f"Name Error: {str(e).split('(')[0].strip()}"
    except TypeError as e:
        return f"Type Error: {str(e).split('(')[0].strip()}"
    except Exception as e:
        return f"{type(e).__name__}: {str(e).split('(')[0].strip()}"

def update_chatbot(user_question, user_answer):
    global cleaned_questions, vectorizer, model
    cleaned_question = clean_text(user_question)
    if cleaned_question not in cleaned_questions:
        dataset["questions"].append(user_question)
        dataset["answers"].append(user_answer)
        with open(DATASET_FILE, "w") as file:
            json.dump(dataset, file)
        cleaned_questions = [clean_text(q) for q in dataset["questions"]]
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(cleaned_questions)
        model = LogisticRegression()
        model.fit(X, dataset["answers"])
        return "Chatbot updated successfully!"
    else:
        return "This question already exists in my database!"

chat_interface = gr.Interface(
    fn=chatbot_response,
    inputs=gr.Textbox(label="Ask me anything"),
    outputs="text",
    title="Smart Chatbot",
    description="Type a message and I'll try to respond!"
)

debug_interface = gr.Interface(
    fn=debug_code,
    inputs=gr.Textbox(label="Enter Python code to debug"),
    outputs="text",
    title="Code Debugger",
    description="Paste your Python code here, and I'll attempt to debug it."
)

update_interface = gr.Interface(
    fn=update_chatbot,
    inputs=[gr.Textbox(label="New Question"), gr.Textbox(label="Correct Answer")],
    outputs="text",
    title="Train the Chatbot",
    description="Teach me new questions and answers!"
)

gr.TabbedInterface([chat_interface, debug_interface, update_interface], ["Chat", "Debug Code", "Train"]).launch()
