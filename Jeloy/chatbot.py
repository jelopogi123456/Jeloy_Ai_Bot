import random
import json
import pickle
import numpy as np

import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model

# Initialize WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

# Open intents.json and load intents
with open('intents.json') as json_file:
    intents = json.load(json_file)

# Load words and classes lists from pickle files
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))

# Load trained model
model = load_model('chatbotmodel.h5')

def clean_up_sentence(sentence):
    # Tokenize sentence
    sentence_words = nltk.word_tokenize(sentence)
    # Lemmatize words and lowercase all words
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bow(sentence, words, show_details=False):
    """
        Create a bag of words from the sentence and compare it to the list of words
        :param sentence: Sentence to be processed
        :param words: List of words used to create bag of words
        :param show_details: Boolean to show details of found words in bag
        :return: Bag of words as a numpy array
        """
    # Tokenize sentence
    sentence_words = clean_up_sentence(sentence)
    # Bag of words
    bag = [0]*len(words)
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s:
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)
    return(np.array(bag))

def predict_class(sentence, model):
    """
        Use trained model to predict the class of the sentence
        :param sentence: Sentence to be classified
        :param model: Trained model used for prediction
        :return: List of dicts containing intent and probability
        """

    # Filter out predictions below a threshold
    p = bow(sentence, words,show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
    # Sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def getResponse(ints, intents_json):
    """
        Get a random response from the intents_json file based on the predicted intent
        :param ints: List of dicts containing intent and probability
        :param intents_json: Intents in json format
        :return: Random response from intents_json
        """
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if(i['tag']== tag):
            result = random.choice(i['responses'])
            break
    return result

#It takes the first item from the list of intents and probability,
# which has the highest probability, and returns a random response from the 'responses'
# key that corresponds to that intent in the intents.json file.

def chatbot_response(text):
    """
        Generate a response for the input text
        :param text: Input text
        :return: Chatbot's response
        """
    ints = predict_class(text, model)
    res = getResponse(ints, intents)
    return res

#This function  def chatbot_response takes in input text, passes it through the predict_class function
# to get the intent and probability, and then passes that to the getResponse function to get a response.

# Test chatbot
if model:
    print("Model Loaded Successfully")
else:
    print("Error Loading Model")


while True:
    user_input = input("What would you like to say to the chatbot? (Type 'exit' to quit) ")
    if user_input.lower() == "exit":
        break
    response = chatbot_response(user_input)
    print(response)
