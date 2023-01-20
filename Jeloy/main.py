import random
import json
import pickle
import numpy as np

import nltk
from nltk.stem import WordNetLemmatizer
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD

# Initialize WordNetLemmatizer to reduce words to their base form
lemmatizer = WordNetLemmatizer()

# Open intents.json and load intents
with open('intents.json') as json_file:
    intents = json.load(json_file)

# Initialize lists to store words, classes, and documents
words = []
classes = []
documents = []
# List of punctuation marks to ignore
ignore_letters = ['?', '!', '.', ',']

# Loop through intents and patterns
for intent in intents['intents']:
    for pattern in intent['patterns']:
        # Tokenize words in pattern
        word_list = nltk.word_tokenize(pattern)
        # Add words to words list
        words.extend(word_list)
        # Add document as tuple of tokenized words and intent tag
        documents.append((word_list, intent['tag']))
        # Add intent tag to classes list if it doesn't already exist
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# Lemmatize words and remove duplicates
words = [lemmatizer.lemmatize(word) for word in words if word not in ignore_letters]
words = sorted(set(words))
# Remove duplicates from classes list
classes = sorted(set(classes))

# Save words and classes lists to pickle files
pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

# Initialize training list and output empty list
training = []
output_empty = [0] * len(classes)

# Loop through documents to create bag of words and intent tag
for doc, tag in documents:
    # Create bag of words
    bag = [1 if word in doc else 0 for word in words]
    # Initialize list to store intent tag
    output_row = [0] * len(classes)
    # Set index of intent tag to 1
    output_row[classes.index(tag)] = 1
    # Append bag of words and intent tag to training list
    training.append([bag, output_row])

# Shuffle training data
random.shuffle(training)
# Split data into X and y arrays
train_x = np.array([train[0] for train in training])
train_y = np.array([train[1] for train in training])

# Initialize model
model = Sequential()
# Add first layer to model with 128 neurons, input shape matching train_x, and relu activation
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
# Add dropout layer with dropout rate of 0.5
model.add(Dropout(0.5))
# Add second layer to model with 64 neurons and relu activation
model.add(Dense(64, activation='relu'))
# Add dropout layer with dropout rate of 0.5
model.add(Dropout(0.5))
# Add output layer to model with number of neurons matching train_y and softmax activation
model.add(Dense(len(train_y[0]), activation='softmax'))

# Define learning rate schedule
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.01,
    decay_steps=100,
    decay_rate=0.96
)
# Initialize sgd optimizer with learning rate schedule, momentum of 0.9, and nesterov=True
sgd = SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
# Compile model with categorical cross-entropy loss, sgd optimizer, and metrics of accuracy
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Fit model on train_x and train_y for 200 epochs with batch size of 5
hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size = 5 , verbose = 1)
# Save model
model.save('chatbotmodel.h5')
# Save training history to pickle file
pickle.dump(hist.history, open('hist.pkl', 'wb'))

# Print message to indicate completion
print('Done')


#This code is building a neural network model with 3 layers, the input layer,
# hidden layer and output layer. The input layer has 128 neurons,
# the hidden layer has 64 neurons, and the output layer has the number of neurons equal to the
# number of classes. The model is trained using the categorical cross-entropy loss function, t
# he Stochastic Gradient Descent optimizer, and a learning rate schedule that decreases over time.
# The model is saved as 'chatbotmodel.h5' and the training history is saved in 'hist.pkl'
