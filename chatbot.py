from chatterbot import ChatBot
from chatterbot.trainers import ListTrainer
from langdetect import detect
from googletrans import Translator
from textblob import TextBlob
import nltk

# Initialize necessary components
nltk.download('punkt')
translator = Translator()
chatbot = ChatBot('SportsBetBot', preprocessors=['chatterbot.preprocessors.clean_whitespace'])
trainer = ListTrainer(chatbot)

# Train chatbot with sample phrases in both English and Italian
trainer.train([
    "What’s the prediction for the next game?",
    "Our model suggests a high probability of Team A winning.",
    "Qual è la previsione per la prossima partita?",
    "Il nostro modello suggerisce una probabilità alta di vittoria per la Squadra A."
])

# Language detection and translation
def detect_language(text):
    try:
        return detect(text)
    except:
        return "en"  # Default to English if detection fails

# Preprocess and translate message
def preprocess_message(message, lang):
    if lang == 'it':
        message = translator.translate(message, src='it', dest='en').text
    return message

# Enhanced response with translation back to user's language
def get_chatbot_response(message):
    user_lang = detect_language(message)
    processed_message = preprocess_message(message, user_lang)
    
    # Sentiment analysis for a professional tone
    sentiment = TextBlob(processed_message).sentiment.polarity
    response = chatbot.get_response(processed_message)
    
    if sentiment < -0.1:
        # Respond empathetically if sentiment is negative
        response_text = "I'm here to help. Let's find the best solution."
        if user_lang == 'it':
            response_text = "Sono qui per aiutarti. Troviamo insieme la soluzione migliore."
    else:
        response_text = str(response)
    
    # Translate back if needed
    if user_lang == 'it':
        response_text = translator.translate(response_text, src='en', dest='it').text
    
    return response_text
