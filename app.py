from flask import Flask, render_template, request
import torch
import nltk
import nltk
nltk.download('punkt_tab')
from transformers import BertForSequenceClassification, BertTokenizer
from nltk.corpus import stopwords 
# nltk.download('stopwords')
# nltk.download('punkt')
from textblob import TextBlob
from nltk.tokenize import word_tokenize, sent_tokenize 


app = Flask(__name__, template_folder='template')

# Load the BERT model and tokenizer
model_directory = 'bert-base-uncased'
model = BertForSequenceClassification.from_pretrained(model_directory)
tokenizer = BertTokenizer.from_pretrained(model_directory)


@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        comment = request.form["comment"]
        text = comment
        blob = TextBlob(text)
        sentiment = blob.sentiment.polarity
        # Tokenizing the text 
        stopWords = set(stopwords.words("english")) 
        words = word_tokenize(text) 
        
        # Creating a frequency table to keep the  
        # score of each word 
        
        freqTable = dict() 
        for word in words: 
            word = word.lower() 
            if word in stopWords: 
                continue
            if word in freqTable: 
                freqTable[word] += 1
            else: 
                freqTable[word] = 1
        
        # Creating a dictionary to keep the score 
        # of each sentence 
        sentences = sent_tokenize(text) 
        sentenceValue = dict() 
        
        for sentence in sentences: 
            for word, freq in freqTable.items(): 
                if word in sentence.lower(): 
                    if sentence in sentenceValue: 
                        sentenceValue[sentence] += freq 
                    else: 
                        sentenceValue[sentence] = freq 
        
        
        
        sumValues = 0
        for sentence in sentenceValue: 
            sumValues += sentenceValue[sentence] 
        
        # Average value of a sentence from the original text 
        
        average = int(sumValues / len(sentenceValue)) 
        
        # Storing sentences into our summary. 
        summary = '' 
        for sentence in sentences: 
            if (sentence in sentenceValue) and (sentenceValue[sentence] > (1.2 * average)): 
                summary += " " + sentence 



        # Truncate the input text to a maximum length of 512 tokens
        text = comment[:512]

        # Tokenize the text
        inputs = tokenizer(text, return_tensors='pt')

        # Make predictions using the loaded model
        with torch.no_grad():
            outputs = model(**inputs)

        # Get the predicted class
        predicted_class = torch.argmax(outputs.logits, dim=1).item()
        verdict = "Real" if predicted_class == 1 else "Fake"
        senti = "Positive" if sentiment > 0 else "Negative"
    return render_template('result.html', prediction=verdict, summary = summary, sentiment= senti)

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0')
