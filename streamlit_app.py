import streamlit as st
import re
import requests
from newspaper import Article
from newspaper import Config
import preprocessor as p
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
import torch.nn.functional as F
from goose3 import Goose
from goose3.configuration import Configuration  
from bs4 import BeautifulSoup

st.write("""
# ESG Prediction App

This is a Proof of Concept for a company ESG (Environmental, Social, and Governance) risk prediction application.
""")

company = st.text_input("Company", placeholder="PT Adaro Minerals Indonesia Tbk")

GOOGLE = 'https://www.google.com/search'
headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Cafari/537.36'}

API_KEY = 'AIzaSyDCfIltnvAQ3lvpovRXydRMhGQ-VxkboQ4'
SEARCH_ENGINE_ID = 'e586ee8a6c7e64d7b'

from googleapiclient.discovery import build
import math

def google_search(search_term, api_key, cse_id, **kwargs):
    service = build("customsearch", "v1", developerKey=api_key)
    
    num_search_results = kwargs['num']
    if num_search_results > 100:
        raise NotImplementedError('Google Custom Search API supports max of 100 results')
    elif num_search_results > 10:
        kwargs['num'] = 10 # this cannot be > 10 in API call 
        calls_to_make = math.ceil(num_search_results / 10)
    else:
        calls_to_make = 1
        
    kwargs['start'] = start_item = 1
    items_to_return = []
    while calls_to_make > 0:
        res = service.cse().list(q=search_term, cx=cse_id, **kwargs).execute()
        items_to_return.extend(res['items'])
        calls_to_make -= 1
        start_item += 10
        kwargs['start'] = start_item
        leftover = num_search_results - start_item + 1
        if 0 < leftover < 10:
            kwargs['num'] = leftover
        
    return items_to_return 

if company:
    links = []
    news_text = []
    
    query = f'{company} after:2023-01-01'
    response = google_search(query, API_KEY, SEARCH_ENGINE_ID, num=100)

    url_collection = [item['link'] for item in response]
    import os
    os.environ['ST_USER_AGENT'] = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36'

    user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36'
    config = Config()
    config.browser_user_agent = user_agent
    config.request_timeout = 60
    config.fetch_images = False
    config.memoize_articles = True
    config.language = 'id'

    # p.set_options(p.OPT.MENTION, p.OPT.EMOJI, p.OPT.HASHTAG, p.OPT.RESERVED, p.OPT.SMILEY, p.OPT.URL)

    def cleaner(text):
        text = re.sub("@[A-Za-z0-9]+", "", text) #Remove @ sign
        text = text.replace("#", "").replace("_", "") #Remove hashtag sign but keep the text
        # text = p.clean(text) # Clean text from any mention, emoji, hashtag, reserve words(such as FAV, RT), smiley, and url
        text = text.strip().replace("\n","")
        return text
    
    for url in url_collection:
        if "http" not in url:
            continue
        lang = "id"
        if "eco-business.com" in url or "thejakartapost.com" in url or "marketforces.org.au" in url or "jakartaglobe.id" in url:
            lang = "en"

        ### Selenium
        # from selenium import webdriver
        # from selenium.webdriver.chrome.options import Options
        # from goose3 import Goose

        # options = Options()
        # options.headless = True
        # options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")

        # driver = webdriver.Chrome(options=options)
        # # url = 'https://example.com/news-article'
        # driver.get(url)

        # html = driver.page_source
        # driver.quit()

        # g = Goose()
        # article = g.extract(raw_html=html)

        # print(article.cleaned_text)
        # news_text.append(article.cleaned_text)
        ###

        # article = Article(url, language=lang, config=config)
        # article.download()
        # article.parse()
        # article_clean = cleaner(article.text)

        # url = 'https://example.com/news-article'
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}

        response = requests.get(url, headers=headers)
        # html = response.text

        soup = BeautifulSoup(response.content, 'html.parser')

        g = Goose()
        article = g.extract(raw_html=str(soup))

        # print(url)
        # print(soup)
        news_empty = True

        possible_class = ['detail', 'body-content', 'article-content', 'detail-konten', 'DetailBlock']
        excluded_sentence = ['Komentar menjadi tanggung-jawab Anda sesuai UU ITE', 'Dapatkan berita terbaru dari kami Ikuti langkah ini untuk mendapatkan notifikasi:']

        if not article.cleaned_text:
            article_content = soup.find('div', class_=possible_class)
            if article_content and article_content.get_text() not in excluded_sentence:
                news_text.append(article_content.get_text())
                news_empty = False
                print(f'{url} News Exist using POSSIBLE CLASS')
        else:
            if article.cleaned_text not in excluded_sentence:
                news_text.append(article.cleaned_text)
                news_empty = False
                print(f'{url} News Exist using ARTICLE CLEANED TEXT')

        if news_empty:
            print(f'Cannot Get URL: {url}')
            # print(soup)

        # print(article.cleaned_text)

        

        # goose = Goose()
        # config = Configuration()
        # config.strict = False  # turn of strict exception handling
        # config.browser_user_agent = 'Mozilla 5.0'  # set the browser agent string
        # config.http_timeout = 5.05  # set http timeout in seconds

        # with Goose(config) as g:
        #     article = goose.extract(url=url)

        #     news_text.append(article.cleaned_text)

    df = pd.DataFrame({
        'news': news_text
        })
    
    # Load the tokenizer and model
    tokenizer_esg = AutoTokenizer.from_pretrained("didev007/ESG-indobert-model")
    model_esg = AutoModelForSequenceClassification.from_pretrained("didev007/ESG-indobert-model")

    # Load the tokenizer and model
    tokenizer_sentiment = AutoTokenizer.from_pretrained("adhityaprimandhika/distillbert_sentiment_analysis")
    model_sentiment = AutoModelForSequenceClassification.from_pretrained("adhityaprimandhika/distillbert_sentiment_analysis")

    def get_chunk_weights(num_chunks):
        center = num_chunks / 2
        sigma = num_chunks / 4
        weights = [np.exp(-0.5 * ((i - center) / sigma) ** 2) for i in range(num_chunks)]
        weights = np.array(weights)
        return weights / weights.sum()

    def tokenize_and_chunk(text, tokenizer, chunk_size=512):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        input_ids = inputs['input_ids'][0]
        
        chunks = [input_ids[i:i+chunk_size] for i in range(0, len(input_ids), chunk_size)]
        return chunks

    def esg_category(chunks, model):
        num_chunks = len(chunks)
        weights = get_chunk_weights(num_chunks)
        
        esg_scores = np.zeros(4)
        labels = ["none", "E", "S", "G"]
        
        for i, chunk in enumerate(chunks):
            inputs = {'input_ids': chunk.unsqueeze(0)}
            outputs = model(**inputs)
            logits = outputs.logits
            probs = F.softmax(logits, dim=1).detach().numpy()[0]
            esg_scores += weights[i] * probs

        predicted_class = esg_scores.argmax()
        aggregated_esg = labels[predicted_class]
        
        return aggregated_esg

    def sentiment_analysis(text, tokenizer, model):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=1).item()
        labels = ["positive", "neutral", "negative"]
        predicted_sentiment = labels[predicted_class]
        return predicted_sentiment

    def apply_model_to_dataframe(df, tokenizer_esg, model_esg, tokenizer_sentiment, model_sentiment, text_column='news'):
        esg_categories = []
        sentiments = []
        for text in df[text_column]:
            if isinstance(text, str): 
                chunks = tokenize_and_chunk(text, tokenizer_esg)
                esg = esg_category(chunks, model_esg)
                sentiment = sentiment_analysis(text, tokenizer_sentiment, model_sentiment)
                esg_categories.append(esg)
                sentiments.append(sentiment)
            else:
                esg_categories.append("none") 
                sentiments.append("neutral") 
        
        df['aggregated_esg'] = esg_categories
        df['sentiment'] = sentiments
        return df
    
    result_data = apply_model_to_dataframe(df, tokenizer_esg, model_esg, tokenizer_sentiment, model_sentiment)

    grouped_counts = df.groupby(['aggregated_esg', 'sentiment']).size().reset_index(name='count')
    data = grouped_counts.pivot(index='aggregated_esg', columns='sentiment', values='count')
    required_columns_sentiment = ['negative', 'positive', 'neutral']
    for col in required_columns_sentiment:
        if col not in data.columns:
            data[col] = 0

    # Handle potential missing values
    data['negative'] = data['negative'].fillna(0)
    data['positive'] = data['positive'].fillna(0)
    data['neutral'] = data['neutral'].fillna(0)

    # print(data)
    
    data['count'] = (data['negative']+data['positive']+data['neutral'])
    data['total'] = data['negative']/data['count'] + data['positive']*(-0.2)/data['count']
    # data['total'] = data['negative'] + data['positive']*(-1)
    if 'none' in data:
        data = data.drop('none')
    # data

    total = data['total'].sum()

    # Min-max normalization
    min_esg = -1
    max_esg = 2
    min_score = 0
    max_score = 60

    ESG_score = ((total - min_esg) / (max_esg - min_esg)) * (max_score - min_score) + min_score
    
    def esg_risk_categorization(esg_score):
        if esg_score <= 10:
            return 'Negligible'
        elif 10 < esg_score <= 20:
            return 'Low'
        elif 20 < esg_score <= 30:
            return 'Medium'
        elif 30 < esg_score <= 40:
            return 'High'
        else:
            return 'Severe'
        
    risk = esg_risk_categorization(ESG_score)

    # st.dataframe(df)

    st.write(company)
    st.write(f'ESG Score Prediction: {ESG_score}')
    st.write(f'ESG Category Risk Prediction: {risk}')

