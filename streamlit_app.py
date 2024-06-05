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

This app predict the ESG risk of company.
         
Data is collected from 30+ sites.
""")

company = st.text_input("Company", placeholder="PT Adaro Minerals Indonesia Tbk")

list_of_sites = [
    ('eco-business.com', 'en'),
    ('thejakartapost.com', 'en'),
    ('marketforces.org.au', 'en'),
    ('jakartaglobe.id', 'en'),
    ('tempo.co', 'id'),
    ('greenpeace.org', 'id'),
    ('enternusantara.org', 'id'),
    ('mongabay.co.id', 'id'),
    ('ecoton.or.id', 'id'),
    ('betahita.id', 'id'),
    ('cnnindonesia.com', 'id'),
    ('cnbcindonesia.com', 'id'),
    ('republika.id', 'id'),
    ('kontan.co.id', 'id'),
    ('tambang.co.id', 'id'),
    ('investortrust.id', 'id'),
    ('investor.id', 'id'),
    ('indopos.co.id', 'id'),
    ('katadata.co.id', 'id'),
    ('beritasatu.com', 'id'),
    ('kompasiana.com', 'id'),
    ('wartaekonomi.co.id', 'id'),
    ('antaranews.com', 'id'),
    ('kompas.id', 'id'),
    ('republika.co.id', 'id'),
    ('bisnis.com', 'id'),
    ('medcom.id', 'id'),
    ('emitennews.com', 'id'),
    ('sudutpandang.id', 'id'),
    ('bisnisnews.id', 'id'),
    # ('infobanknews.com', 'id'),
    # ('topbusiness.id', 'id'),
    # ('detik.com', 'id'),
    # ('viva.co.id', 'id'),
    # ('sinergipos.com', 'id'),
    # ('rakyatjelata.com', 'id'),
    # ('baznas.go.id', 'id'),
    # ('majalahlintas.com', 'id'),
]

GOOGLE = 'https://www.google.com/search'
headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Cafari/537.36'}

proxies = [
    '212.102.34.55:8443',
    '212.102.34.56:8443',
    '85.209.154.148:33333',
    '212.102.34.53:8443',
    '178.48.68.61:18080',
    '72.44.32.113:3128',
    '52.55.17.104:6000',
    '212.102.34.52:8443',
    '172.183.241.1:8080',
    '41.65.251.47:1976',
    '212.102.34.54:8443',
    '41.65.0.208:1981',
    '35.185.196.38:3128',
    '23.152.40.14:3128',
    '119.39.109.233:3128',
    '221.195.167.200:9999',
    '94.102.234.186:32650',
    '159.69.221.82:3128',
    '199.167.236.12:3128',
    '119.12.170.184:80',
    '24.112.3.220:8080',
    '112.106.145.190:80',
    '201.71.3.42:999',
    '220.77.233.228:3128',
    '34.92.250.88:11111',
    '34.92.250.88:10000',
    '72.10.160.174:2645',
    '59.157.2.63:32304',
    '3.10.93.50:1080',
    '119.2.43.143:8080',
    '41.65.103.29:1981',
    '185.243.114.105:3128',
    '111.53.212.6:80',
    '45.12.214.202:3128',
    '139.167.54.114:8081',
    '43.228.85.83:3128',
    '45.225.89.145:999',
    '176.33.141.130:3128',
    '176.40.40.63:3310',
    '172.96.117.205:38001',
    '8.219.97.248:80',
    '101.42.157.101:6666',
    '152.99.145.45:80',
    '88.247.17.91:1459',
    '194.31.108.52:55555',
    '72.10.164.178:1687',
    '202.93.244.38:8080',
    '88.87.78.95:8080',
    '88.247.187.212:1459',
    '203.74.125.18:8888',
    '114.132.202.246:8080',
    '103.48.71.18:83',
    '78.186.10.92:3310',
    '182.151.17.172:3128',
    '91.187.113.68:8080',
    '67.43.227.228:29509',
    '72.10.160.90:24465',
    '43.133.59.220:3128',
    '189.240.60.171:9090',
    '189.240.60.164:9090',
    '72.10.164.178:16425',
    '67.43.227.227:2303',
    '189.240.60.163:9090',
    '189.240.60.168:9090',
    '89.237.33.225:37647',
    '20.37.207.8:8080',
    '72.10.164.178:13323',
    '189.240.60.169:9090',
    '152.99.145.25:80',
    '72.10.160.90:11145',
    '72.10.160.90:12365',
    '67.213.212.54:41388',
    '41.128.183.10:1981',
    '72.10.160.171:8785',
    '113.21.238.40:8800',
    '67.43.228.252:4083',
    '161.34.39.147:3128',
    '161.34.39.156:3128',
    '160.248.91.68:3128',
    '160.248.9.69:3128',
    '155.93.96.210:8080',
    '160.248.8.58:3128',
    '67.43.228.253:2799',
    '67.43.227.227:19603',
    '161.34.39.154:3128',
    '160.248.9.74:3128',
    '160.248.9.73:3128',
    '72.10.160.170:1255',
    '49.88.191.163:8888',
    '187.216.229.150:8080',
    '72.10.164.178:31033',
    '221.168.33.155:8080',
    '103.116.82.135:8080',
    '34.215.74.117:3128',
    '185.73.103.23:3128',
    '103.127.220.98:8090',
    '185.126.183.120:8080',
    '61.9.34.146:1',
    '161.34.39.151:3128',
    '152.70.235.185:9002',
    '103.68.214.97:8080',
    '185.191.236.162:3128',
    '181.78.105.156:999',
    '202.5.60.46:5020',
    '67.43.236.20:27735',
    '190.111.209.207:3128',
    '103.165.128.171:8080',
    '67.43.228.250:3289',
    '200.106.184.14:999',
    '78.186.183.5:3310',
    '84.54.185.203:8080',
    '72.10.160.90:24845',
    '177.73.136.29:8080',
    '41.159.154.43:3128',
    '67.43.228.253:31703',
    '182.160.120.228:5020',
    '67.43.228.254:6363',
    '178.153.8.211:8080',
    '190.61.46.228:999',
    '203.205.9.105:8080',
    '103.111.207.138:80',
    '118.70.139.8:3128',
    '103.165.155.195:2016',
    '67.43.228.251:18371',
    '176.194.189.40:80',
    '38.156.72.57:8080',
    '168.228.51.84:999',
    '191.37.208.201:8080',
    '147.139.140.74:80',
    '103.49.28.23:12113',
    '85.117.63.37:8080',
    '161.34.68.229:8888',
    '160.248.90.229:3128',
    '77.237.28.191:8080',
    '111.89.130.105:3128',
    '41.65.174.98:1981',
    '160.248.9.70:3128',
    '161.34.39.155:3128',
    '161.34.39.153:3128',
    '103.156.15.26:8080',
    '31.45.237.146:8080',
    '103.173.138.171:8080',
    '72.10.160.172:16879',
    '14.140.167.189:10176',
    '38.123.79.3:999',
    '72.10.160.93:2477',
    '191.102.254.28:8081',
    '154.127.240.120:64004',
    '45.191.46.210:999',
    '178.218.44.79:3128',
    '102.68.129.54:8080',
    '103.132.52.181:8080',
    '217.199.151.94:83',
    '188.132.222.6:8080',
    '103.176.96.140:8082',
    '50.192.195.69:52018',
    '36.77.35.39:8080',
    '179.1.192.5:999',
    '65.1.244.232:1080',
    '103.173.230.88:8080',
    '66.186.199.16:8080',
    '189.240.60.166:9090',
    '114.132.202.125:8080',
    '197.246.10.149:8080',
    '103.105.76.214:9090',
    '66.210.33.34:8080',
    '38.156.191.228:999',
    '209.45.40.33:999',
    '114.132.202.80:8080',
    '41.33.219.130:1981',
    '113.125.82.11:3128',
    '195.250.92.58:8080',
    '177.55.247.174:8080',
    '103.153.246.142:8181',
    '38.156.72.135:8080',
    '103.181.25.158:8080',
    '45.176.97.90:999',
    '103.155.62.163:8080',
    '200.106.184.21:999',
    '201.71.3.52:999',
    '190.128.225.117:999',
    '140.227.204.70:3128',
    '160.248.0.115:3128',
    '160.248.91.69:3128',
    '168.138.211.5:8080',
    '77.41.146.13:8080',
    '41.128.91.186:1981',
    '124.158.153.218:8180',
    '176.213.141.107:8080',
    '180.74.171.206:8080',
    '41.65.0.208:1976',
    '23.131.184.76:3128',
    '38.50.165.56:999',
    '122.2.48.121:8080',
    '103.203.174.182:84',
    '12.163.95.161:8080',
    '177.234.210.56:999',
    '161.132.125.244:8080',
    '185.111.156.170:80',
    '122.3.103.17:8082',
    '103.134.223.139:8181',
    '45.190.170.254:999',
    '103.203.175.33:84',
    '179.42.72.186:85',
    '164.163.42.27:10000',
    '202.154.18.145:8085',
    '103.214.219.23:8080',
    '103.115.227.198:8071',
    '79.106.33.26:8079',
    '47.108.191.13:8080',
    '77.137.21.78:19000',
    '190.94.212.216:999',
    '190.110.34.245:999',
    '103.13.204.24:8082',
    '109.195.98.207:80',
    '103.146.185.90:8080',
    '181.78.27.34:999',
    '156.200.116.69:1981',
    '85.209.153.175:4153',
    '177.75.96.18:3128',
    '187.221.196.175:3128',
    '170.79.36.60:8081',
    '95.0.168.62:1976',
    '45.5.92.94:8137',
    '83.151.4.172:57812',
    '36.64.195.242:8080',
    '36.91.15.241:8080',
    '176.235.139.22:10001',
    '103.203.175.49:84',
    '202.62.67.209:53281',
    '103.84.177.26:8083',
    '38.156.73.62:8080',
    '217.52.247.78:1976',
    '72.10.160.90:5673',
    '181.209.122.74:999',
    '91.93.42.119:10001',
    '201.174.224.174:999',
    '160.248.9.94:3128',
    '190.2.211.146:999',
    '111.89.130.104:3128',
    '37.195.222.7:52815',
    '175.178.77.128:3128',
    '140.99.122.244:999',
    '38.159.232.6:8080',
    '80.80.163.190:46276',
    '190.102.139.152:999',
    '103.76.201.110:8080',
    '161.34.39.152:3128',
    '103.146.170.193:83',
    '216.87.69.230:8383',
    '103.189.254.245:1111',
    '114.30.92.2:8989',
    '79.143.177.29:21972',
    '202.12.80.8:83',
    '202.12.80.7:83',
    '45.70.221.22:18080',
    '157.100.7.146:999',
    '179.60.129.195:8080',
    '5.187.79.198:44331',
    '196.219.202.74:8080',
    '103.156.141.87:8080',
    '91.235.75.33:8282',
    '188.168.28.37:81',
    '223.27.144.34:8080',
]

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

        if not article.cleaned_text:
            article_content = soup.find('div', class_='article-content')
            if article_content:
                news_text.append(article_content.get_text())
        else:
            news_text.append(article.cleaned_text)

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

    st.dataframe(df)

