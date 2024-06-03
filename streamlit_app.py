import streamlit as st
import requests
import re
from bs4 import BeautifulSoup as BS
import urllib.parse
from newspaper import Article
from newspaper import Config
import preprocessor as p
import pandas as pd

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
    ('tempo.co', 'id'),
    ('greenpeace.org', 'id'),
    ('enternusantara.org', 'id'),
    ('mongabay.co.id', 'id'),
    ('ecoton.or.id', 'id'),
    ('betahita.id', 'id'),
    ('cnnindonesia.com', 'id'),
    ('cnbcindonesia.com', 'id'),
    ('detik.com', 'id'),
    ('republika.id', 'id'),
    ('kontan.co.id', 'id'),
    ('tambang.co.id', 'id'),
    ('topbusiness.id', 'id'),
    ('viva.co.id', 'id'),
    ('investortrust.id', 'id'),
    ('investor.id', 'id'),
    ('indopos.co.id', 'id'),
    ('katadata.co.id', 'id'),
    ('beritasatu.com', 'id'),
    ('kompasiana.com', 'id'),
    ('wartaekonomi.co.id', 'id'),
    ('sinergipos.com', 'id'),
    ('antaranews.com', 'id'),
    ('kompas.id', 'id'),
    ('republika.co.id', 'id'),
    ('infobanknews.com', 'id'),
    ('bisnis.com', 'id'),
    ('medcom.id', 'id'),
    ('rakyatjelata.com', 'id'),
    ('baznas.go.id', 'id'),
    ('jakartaglobe.id', 'id'),
    ('emitennews.com', 'id'),
    ('sudutpandang.id', 'id'),
    ('majalahlintas.com', 'id'),
    ('bisnisnews.id', 'id'),
]

GOOGLE = 'https://www.google.com/search'
headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Cafari/537.36'}

if company:
    links = []
    url_collection = []
    news_text = []
    def get_query(site, company) -> str:
        return 'site:{} after:2023-01-01 {}'.format(site, company)
    
    for site in list_of_sites:
        query = get_query(site[0], company)

        params = {
            'q': query,
            'num': 1, # max number of search results per execution
        }

        response = requests.get(GOOGLE, params=params, headers=headers)
        response.raise_for_status()
        soup = BS(response.text, 'lxml')
        links += soup.find_all("a")[16:]

    for link in links:
        str_link = str(link)
        if str_link.find('/url?q=') == -1 or 'accounts.google.com' in str_link or 'support.google.com' in str_link:
            continue
        url = str_link[str_link.find('/url?q='):str_link.find('>')]
        url_parsed = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)['q'][0]
        url_collection.append(url_parsed)

    user_agent = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (HTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36'
    config = Config()
    config.browser_user_agent = user_agent
    config.request_timeout = 60
    config.fetch_images = False
    config.memoize_articles = True
    config._language = 'id'

    p.set_options(p.OPT.MENTION, p.OPT.EMOJI, p.OPT.HASHTAG, p.OPT.RESERVED, p.OPT.SMILEY, p.OPT.URL)

    def cleaner(text):
        text = re.sub("@[A-Za-z0-9]+", "", text) #Remove @ sign
        text = text.replace("#", "").replace("_", "") #Remove hashtag sign but keep the text
        text = p.clean(text) # Clean text from any mention, emoji, hashtag, reserve words(such as FAV, RT), smiley, and url
        text = text.strip().replace("\n","")
        return text
    
    for url in url_collection:
        if "http" not in url:
            continue
        lang = "id"
        if "eco-business.com" in url or "thejakartapost.com" in url or "marketforces.org.au" in url:
            lang = "en"
        article = Article(url, language=lang, config=config)
        article.download()
        article.parse()
        article_clean = cleaner(article.text)
        news_text.append(article_clean[:1000])

    df = pd.DataFrame({
        'news': news_text
        })
    # st.dataframe(df)

    # tested in transformers==4.18.0 
    from transformers import BertTokenizer, BertForSequenceClassification, pipeline

    finbert = BertForSequenceClassification.from_pretrained('yiyanghkust/finbert-esg',num_labels=4)
    tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-esg')
    nlp = pipeline("text-classification", model=finbert, tokenizer=tokenizer)
    # results = nlp('Rhonda has been volunteering for several years for a variety of charitable community programs.')
    # print(results) # [{'label': 'Social', 'score': 0.9906041026115417}]

    sentiment_classifier = pipeline(
        model="lxyuan/distilbert-base-multilingual-cased-sentiments-student", 
        return_all_scores=True
    )

    def esg_categorization(row):
        return nlp(row['news'])
    
    def sentiment_analysis(row):
        return sentiment_classifier(row['news'])
    
    df['esg_category'] = df.apply(esg_categorization, axis=1)

    df['sentiment'] = df.apply(sentiment_analysis, axis=1)

    st.dataframe(df)

