# import libraries

from selenium import webdriver
from tqdm import tqdm
import os
import pandas as pd
import time

# setting

BASE = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE + '\setting')

PATH = open('selenium_path.txt', 'r').read()
URL_PAGE = str(input('input youtube link videos: '))
Total_comment = int(input('input Total_comment: '))

# import model

from predict import *

vectoriser, LRmodel = load_models()

driver = webdriver.Chrome(PATH)

# function

def loop(number_loop):
    driver.execute_script('window.scrollTo(1, document.documentElement.scrollHeight);')
    time.sleep(number_loop)

    
def check():
    name = driver.find_elements_by_xpath('//*[@id="author-text"]')
    number_of_items=len(name)
    
    return number_of_items

# main

def comment_crawl(url_page, Total_comment): 

    results = []
    driver.get(url_page)

    # just scoll until comment appers

    time.sleep(5)
    driver.execute_script('window.scrollTo(1, 500);')
    time.sleep(5)

    #loop
    while check() < Total_comment:

        loop(2)
        print(f'Prossesing : {(check()/Total_comment)*100} %')

    comment  =   driver.find_elements_by_xpath('//*[@id="content-text"]')
    name     =   driver.find_elements_by_xpath('//*[@id="author-text"]')
    like     =   driver.find_elements_by_xpath('//*[@id="vote-count-middle"]')

    number_of_items=len(name)    
    print(f'Total_comment : {number_of_items}')

    for i in tqdm (range(number_of_items)):
        
        result =  {
                    'user_name' : name[i].text,
                    'comment' : comment[i].text,
                    'likes' : like[i].text
            
                    }
        results.append(result)

    driver.quit()

    os.chdir(BASE)
    box_comment = pd.DataFrame(results, columns=['user_name', 'comment','likes'])
    

    print(box_comment.head(10))

    return box_comment

def main(url_page, Total_comment, vectoriser, LRmodel):

    box_comment = comment_crawl(url_page, Total_comment)
    df = predict(vectoriser, LRmodel, box_comment['comment'])
    #print(df)
    pos = len(df[df['sentiment'] == 1])/len(df)
    neg = len(df[df['sentiment'] == -1])/len(df)

    box_comment['sentiment'] = df['sentiment']
    box_comment.to_csv('Results.csv', index=False)

    print(f'sentiment analysis of {url_page} in {Total_comment} comments: ')
    print(f'positif : {pos*100} % , negatif : {neg*100}')




main(URL_PAGE, Total_comment, vectoriser, LRmodel)