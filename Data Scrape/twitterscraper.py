#!/usr/bin/env python3
import time
import requests
import os
import json
import csv
import pandas as pd
a = time.time()

bearer_token = "AAAAAAAAAAAAAAAAAAAAAKBnQAEAAAAA%2F7wDv%2BKDTrGnt2nW1Dqx7VKVOyg%3DjBNBO37zt9MZqH8pfgR3DIXhrApCoSnbbThHHWfBEQrBNkFbhH"
search_url = "https://api.twitter.com/2/tweets/search/all"
query_params = {
    'query':"(flood -is:retweet lang:en)",
                'max_results':'500',
                'start_time':'2018-01-01T00:00:00Z',
                'end_time':'2018-07-18T02:30:44.000Z',
                'tweet.fields':'id,text,created_at',
    #'next_token':'next_token'
              }


def create_headers(bearer_token):
    headers = {"Authorization": "Bearer {}".format(bearer_token)}
    return headers

def connect_to_endpoint(url, headers, params):
    response = requests.request("GET", search_url, headers=headers, params=params)
    print(response.status_code)
    if response.status_code != 200:
        raise Exception(response.status_code, response.text)
    return response.json()

def main():
    headers = create_headers(bearer_token)
    #response = connect_to_endpoint(search_url, headers, query_params)
    #df = pd.DataFrame(response['data'])
    #df.to_csv('tweet_test.csv')
    #df = pd.DataFrame(columns=4)
    go = True
    next_token = ''
    tweets = []
    i=0
    while(go and i<200):
        i+=1
        if next_token == '':
            print('first request')
            response = connect_to_endpoint(search_url, headers, query_params)
            #df.append(response['data'])

        else:
            print('additional')
            query_params2 = {
                'query':"(flood -is:retweet lang:en)",
                'max_results':'500',
                'start_time':'2018-01-01T00:00:00Z',
                'end_time':'2018-07-18T02:30:44.000Z',
                'tweet.fields':'id,text,created_at',
                'next_token': next_token
              }

            response = connect_to_endpoint(search_url, headers, query_params2)
            #df.append(response['data'])
        meta = response['meta']
        if 'next_token' not in meta:
            go = False
            if response['meta']['result_count'] == 0:
                break
        else:
            next_token = meta['next_token']

        for item in response['data']:
            crated_at = item['created_at']
            ifd = item['id']
            text = item['text']
            tweets.append([crated_at, ifd, text])
    #df.to_csv('q1_2020.csv')

    df2 = pd.DataFrame(tweets)
    df2.to_csv('2018_14')
    b = time.time()
    print(b-a)
    from datetime import datetime
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("Current Time =", current_time)

if __name__ == "__main__":
    main()


