import requests

bearer_token = 'AAAAAAAAAAAAAAAAAAAAAKBnQAEAAAAA%2F7wDv%2BKDTrGnt2nW1Dqx7VKVOyg%3DjBNBO37zt9MZqH8pfgR3DIXhrApCoSnbbThHHWfBEQrBNkFbhH'
search_url = 'https://twitter.com/search?q=%23Pune%20%23Maharashtra%20%23floods%20%23rain&src=typed_query&f=top'
query = ""

def create_headers(bearer_token):
    headers = {'Authorization': f'Bearer {bearer_token}'}
    return headers

def create_params(query):
    params = {'query': query, 'tweet.fields': 'created_at'}
    return params

def connect_to_endpoint(url, headers, params):
    response = requests.get(url, headers=headers, params=params)
    if response.status_code != 200:
        raise Exception(f'Request returned an error: {response.status_code} {response.text}')
    return response.json()

def main():
    headers = create_headers(bearer_token)
    params = create_params(query)
    response = connect_to_endpoint(search_url, headers)
    
    # Process the response data
    for tweet in response['data']:
        tweet_id = tweet['id']
        created_at = tweet['created_at']
        text = tweet['text']
        print(f'Tweet ID: {tweet_id}')
        print(f'Created at: {created_at}')
        print(f'Text: {text}')
        print('---')

if __name__ == '__main__':
    main()
