import tweepy
import configparser

# read configuration
config = configparser.ConfigParser()
config.read('config.ini')

api_key = config['twitter']['Api_key']
api_key_secret = config['twitter']['Api_keys_secret']

access_token = config['twitter']['Access_Token']
access_token_secret = config['twitter']['Access_Token_secret']

print("Api  key")
print(api_key)


# authentication

auth = tweepy.OAuthHandler(api_key, api_key_secret)
auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth)



# # Define the username or user ID of the Twitter account you want to retrieve tweets from
username = "@AbdulRehmanML"

# # Specify the number of tweets you want to retrieve (less than 1000)
tweet_count = 1

print(tweet_count)
# # Retrieve the tweets
tweets = tweepy.Cursor(api.user_timeline, screen_name=username, count=tweet_count).items()

# # Iterate over the tweets and print their text
for tweet in tweets:
    print(tweet.text)

# public_tweets = api.home_timeline()

# print( public_tweets[0].text)