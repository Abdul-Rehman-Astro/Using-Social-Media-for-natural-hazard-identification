import requests
from bs4 import BeautifulSoup

# Get the Twitter user's handle.
# handle = input("Enter the Twitter user's handle: ")

# Make a request to the Twitter user's profile page.
response = requests.get("https://twitter.com/home")

# Parse the response as HTML.
soup = BeautifulSoup(response.content, "lxml")

print(soup)
# Find all of the tweets on the page.
tweets = soup.find_all("div", class_="tweet")

# Print each tweet.
for tweet in tweets:
    print(tweet.find("p").text)