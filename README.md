# Title: Using Social Media for natural hazard identification

## Aim: 

Text based classification for floods and non floods tweets
Top 10 cities based on number of tweets

## Dataset:

We are working with a twitter dataset which is extracted using **snscraper** ( snscrape is a scraper for social networking services (SNS). It scrapes things like user posts ,user profiles, hashtags, or searches and returns the discovered items, e.g. the relevant posts) using multiple fine tuned searches.

In our case we used snscraper to extract users location , hashtags(#flood #floodinAssam ), searches (live tweets, top tweets, and users), tweets (single or surrounding thread), list posts and trends.

Dataset size : 2334 tweets extracted
Search used : flood  "flood" (flood OR Monsoon) -Pakisthan -Bangladesh -California
Further optimization : min_replies:2 min_faves:50 min_retweets:5 lang:en since:2022-06-01

## Techniques:
We tried NLTK, spacy and Tranformers(BERT model)  for Named Entity Recognition.

Named Entity Recognition is also provided by libraries NLTK and Spacy. Spacy provides a model "en_core_web_sm()" which does the task pretty effectively but lacks in tokenizing the sentences effectively leading to low accuracy. So using the pre_trained model from transformers does the task of extracting locations from the tweets effectively. 

We have used **bert-large-NER** for extracting location (LOC) from given tweets because bert-large-NER is a fine-tuned BERT model that is ready to use for Named Entity Recognition and achieves state-of-the-art performance for the NER task. It has been trained to recognize four types of entities: location (LOC), organizations (ORG), person (PER) and Miscellaneous (MISC).
Another BERT model- bert-base-uncased was also used and fine tuned using CoNLL-2003 dataset which is a widely used benchmark for NER, the bert-base-uncased model achieves state-of-the-art performance. It surpasses the previous best results on this dataset and achieves an F1-score of around 92%.
Progress till now:
We have extracted user location and location from a given tweet,we have placed all of the used code and dataset here.

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
