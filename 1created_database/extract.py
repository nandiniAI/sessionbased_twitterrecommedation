import tweepy 
import csv
from tweepy import OAuthHandler
from tweepy.streaming import StreamListener
import pandas as pd

consumer_key = 'dp45720Jpqn89pR79KKZ7yd0N'
consumer_secret ='h3VXQdsrLwciH3ITWqL4u6vZ12cECOuOsDZWDdAONe5gK5X9i5'
access_key = '286552408-82QYeHABYcelSZGvRAjcSIlta14DVv1XYfOOC2W4'
access_secret = '9aCLCcxP2IlnwkPUrdcFJDliKhTJds71h1dYUTL0HfShr'

screen_name=['thekiranbedi','womenofhistory','mahboob_h','mrsfunnybones']

def get_all_tweets(screen_name):
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_key, access_secret)
    api = tweepy.API(auth)
    new_tweets = api.user_timeline(screen_name = screen_name,count=200)
    outtweets = [[tweet.id_str, tweet.created_at, tweet.text.encode("utf-8"),screen_name] for tweet in new_tweets]
    with open('tweets.csv', 'a') as f:
        writer = csv.writer(f)
        writer.writerow(["id","created_at","text","screen_name"])
        writer.writerows(outtweets)
	
    pass


if __name__ == '__main__':
    for name in screen_name:
        get_all_tweets(name)

