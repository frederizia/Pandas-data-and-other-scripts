from __future__ import division
import tweepy
import json
import time
import datetime
import collections
from keys import *

def twitter_auth():
    # Authenticate
    auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
    auth.set_access_token(ACCESS_TOKEN, ACCESS_SECRET)

    # Make API call
    api = tweepy.API(auth)
    return api
    
def log_time():
    return time.strftime("%Y-%m-%d:%H:%M:%S")	
	
def check_entry(wd,name,screenname, tweet_id, tweet_time):
    return_code = 0
    date = file_date(tweet_time)
    data_list =[]
    try:
        print "Checking file", "{}-{}-{}.log".format(name,screenname, date), "..."
        check_file = open("{}/{}-{}-{}.log".format(wd,name,screenname, date), 'r')
        for line in check_file:
            data_list.append(json.loads(line))
        for obj in data_list:
            for key, value in obj.iteritems():
                    if key == 'id':
                        if value == tweet_id:
                            return_code = 1
                            break
        check_file.close()
    except:
        return_code = 0
    
    return return_code

def file_date(t):
    date_new = t.strftime("%Y%m%d")
    return date_new

def time_to_string(t):
    date_str = t.strftime("%Y-%m-%d:%H:%M:%S")
    return date_str

def find_tweets(api,screenname,wd):
    tweets = api.user_timeline(screen_name=screenname, tweet_mode='extended',count=200)

    # save to file
    entry_count = 0
    for tweet in tweets:
        dupl_check = check_entry(wd,'tweets',screenname, tweet.id, tweet.created_at)
        if dupl_check == 0:
            entry_count +=1
            tweet_dict                = collections.OrderedDict()
            tweet_dict["t"]           = time_to_string(tweet.created_at)
            tweet_dict["screen_name"]  = screenname
            tweet_dict["log_time"]    = log_time()
            tweet_dict["id"]          = tweet.id
            tweet_dict["content"]     = tweet.full_text
            tweet_dict["source"]      = tweet.source
            tweet_dict["fav_count"]   = tweet.favorite_count
            tweet_dict["retweet_count"] = tweet.retweet_count
            tweet_dict["mentions"]    = tweet.entities["user_mentions"]
            tweet_dict["hashtags"]    = tweet.entities["hashtags"]
            tweet_dict["location"]    = tweet.user.location

            tweet_file = open("{}/tweets-{}-{}.log".format(wd,screenname, file_date(tweet.created_at)),'a+')
            tweet_file.write(json.dumps(tweet_dict)+"\n")
            tweet_file.close()
        else:
            print "Tweet already saved."

    print entry_count, "new tweets for", screenname

    return

def search_tweets(api,query,name,wd):
    tweets = api.search(q=query, count=200, tweet_mode='extended')

    # save to file
    entry_count = 0
    for tweet in tweets:
        dupl_check = check_entry(wd,'search',name, tweet.id, tweet.created_at)
        if dupl_check == 0:
            if (not tweet.retweeted) and ('RT @' not in tweet.full_text):
                entry_count +=1
                tweet_dict                = collections.OrderedDict()
                tweet_dict["t"]           = time_to_string(tweet.created_at)
                tweet_dict["screen_name"]  = tweet.user.screen_name
                tweet_dict["log_time"]    = log_time()    
                tweet_dict["id"]          = tweet.id
                tweet_dict["content"]     = tweet.full_text
                tweet_dict["source"]      = tweet.source
                tweet_dict["fav_count"]   = tweet.favorite_count
                tweet_dict["retweet_count"] = tweet.retweet_count
                tweet_dict["mentions"]    = tweet.entities["user_mentions"]
                tweet_dict["hashtags"]    = tweet.entities["hashtags"]
                tweet_dict["location"]    = tweet.user.location

                tweet_file = open("{}/search-{}-{}.log".format(wd,name, file_date(tweet.created_at)),'a+')
                tweet_file.write(json.dumps(tweet_dict)+"\n")
                tweet_file.close()
        else:
            print "Tweet already saved."

    print entry_count, "new tweets for", query

    return

def main():

    # Get the following from keys.py:
    #
    # For the Twitter API:
    # CONSUMER_KEY, CONSUMER_SECRET
    # ACCESS_TOKEN, ACCESS_SECRET
    #
    # For saving files:
    # WD = working directory


    users_politicians = ['realDonaldTrump', 'HillaryClinton', 'BarackObama']
    users_media = ['BBCWorld', 'cbcnews', 'cnn', 'foxnews', 'guardian', 'MSNBC', 'BreitbartNews', 'Reuters', 'CBSNews', 'NBCNews']
    search = ['Trump', 'Obama', 'Hillary']
    
    # make API call
    API = twitter_auth()



    # politician tweets
    for p in users_politicians:
        find_tweets(API,p,WD)
        search_tweets(API, "@{}-filter:noretweets".format(p),p, WD)
        #search_tweets(API, "%40{}%20-filter%3Anoretweets".format(p),p, WD)

    # news tweets
    for m in users_media:
        find_tweets(API,m,WD)

    # tweets containing
    for s in search:
        search_tweets(API, s, s, WD)




    return


if __name__ == "__main__":
    main()
