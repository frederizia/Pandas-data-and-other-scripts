This is a project involving scraping Twitter data using ```tweepy``` and the official Twitter REST API. I've chosen popular American political figures and some news outlets, with a view of tracking frequency of tweets and doing some sentiment analysis on the tweets.

The output is saved to logfiles with names

```tweets-screen_name-YearMonthDay.log```
```search-query-YearMonthDay.log```

where JSON objects are returned including essential information.

This is preliminary. Due to the volume of tweets and the limitations of the API the search cannot capture many tweets. Retweets are excluded. 
