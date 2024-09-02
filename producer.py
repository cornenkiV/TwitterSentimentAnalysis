import json
import os
import time
from typing import List, Dict

from kafka import KafkaProducer
import dotenv
import requests


class TweetProducer:
    def __init__(self, bootstrap_servers: List[str], topic: str):
        self.producer = KafkaProducer(
            bootstrap_servers=bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode('utf-8')
        )
        self.topic = topic

    def send_tweet(self, tweet: Dict):
        self.producer.send(self.topic, value=tweet)
        print(f"Sent: {tweet}")

    def send_hardcoded_data(self):
        hardcoded_tweets = [
            {"id_str": "1", "user": {"screen_name": "user1"}, "text": "I love this movie!"},
            {"id_str": "2", "user": {"screen_name": "user2"}, "text": "This is the worst thing ever bad."},
            {"id_str": "3", "user": {"screen_name": "user3"}, "text": "Going to the cinema."}
        ]
        for tweet in hardcoded_tweets:
            self.send_tweet(tweet)
            time.sleep(10)

    def send_api_data(self, data: Dict):
        for tweet in data.get('tweets', []):
            try:
                formatted_tweet = {
                    "created_at": tweet['legacy']['created_at'],
                    "id_str": tweet['legacy']['id_str'],
                    "text": tweet['legacy']['full_text'],
                    "user": {
                        "id_str": tweet['legacy']['user_id_str'],
                        "screen_name": tweet['core']['user_results']['result']['legacy']['screen_name']
                    }
                }
                self.send_tweet(formatted_tweet)
            except KeyError as e:
                print(f"Failed to format tweet: {e}")
            except Exception as e:
                print(f"Unexpected error while formatting tweet: {e}")


class TwitterAPIClient:
    def __init__(self, api_key: str, api_host: str):
        self.api_key = api_key
        self.api_host = api_host

    def search_tweets(self, query: str, tweet_type: str = "Latest") -> Dict:
        url = "https://twitter-api47.p.rapidapi.com/v2/search"
        querystring = {"query": query, "type": tweet_type}

        headers = {
            'x-rapidapi-key': self.api_key,
            'x-rapidapi-host': self.api_host
        }

        response = requests.get(url, headers=headers, params=querystring)
        response.raise_for_status()
        return response.json()


def main():
    dotenv.load_dotenv("environment_variables.env")

    producer = TweetProducer(bootstrap_servers=['localhost:9092'], topic='tweets')

    # Uncomment to send hardcoded data
    # producer.send_hardcoded_data()

    api_key = os.environ.get('RAPIDAPI_KEY')
    api_host = "twitter-api47.p.rapidapi.com"

    twitter_client = TwitterAPIClient(api_key=api_key, api_host=api_host)

    while True:
        # Input search parameter from console
        search_query = input("Enter a search query (or 'X' to exit): ")

        if search_query.strip().upper() == 'X':
            print("Exiting...")
            break

        try:
            data = twitter_client.search_tweets(query=search_query)
            producer.send_api_data(data)
        except requests.HTTPError as e:
            print(f"HTTP error occurred: {e}")
        except Exception as e:
            print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
