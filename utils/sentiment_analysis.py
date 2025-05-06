import pandas as pd
import numpy as np
import logging
import requests
from datetime import datetime, timedelta
import re
from textblob import TextBlob
import json

class SentimentAnalysis:
    """
    Class for performing sentiment analysis on market data
    """
    def __init__(self, api_key=None):
        """
        Initialize the sentiment analysis class
        
        Args:
            api_key (str, optional): API key for news sources (if needed)
        """
        self.api_key = api_key
        
        # Setup logger
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
    
    def get_news_articles(self, symbol, days=7):
        """
        Get recent news articles for a given symbol
        
        Args:
            symbol (str): Stock or index symbol
            days (int): Number of days to look back
            
        Returns:
            list: List of news article dictionaries
        """
        try:
            # In a real implementation, you would use a news API like News API, Alpha Vantage, etc.
            # For demo purposes, we'll simulate getting news articles
            
            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            # Simulate news articles
            company_names = {
                'NIFTY': 'Nifty 50',
                'BANKNIFTY': 'Bank Nifty',
                'SENSEX': 'Sensex',
                'RELIANCE': 'Reliance Industries',
                'TCS': 'Tata Consultancy Services',
                'INFY': 'Infosys',
                'HDFCBANK': 'HDFC Bank'
            }
            
            company_name = company_names.get(symbol, symbol)
            
            # Generate random news articles
            articles = []
            
            # Simulate news retrieval
            self.logger.info(f"Getting news for {company_name}")
            
            # Return empty list for now
            # In a real implementation, you would get actual news articles from an API
            return []
        except Exception as e:
            self.logger.error(f"Error getting news articles: {str(e)}")
            return []
    
    def analyze_news_sentiment(self, articles):
        """
        Analyze sentiment from news articles
        
        Args:
            articles (list): List of news article dictionaries
            
        Returns:
            dict: Sentiment analysis results
        """
        try:
            if not articles:
                return {
                    'sentiment_score': 0,
                    'sentiment_label': 'neutral',
                    'positive_count': 0,
                    'negative_count': 0,
                    'neutral_count': 0,
                    'article_count': 0
                }
            
            sentiments = []
            
            for article in articles:
                # Extract title and content
                title = article.get('title', '')
                content = article.get('content', '')
                
                # Combine title and content with title having more weight
                text = title + " " + content
                
                # Analyze sentiment using TextBlob
                blob = TextBlob(text)
                sentiment_score = blob.sentiment.polarity
                
                sentiments.append(sentiment_score)
            
            # Calculate overall sentiment
            avg_sentiment = np.mean(sentiments) if sentiments else 0
            
            # Count positive, negative, and neutral articles
            positive_count = sum(1 for s in sentiments if s > 0.1)
            negative_count = sum(1 for s in sentiments if s < -0.1)
            neutral_count = len(sentiments) - positive_count - negative_count
            
            # Determine sentiment label
            if avg_sentiment > 0.1:
                sentiment_label = 'positive'
            elif avg_sentiment < -0.1:
                sentiment_label = 'negative'
            else:
                sentiment_label = 'neutral'
            
            return {
                'sentiment_score': avg_sentiment,
                'sentiment_label': sentiment_label,
                'positive_count': positive_count,
                'negative_count': negative_count,
                'neutral_count': neutral_count,
                'article_count': len(articles)
            }
        except Exception as e:
            self.logger.error(f"Error analyzing news sentiment: {str(e)}")
            return {
                'sentiment_score': 0,
                'sentiment_label': 'neutral',
                'positive_count': 0,
                'negative_count': 0,
                'neutral_count': 0,
                'article_count': 0
            }
    
    def get_social_media_sentiment(self, symbol):
        """
        Get social media sentiment for a given symbol
        
        Args:
            symbol (str): Stock or index symbol
            
        Returns:
            dict: Social media sentiment analysis results
        """
        try:
            # In a real implementation, you would use social media APIs or web scraping
            # For demo purposes, we'll return a placeholder sentiment analysis
            
            return {
                'sentiment_score': 0,
                'sentiment_label': 'neutral',
                'mentions': 0,
                'positive_mentions': 0,
                'negative_mentions': 0,
                'neutral_mentions': 0
            }
        except Exception as e:
            self.logger.error(f"Error getting social media sentiment: {str(e)}")
            return {
                'sentiment_score': 0,
                'sentiment_label': 'neutral',
                'mentions': 0,
                'positive_mentions': 0,
                'negative_mentions': 0,
                'neutral_mentions': 0
            }
    
    def get_analyst_ratings(self, symbol):
        """
        Get analyst ratings for a given symbol
        
        Args:
            symbol (str): Stock or index symbol
            
        Returns:
            dict: Analyst ratings analysis results
        """
        try:
            # In a real implementation, you would use financial data APIs
            # For demo purposes, we'll return a placeholder analysis
            
            return {
                'consensus': 'hold',
                'buy_ratings': 0,
                'hold_ratings': 0,
                'sell_ratings': 0,
                'target_price': 0,
                'current_price': 0
            }
        except Exception as e:
            self.logger.error(f"Error getting analyst ratings: {str(e)}")
            return {
                'consensus': 'hold',
                'buy_ratings': 0,
                'hold_ratings': 0,
                'sell_ratings': 0,
                'target_price': 0,
                'current_price': 0
            }
    
    def get_market_sentiment(self):
        """
        Get overall market sentiment
        
        Returns:
            dict: Market sentiment analysis results
        """
        try:
            # In a real implementation, you would use market data and indices
            # For demo purposes, we'll return a placeholder sentiment
            
            return {
                'fear_greed_index': 50,  # 0-100 scale: 0=extreme fear, 100=extreme greed
                'sentiment_label': 'neutral',
                'market_trend': 'sideways'
            }
        except Exception as e:
            self.logger.error(f"Error getting market sentiment: {str(e)}")
            return {
                'fear_greed_index': 50,
                'sentiment_label': 'neutral',
                'market_trend': 'sideways'
            }
    
    def analyze_sentiment(self, symbol):
        """
        Perform comprehensive sentiment analysis for a symbol
        
        Args:
            symbol (str): Stock or index symbol
            
        Returns:
            dict: Comprehensive sentiment analysis results
        """
        try:
            # Get news articles
            articles = self.get_news_articles(symbol)
            
            # Analyze news sentiment
            news_sentiment = self.analyze_news_sentiment(articles)
            
            # Get social media sentiment
            social_sentiment = self.get_social_media_sentiment(symbol)
            
            # Get analyst ratings
            analyst_ratings = self.get_analyst_ratings(symbol)
            
            # Get market sentiment
            market_sentiment = self.get_market_sentiment()
            
            # Calculate overall sentiment score
            # Weight different sources based on importance
            overall_score = (
                news_sentiment['sentiment_score'] * 0.4 +
                social_sentiment['sentiment_score'] * 0.2 +
                (1 if analyst_ratings['consensus'] == 'buy' else 
                 -1 if analyst_ratings['consensus'] == 'sell' else 0) * 0.3 +
                (market_sentiment['fear_greed_index'] / 50 - 1) * 0.1
            )
            
            # Normalize score to -1 to 1 range
            overall_score = max(-1, min(1, overall_score))
            
            # Determine overall sentiment label
            if overall_score > 0.2:
                overall_label = 'bullish'
            elif overall_score < -0.2:
                overall_label = 'bearish'
            else:
                overall_label = 'neutral'
            
            return {
                'symbol': symbol,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'overall_score': overall_score,
                'overall_label': overall_label,
                'news_sentiment': news_sentiment,
                'social_sentiment': social_sentiment,
                'analyst_ratings': analyst_ratings,
                'market_sentiment': market_sentiment
            }
        except Exception as e:
            self.logger.error(f"Error analyzing sentiment: {str(e)}")
            return {
                'symbol': symbol,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'overall_score': 0,
                'overall_label': 'neutral',
                'news_sentiment': {},
                'social_sentiment': {},
                'analyst_ratings': {},
                'market_sentiment': {}
            }
