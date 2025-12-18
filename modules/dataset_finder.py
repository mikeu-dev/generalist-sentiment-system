"""
Dataset Finder dengan dukungan multiple sources.
Menggunakan Strategy Pattern untuk mendukung berbagai sumber data.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Optional
import time
import logging
from ddgs import DDGS
import requests
from bs4 import BeautifulSoup
import os

logger = logging.getLogger(__name__)


class BaseSourceScraper(ABC):
    """Base class untuk semua scraper sumber data."""
    
    def __init__(self):
        self.enabled = True
        self.rate_limit = 10
        self.timeout = 30
        
    @abstractmethod
    def search(self, query: str, max_results: int = 100) -> List[Dict]:
        """
        Mencari data dari sumber.
        
        Args:
            query: Kata kunci pencarian
            max_results: Maksimum hasil yang diinginkan
            
        Returns:
            List of dict dengan format: {'text': str, 'source': str, 'title': str}
        """
        pass
    
    @abstractmethod
    def get_source_id(self) -> str:
        """Return unique identifier untuk sumber ini."""
        pass
    
    @abstractmethod
    def get_source_name(self) -> str:
        """Return human-readable name untuk sumber ini."""
        pass
    
    def is_enabled(self) -> bool:
        """Check apakah sumber ini enabled."""
        return self.enabled


class DuckDuckGoScraper(BaseSourceScraper):
    """Scraper untuk DuckDuckGo search (existing functionality)."""
    
    def __init__(self):
        super().__init__()
        logger.info("Initializing DuckDuckGo Scraper...")
        self.ddgs = DDGS()
        
    def get_source_id(self) -> str:
        return "all"
    
    def get_source_name(self) -> str:
        return "Semua Sumber (DuckDuckGo)"
    
    def search(self, query: str, max_results: int = 100) -> List[Dict]:
        """Search menggunakan DuckDuckGo dengan query expansion."""
        logger.info(f"DuckDuckGo search for: '{query}', target: {max_results} results")
        
        unique_results = []
        seen_texts = set()
        
        # Query variations untuk hasil lebih banyak
        queries = [query]
        if max_results > 40:
            queries.extend([
                f"{query} berita",
                f"{query} opini",
                f"{query} terkini",
                f"{query} analisis"
            ])
        
        for q_idx, current_query in enumerate(queries):
            if len(unique_results) >= max_results:
                break
                
            logger.info(f"Processing query variant {q_idx+1}/{len(queries)}: '{current_query}'")
            
            backends = ['auto', 'html', 'lite']
            
            for attempt in range(3):  # 3 retries
                try:
                    for backend in backends:
                        if len(unique_results) >= max_results:
                            break
                            
                        try:
                            ddg_results = self.ddgs.text(
                                current_query,
                                region='id-id',
                                backend=backend,
                                max_results=max_results
                            )
                            
                            if ddg_results:
                                count_before = len(unique_results)
                                for res in ddg_results:
                                    if len(unique_results) >= max_results:
                                        break
                                        
                                    text = res.get('body') or res.get('snippet')
                                    if not text or text in seen_texts:
                                        continue
                                        
                                    seen_texts.add(text)
                                    unique_results.append({
                                        'text': text,
                                        'source': res.get('href', 'Unknown'),
                                        'title': res.get('title', 'No Title')
                                    })
                                
                                new_items = len(unique_results) - count_before
                                logger.info(f"Backend '{backend}' added {new_items} unique items")
                                
                        except Exception as e:
                            logger.warning(f"Backend '{backend}' error: {e}")
                            time.sleep(1)
                    
                    break  # Success, exit retry loop
                    
                except Exception as e:
                    logger.error(f"Search error (attempt {attempt+1}): {e}")
                    time.sleep(1)
        
        logger.info(f"DuckDuckGo search completed: {len(unique_results)} results")
        return unique_results


class IndonesianNewsScraper(BaseSourceScraper):
    """Scraper untuk situs berita Indonesia."""
    
    def __init__(self):
        super().__init__()
        self.timeout = 20
        self.news_sites = {
            'kompas': 'https://www.kompas.com',
            'detik': 'https://www.detik.com',
            'cnn': 'https://www.cnnindonesia.com',
            'tempo': 'https://www.tempo.co'
        }
        
    def get_source_id(self) -> str:
        return "news_indonesia"
    
    def get_source_name(self) -> str:
        return "Berita Indonesia"
    
    def search(self, query: str, max_results: int = 100) -> List[Dict]:
        """Search dari situs berita Indonesia menggunakan Google search."""
        logger.info(f"Indonesian News search for: '{query}'")
        
        results = []
        seen_texts = set()
        
        # Gunakan DuckDuckGo untuk mencari di situs berita Indonesia
        try:
            ddgs = DDGS()
            
            for site_name, site_url in self.news_sites.items():
                if len(results) >= max_results:
                    break
                    
                # Search dengan site: operator
                site_query = f"site:{site_url.replace('https://', '').replace('www.', '')} {query}"
                logger.info(f"Searching {site_name}: {site_query}")
                
                try:
                    search_results = ddgs.text(
                        site_query,
                        region='id-id',
                        max_results=max_results // len(self.news_sites)
                    )
                    
                    if search_results:
                        for res in search_results:
                            if len(results) >= max_results:
                                break
                                
                            text = res.get('body') or res.get('snippet')
                            if not text or text in seen_texts:
                                continue
                            
                            seen_texts.add(text)
                            results.append({
                                'text': text,
                                'source': f"{site_name.capitalize()}: {res.get('href', 'Unknown')}",
                                'title': res.get('title', 'No Title')
                            })
                        
                        logger.info(f"Found {len(results)} results from {site_name}")
                    
                    time.sleep(1)  # Rate limiting
                    
                except Exception as e:
                    logger.warning(f"Error searching {site_name}: {e}")
                    continue
        
        except Exception as e:
            logger.error(f"Indonesian News search error: {e}")
        
        logger.info(f"Indonesian News search completed: {len(results)} results")
        return results


class TwitterScraper(BaseSourceScraper):
    """Scraper untuk Twitter/X menggunakan ntscraper."""
    
    def __init__(self):
        super().__init__()
        self.rate_limit = 15
        try:
            from ntscraper import Nitter
            self.scraper = Nitter(log_level=1)
            self.enabled = True
            logger.info("Twitter scraper initialized successfully")
        except ImportError:
            logger.warning("ntscraper not installed. Twitter scraping disabled.")
            self.enabled = False
        except Exception as e:
            logger.error(f"Failed to initialize Twitter scraper: {e}")
            self.enabled = False
    
    def get_source_id(self) -> str:
        return "twitter"
    
    def get_source_name(self) -> str:
        return "Twitter/X"
    
    def search(self, query: str, max_results: int = 100) -> List[Dict]:
        """Search tweets menggunakan ntscraper."""
        if not self.enabled:
            logger.warning("Twitter scraper is disabled")
            return []
        
        logger.info(f"Twitter search for: '{query}'")
        results = []
        
        try:
            # Search tweets
            tweets = self.scraper.get_tweets(query, mode='term', number=max_results)
            
            if tweets and 'tweets' in tweets:
                for tweet in tweets['tweets']:
                    text = tweet.get('text', '')
                    if text:
                        results.append({
                            'text': text,
                            'source': f"Twitter: @{tweet.get('user', {}).get('username', 'unknown')}",
                            'title': f"Tweet by @{tweet.get('user', {}).get('username', 'unknown')}"
                        })
                    
                    if len(results) >= max_results:
                        break
            
            logger.info(f"Twitter search completed: {len(results)} results")
            
        except Exception as e:
            logger.error(f"Twitter search error: {e}")
        
        return results


class RedditScraper(BaseSourceScraper):
    """Scraper untuk Reddit menggunakan PRAW."""
    
    def __init__(self):
        super().__init__()
        self.rate_limit = 10
        
        # Check for credentials
        client_id = os.environ.get('REDDIT_CLIENT_ID', '')
        client_secret = os.environ.get('REDDIT_CLIENT_SECRET', '')
        user_agent = os.environ.get('REDDIT_USER_AGENT', 'SentimentAnalysisBot/1.0')
        
        if not client_id or not client_secret:
            logger.warning("Reddit credentials not found. Reddit scraping disabled.")
            self.enabled = False
            return
        
        try:
            import praw
            self.reddit = praw.Reddit(
                client_id=client_id,
                client_secret=client_secret,
                user_agent=user_agent
            )
            self.enabled = True
            logger.info("Reddit scraper initialized successfully")
        except ImportError:
            logger.warning("praw not installed. Reddit scraping disabled.")
            self.enabled = False
        except Exception as e:
            logger.error(f"Failed to initialize Reddit scraper: {e}")
            self.enabled = False
    
    def get_source_id(self) -> str:
        return "reddit"
    
    def get_source_name(self) -> str:
        return "Reddit"
    
    def search(self, query: str, max_results: int = 100) -> List[Dict]:
        """Search Reddit posts and comments."""
        if not self.enabled:
            logger.warning("Reddit scraper is disabled")
            return []
        
        logger.info(f"Reddit search for: '{query}'")
        results = []
        
        try:
            # Search subreddits
            for submission in self.reddit.subreddit('all').search(query, limit=max_results):
                # Add post title and selftext
                if submission.selftext:
                    results.append({
                        'text': f"{submission.title}. {submission.selftext}",
                        'source': f"Reddit: r/{submission.subreddit.display_name}",
                        'title': submission.title
                    })
                else:
                    results.append({
                        'text': submission.title,
                        'source': f"Reddit: r/{submission.subreddit.display_name}",
                        'title': submission.title
                    })
                
                # Add top comments
                submission.comments.replace_more(limit=0)
                for comment in submission.comments.list()[:3]:  # Top 3 comments per post
                    if len(results) >= max_results:
                        break
                    
                    if hasattr(comment, 'body') and comment.body:
                        results.append({
                            'text': comment.body,
                            'source': f"Reddit: r/{submission.subreddit.display_name}",
                            'title': f"Comment on: {submission.title}"
                        })
                
                if len(results) >= max_results:
                    break
            
            logger.info(f"Reddit search completed: {len(results)} results")
            
        except Exception as e:
            logger.error(f"Reddit search error: {e}")
        
        return results[:max_results]


class YouTubeScraper(BaseSourceScraper):
    """Scraper untuk YouTube comments."""
    
    def __init__(self):
        super().__init__()
        self.rate_limit = 10
        
        try:
            from youtube_comment_downloader import YoutubeCommentDownloader
            self.downloader = YoutubeCommentDownloader()
            self.enabled = True
            logger.info("YouTube scraper initialized successfully")
        except ImportError:
            logger.warning("youtube-comment-downloader not installed. YouTube scraping disabled.")
            self.enabled = False
        except Exception as e:
            logger.error(f"Failed to initialize YouTube scraper: {e}")
            self.enabled = False
    
    def get_source_id(self) -> str:
        return "youtube"
    
    def get_source_name(self) -> str:
        return "YouTube Comments"
    
    def search(self, query: str, max_results: int = 100) -> List[Dict]:
        """Search YouTube videos and extract comments."""
        if not self.enabled:
            logger.warning("YouTube scraper is disabled")
            return []
        
        logger.info(f"YouTube search for: '{query}'")
        results = []
        
        try:
            # First, search for videos using DuckDuckGo
            ddgs = DDGS()
            video_query = f"site:youtube.com {query}"
            search_results = ddgs.text(video_query, region='id-id', max_results=5)
            
            if not search_results:
                logger.warning("No YouTube videos found")
                return []
            
            # Extract comments from found videos
            for video in search_results:
                if len(results) >= max_results:
                    break
                
                url = video.get('href', '')
                if 'youtube.com/watch' not in url:
                    continue
                
                try:
                    comments = self.downloader.get_comments_from_url(url, sort_by=0)
                    
                    for comment in comments:
                        if len(results) >= max_results:
                            break
                        
                        text = comment.get('text', '')
                        if text:
                            results.append({
                                'text': text,
                                'source': f"YouTube: {url}",
                                'title': video.get('title', 'YouTube Video')
                            })
                    
                except Exception as e:
                    logger.warning(f"Error extracting comments from {url}: {e}")
                    continue
            
            logger.info(f"YouTube search completed: {len(results)} results")
            
        except Exception as e:
            logger.error(f"YouTube search error: {e}")
        
        return results


class DatasetFinder:
    """Main class untuk mencari dataset dari berbagai sumber."""
    
    def __init__(self):
        logger.info("Initializing DatasetFinder with multiple sources...")
        
        # Initialize all scrapers
        self.scrapers = {
            'all': DuckDuckGoScraper(),
            'news_indonesia': IndonesianNewsScraper(),
            'twitter': TwitterScraper(),
            'reddit': RedditScraper(),
            'youtube': YouTubeScraper()
        }
        
        # Log enabled sources
        enabled_sources = [s for s, scraper in self.scrapers.items() if scraper.is_enabled()]
        logger.info(f"Enabled sources: {', '.join(enabled_sources)}")
    
    def get_available_sources(self) -> List[Dict]:
        """
        Get list of available data sources.
        
        Returns:
            List of dict dengan format: {'id': str, 'name': str, 'enabled': bool}
        """
        sources = []
        for source_id, scraper in self.scrapers.items():
            sources.append({
                'id': scraper.get_source_id(),
                'name': scraper.get_source_name(),
                'enabled': scraper.is_enabled()
            })
        return sources
    
    def search(self, query: str, source: str = 'all', max_results: int = 100) -> List[Dict]:
        """
        Search untuk data dari sumber yang ditentukan.
        
        Args:
            query: Kata kunci pencarian
            source: Source ID (default: 'all')
            max_results: Maksimum hasil yang diinginkan
            
        Returns:
            List of dict dengan format: {'text': str, 'source': str, 'title': str}
        """
        logger.info(f"DatasetFinder.search(query='{query}', source='{source}', max_results={max_results})")
        
        # Validate source
        if source not in self.scrapers:
            logger.error(f"Invalid source: {source}")
            raise ValueError(f"Invalid source: {source}. Available: {list(self.scrapers.keys())}")
        
        scraper = self.scrapers[source]
        
        # Check if enabled
        if not scraper.is_enabled():
            logger.warning(f"Source '{source}' is disabled")
            return []
        
        # Perform search
        try:
            results = scraper.search(query, max_results)
            logger.info(f"Search completed: {len(results)} results from '{source}'")
            return results
        except Exception as e:
            logger.error(f"Search failed for source '{source}': {e}", exc_info=True)
            return []


if __name__ == "__main__":
    # Test
    finder = DatasetFinder()
    
    print("\n=== Available Sources ===")
    for source in finder.get_available_sources():
        status = "✓" if source['enabled'] else "✗"
        print(f"{status} {source['name']} (id: {source['id']})")
    
    print("\n=== Testing DuckDuckGo ===")
    results = finder.search("Ibukota Baru Indonesia", source='all', max_results=5)
    print(f"Got {len(results)} results")
    for r in results[:3]:
        print(f"- {r['text'][:100]}...")
