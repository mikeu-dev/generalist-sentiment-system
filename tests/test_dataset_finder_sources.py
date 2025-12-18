
import pytest
import sys
from unittest.mock import MagicMock, patch

# --- DuckDuckGo Scraper Tests ---

def test_duckduckgo_scraper_initialization():
    from modules.dataset_finder import DuckDuckGoScraper
    scraper = DuckDuckGoScraper()
    assert scraper.get_source_id() == "all"
    assert "DuckDuckGo" in scraper.get_source_name()
    assert scraper.is_enabled() is True

@patch('modules.dataset_finder.DDGS')
def test_duckduckgo_search(mock_ddgs):
    from modules.dataset_finder import DuckDuckGoScraper
    # Setup mock
    mock_instance = mock_ddgs.return_value
    mock_instance.text.return_value = [
        {'body': 'Test result 1', 'href': 'http://test.com/1', 'title': 'Title 1'}
    ]
    
    scraper = DuckDuckGoScraper()
    results = scraper.search("test query", max_results=1)
    
    assert len(results) == 1
    assert results[0]['text'] == 'Test result 1'

# --- Indonesian News Scraper Tests ---

def test_news_scraper_initialization():
    from modules.dataset_finder import IndonesianNewsScraper
    scraper = IndonesianNewsScraper()
    assert scraper.get_source_id() == "news_indonesia"

@patch('modules.dataset_finder.DDGS')
def test_news_search(mock_ddgs):
    from modules.dataset_finder import IndonesianNewsScraper
    # Setup mock
    mock_instance = mock_ddgs.return_value
    mock_instance.text.return_value = [
        {'body': 'Berita terkini', 'href': 'https://news.detik.com/read/1', 'title': 'Judul Berita'}
    ]
    
    scraper = IndonesianNewsScraper()
    with patch('time.sleep', return_value=None):
        results = scraper.search("pemilu", max_results=1)
    
    assert len(results) >= 1
    assert "Berita terkini" in results[0]['text']
    # Flexible source check
    assert any(site in results[0]['source'] for site in ["Detik", "Kompas", "CNN", "Tempo"])

# --- Twitter Scraper Tests ---

def test_twitter_scraper_initialization():
    from modules.dataset_finder import TwitterScraper
    scraper = TwitterScraper()
    # Check if we can initialize it (might be enabled or disabled based on env)
    assert scraper.get_source_id() == "twitter"

def test_twitter_search():
    # We need to mock 'ntscraper' module BEFORE importing modules.dataset_finder
    # or ensure we mock where it is used.
    # Since imports are inside __init__, we mock sys.modules for 'ntscraper'
    
    mock_nitter = MagicMock()
    mock_nitter_instance = mock_nitter.return_value
    mock_nitter_instance.get_tweets.return_value = {
        'tweets': [
            {'text': 'Tweet 1', 'user': {'username': 'user1'}}
        ]
    }
    
    with patch.dict(sys.modules, {'ntscraper': mock_nitter}):
        # Reload or re-import inside the context might be needed if it was top-level,
        # but here it is local import inside __init__.
        from modules.dataset_finder import TwitterScraper
        
        # We also need to patch the Nitter class specifically if the code does `from ntscraper import Nitter`
        mock_nitter.Nitter = MagicMock(return_value=mock_nitter_instance)
        
        scraper = TwitterScraper()
        # Force enable if it failed due to some other reason
        scraper.scraper = mock_nitter_instance 
        scraper.enabled = True
        
        results = scraper.search("query", max_results=1)
        
        assert len(results) == 1
        assert results[0]['text'] == 'Tweet 1'
        assert '@user1' in results[0]['source']

# --- YouTube Scraper Tests ---

def test_youtube_search():
    # Mock youtube_comment_downloader
    mock_ycd = MagicMock()
    mock_downloader = MagicMock()
    mock_ycd.YoutubeCommentDownloader.return_value = mock_downloader
    mock_downloader.get_comments_from_url.return_value = iter([
        {'text': 'Great video!'}
    ])
    
    # Mock DDGS for video search
    with patch.dict(sys.modules, {'youtube_comment_downloader': mock_ycd}):
        with patch('modules.dataset_finder.DDGS') as mock_ddgs:
            mock_ddgs_instance = mock_ddgs.return_value
            mock_ddgs_instance.text.return_value = [
                {'href': 'https://www.youtube.com/watch?v=123', 'title': 'Video Review'}
            ]
            
            from modules.dataset_finder import YouTubeScraper
            
            scraper = YouTubeScraper()
            scraper.enabled = True
            # Inject dependency manually if needed or rely on mock
            scraper.downloader = mock_downloader
            
            results = scraper.search("review", max_results=1)
            
            assert len(results) == 1
            assert results[0]['text'] == 'Great video!'

# --- DatasetFinder Integration Tests ---

def test_dataset_finder_sources():
    from modules.dataset_finder import DatasetFinder
    finder = DatasetFinder()
    sources = finder.get_available_sources()
    source_ids = [s['id'] for s in sources]
    assert 'all' in source_ids
    assert 'news_indonesia' in source_ids

def test_dataset_finder_dispatch():
    from modules.dataset_finder import DatasetFinder
    finder = DatasetFinder()
    
    # Mock specific scraper in the dictionary
    mock_scraper = MagicMock()
    mock_scraper.is_enabled.return_value = True
    mock_scraper.search.return_value = []
    
    finder.scrapers['mock'] = mock_scraper
    
    finder.search("query", source='mock')
    mock_scraper.search.assert_called_once()
