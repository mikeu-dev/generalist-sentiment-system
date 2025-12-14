from duckduckgo_search import DDGS
import requests
from bs4 import BeautifulSoup
import re
import time

class DatasetFinder:
    def __init__(self):
        print("Initializing Dataset Finder...")
        self.ddgs = DDGS()

    def search(self, query, max_results=30):
        """
        Search for a topic and return a list of texts.
        """
        print(f"Searching for: {query}")
        results = []
        
        # 1. Search for text snippets using DuckDuckGo
        try:
            # Try default auto backend first
            print("  Trying backend='auto'...")
            ddg_results = self.ddgs.text(query, max_results=max_results)
            
            # Fallback backends if empty
            if not ddg_results:
                print("  'auto' returned 0. Trying backend='html'...")
                time.sleep(1)
                ddg_results = self.ddgs.text(query, region='id-id', backend='html', max_results=max_results)
                
            if not ddg_results:
                print("  'html' returned 0. Trying backend='lite'...")
                time.sleep(1)
                ddg_results = self.ddgs.text(query, region='id-id', backend='lite', max_results=max_results)

            if ddg_results:
                for res in ddg_results:
                    # Collect snippets as they are often good enough for sentiment
                    if 'body' in res:
                        results.append(res['body'])
                    elif 'snippet' in res:
                         results.append(res['snippet'])
            
            print(f"Found {len(results)} snippets from search.")
            
        except Exception as e:
            print(f"Search error: {e}")

        # 2. (Optional) Basic scraping of referenced pages could go here
        # For simplicity and speed, we will rely on snippets for now as they are direct.
        # Deep scraping requires handling connection timeouts, parsing various HTML structures, etc.
        
        unique_results = list(set(results))
        return unique_results

if __name__ == "__main__":
    # Test
    finder = DatasetFinder()
    res = finder.search("Ibukota Baru Indonesia")
    print(f"Got {len(res)} results.")
    for r in res[:5]:
        print(f"- {r[:100]}...")
