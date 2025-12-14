from ddgs import DDGS
import requests
from bs4 import BeautifulSoup
import re
import time

class DatasetFinder:
    def __init__(self):
        print("Initializing Dataset Finder...")
        self.ddgs = DDGS()

    def search(self, query, max_results=30, retries=3):
        """
        Search for a topic and return a list of texts.
        Includes retry mechanism and error handling.
        """
        print(f"Searching for: {query}")
        print(f"Max results: {max_results}")
        results = []
        
        backends = ['auto', 'html', 'lite']
        
        for attempt in range(retries):
            try:
                # Iterate through backends
                for backend in backends:
                    print(f"  Attempt {attempt+1}/{retries} - Using backend='{backend}'...")
                    try:
                        ddg_results = self.ddgs.text(query, region='id-id', backend=backend, max_results=max_results)
                        
                        # Convert generator/list to list if needed and check contents
                        # ddqs usually returns a generator or list of dicts
                        if ddg_results:
                            current_batch = []
                            for res in ddg_results:
                                item = {}
                                if 'body' in res:
                                    item['text'] = res['body']
                                elif 'snippet' in res:
                                    item['text'] = res['snippet']
                                else:
                                    continue
                                
                                item['source'] = res.get('href', 'Unknown')
                                item['title'] = res.get('title', 'No Title')
                                current_batch.append(item)
                            
                            if current_batch:
                                results.extend(current_batch)
                                print(f"    Backend '{backend}' found {len(current_batch)} results.")
                                break # Stop trying backends if we found something
                            else:
                                print(f"    Backend '{backend}' returned empty content.")
                        else:
                             print(f"    Backend '{backend}' returned no results.")
                             
                    except Exception as b_error:
                        print(f"    Backend '{backend}' error: {b_error}")
                        time.sleep(1) # Short pause between backends
                
                if results:
                    break # Stop retrying if we have results
                
                print(f"  No results found in attempt {attempt+1}. Waiting before retry...")
                time.sleep(2 * (attempt + 1)) # Exponential backoff
                
            except Exception as e:
                print(f"Search loop error: {e}")
                time.sleep(2)

        # Scrape / cleaning logic fallback? 
        # For now just return what we have.
        
        # Remove duplicates based on text
        seen_texts = set()
        unique_results = []
        for r in results:
            if r['text'] not in seen_texts:
                seen_texts.add(r['text'])
                unique_results.append(r)
                
        return unique_results

if __name__ == "__main__":
    # Test
    finder = DatasetFinder()
    res = finder.search("Ibukota Baru Indonesia")
    print(f"Got {len(res)} results.")
    for r in res[:5]:
        print(f"- {r[:100]}...")
