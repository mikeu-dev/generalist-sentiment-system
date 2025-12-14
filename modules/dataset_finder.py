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
        Uses query expansion and multiple backends to maximize results.
        """
        print(f"Target results: {max_results}")
        
        unique_results = []
        seen_texts = set()
        
        # Prepare content variations to fetch more data
        queries = [query]
        if max_results > 40:
            queries.extend([f"{query} berita", f"{query} opini", f"{query} terkini", f"{query} analisis"])
        
        for q_idx, current_query in enumerate(queries):
            if len(unique_results) >= max_results:
                break
                
            print(f"Processing query variant {q_idx+1}/{len(queries)}: '{current_query}'")
            
            # Search for current_query
            backends = ['auto', 'html', 'lite']
            current_found = False
            
            for attempt in range(retries):
                try:
                    for backend in backends:
                        if len(unique_results) >= max_results:
                            break
                            
                        print(f"  Attempt {attempt+1}/{retries} - Backend '{backend}'...")
                        try:
                            # Fetch batch
                            ddg_results = self.ddgs.text(current_query, region='id-id', backend=backend, max_results=max_results)
                            
                            if ddg_results:
                                count_before = len(unique_results)
                                for res in ddg_results:
                                    if len(unique_results) >= max_results:
                                        break
                                        
                                    text = res.get('body') or res.get('snippet')
                                    if not text:
                                        continue
                                        
                                    if text not in seen_texts:
                                        seen_texts.add(text)
                                        item = {
                                            'text': text,
                                            'source': res.get('href', 'Unknown'),
                                            'title': res.get('title', 'No Title')
                                        }
                                        unique_results.append(item)
                                
                                count_after = len(unique_results)
                                new_items = count_after - count_before
                                print(f"    Backend '{backend}' added {new_items} unique items.")
                                
                                if new_items > 0:
                                    current_found = True
                            else:
                                print(f"    Backend '{backend}' returned no results.")
                                
                        except Exception as b_error:
                            print(f"    Backend '{backend}' error: {b_error}")
                            time.sleep(1)
                    
                    if current_found:
                        break # Move to next query variant if we found something for this one
                    
                    print(f"  No results for '{current_query}' in attempt {attempt+1}. Retry...")
                    time.sleep(1)
                    
                except Exception as e:
                    print(f"  Search error: {e}")
                    time.sleep(1)
                    
        return unique_results

if __name__ == "__main__":
    # Test
    finder = DatasetFinder()
    res = finder.search("Ibukota Baru Indonesia")
    print(f"Got {len(res)} results.")
    for r in res[:5]:
        print(f"- {r[:100]}...")
