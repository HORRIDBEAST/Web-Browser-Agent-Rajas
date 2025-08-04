#!/usr/bin/env python3

"""
Web Browser Query Agent - CLI Implementation with Fixed Google Scraping
A complete implementation with query validation, similarity matching, web scraping, and caching.
"""

import os
from dotenv import load_dotenv
load_dotenv()

import json
import asyncio
import sqlite3
import hashlib
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import random

import click
import numpy as np
import requests
from playwright.async_api import async_playwright
from bs4 import BeautifulSoup
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class QueryResult:
    query: str
    results: List[Dict]
    summary: str
    timestamp: datetime
    similarity_score: float = 0.0


class QueryValidator:
    """Validates if a query is suitable for web search"""

    def __init__(self):
        self.invalid_patterns = [
            # Commands
            r'\b(walk|add|delete|remove|create|make|do|execute)\b.*\b(my|the|to)\b',
            # Too short or incoherent
            r'^.{1,3}$',
            # Personal commands
            r'\b(remind me|set alarm|call|text|email)\b'
        ]

    def is_valid(self, query: str) -> Tuple[bool, str]:
        """Validate query and return (is_valid, reason)"""
        if not query or len(query.strip()) < 3:
            return False, "Query too short"

        # Check for question words or search intent
        search_indicators = ['what', 'where', 'when', 'how', 'why', 'best', 'top', 'find', 'search']
        query_lower = query.lower()

        # Simple heuristic: if it contains search indicators or is longer than 10 chars, likely valid
        has_search_intent = any(word in query_lower for word in search_indicators)
        is_reasonable_length = len(query.strip()) > 10

        # Check for obvious commands
        command_words = ['walk', 'add to', 'delete', 'create', 'make me', 'do this']
        has_command = any(cmd in query_lower for cmd in command_words)

        if has_command and not has_search_intent:
            return False, "Appears to be a command, not a search query"

        if has_search_intent or is_reasonable_length:
            return True, "Valid search query"

        return False, "Does not appear to be a search query"


class SimilarityEngine:
    """Handles query similarity matching using sentence embeddings"""

    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.similarity_threshold = 0.75

    def get_embedding(self, text: str) -> np.ndarray:
        """Get sentence embedding for text"""
        return self.model.encode([text])[0]

    def find_similar_query(self, query: str, stored_queries: List[Dict]) -> Optional[Dict]:
        """Find similar query from stored queries"""
        if not stored_queries:
            return None

        query_embedding = self.get_embedding(query)
        best_match = None
        best_score = 0

        for stored in stored_queries:
            stored_embedding = np.array(stored['embedding'])
            similarity = cosine_similarity([query_embedding], [stored_embedding])[0][0]

            if similarity > self.similarity_threshold and similarity > best_score:
                best_score = similarity
                best_match = stored
                best_match['similarity_score'] = similarity

        return best_match

class WebScraper:
    """Handles web scraping using Playwright with Google as primary engine"""

    def __init__(self):
        self.max_results = 5
        self.user_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36"
        ]
        
        # Smart selector performance tracking
        self.selector_performance = {
            'div[data-ved] a[href^="http"]:has(h3)': {'success': 0, 'total': 0, 'avg_results': 0},
            'a[href^="http"]:has(h3)': {'success': 0, 'total': 0, 'avg_results': 0},
            '.yuRUbf h3 a': {'success': 0, 'total': 0, 'avg_results': 0},
            '.tF2Cxc h3 a': {'success': 0, 'total': 0, 'avg_results': 0},
            'div[data-ved] h3 a': {'success': 0, 'total': 0, 'avg_results': 0},
            'h3 a[href*="http"]': {'success': 0, 'total': 0, 'avg_results': 0},
            '.g h3 a': {'success': 0, 'total': 0, 'avg_results': 0},
            '.g a[href^="http"]': {'success': 0, 'total': 0, 'avg_results': 0},
            '[data-ved] a[href^="http"]': {'success': 0, 'total': 0, 'avg_results': 0}
        }

    def get_optimized_selectors(self):
        """Get selectors ordered by performance"""
        # Sort by success rate, then by average results
        sorted_selectors = sorted(
            self.selector_performance.items(),
            key=lambda x: (
                x[1]['success'] / max(x[1]['total'], 1),  # Success rate
                x[1]['avg_results']                        # Average results count
            ),
            reverse=True
        )
        
        return [selector for selector, _ in sorted_selectors]

    def update_selector_performance(self, selector: str, success: bool, result_count: int = 0):
        """Update selector performance metrics"""
        if selector in self.selector_performance:
            self.selector_performance[selector]['total'] += 1
            if success:
                self.selector_performance[selector]['success'] += 1
                self.selector_performance[selector]['avg_results'] = (
                    (self.selector_performance[selector]['avg_results'] + result_count) / 2
                )

    async def search_and_scrape(self, query: str) -> List[Dict]:
        """Search query and scrape top results with improved Google handling"""
        async with async_playwright() as p:
            browser = await p.chromium.launch(
                headless=True,
                args=[
                    '--no-sandbox',
                    '--disable-blink-features=AutomationControlled',
                    '--disable-web-security',
                    '--disable-features=VizDisplayCompositor',
                    '--disable-extensions',
                    '--disable-plugins',
                    '--disable-images',
                    '--disable-javascript',
                    '--no-first-run'
                ]
            )
            
            context = await browser.new_context(
                user_agent=random.choice(self.user_agents),
                viewport={"width": 1366, "height": 768},
                locale='en-US',
                timezone_id='America/New_York',
                extra_http_headers={
                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                    'Accept-Language': 'en-US,en;q=0.5',
                    'Accept-Encoding': 'gzip, deflate',
                    'DNT': '1',
                    'Connection': 'keep-alive',
                    'Upgrade-Insecure-Requests': '1',
                }
            )
            
            # Add stealth settings
            await context.add_init_script("""
                Object.defineProperty(navigator, 'webdriver', {
                    get: () => undefined,
                });
                
                Object.defineProperty(navigator, 'plugins', {
                    get: () => [1, 2, 3, 4, 5],
                });
                
                Object.defineProperty(navigator, 'languages', {
                    get: () => ['en-US', 'en'],
                });
                
                window.chrome = {
                    runtime: {},
                };
            """)
            
            page = await context.new_page()
            
            try:
                # Try Google first with improved approach
                results = await self._search_google_improved(page, query)
                if results:
                    print(f"✓ Found {len(results)} Google results")
                    return results
                
                # Fallback to Bing
                print("Google failed, trying Bing...")
                results = await self._search_bing(page, query)
                if results:
                    print(f"✓ Found {len(results)} Bing results")
                    return results
                
                print("All search engines failed.")
                return []
                
            except Exception as e:
                print(f"Error in search_and_scrape: {e}")
                return []
            finally:
                await context.close()
                await browser.close()

    async def _search_google_improved(self, page, query: str) -> List[Dict]:
        """Improved Google search with better anti-detection"""
        try:
            # Navigate to Google homepage first
            print("Navigating to Google...")
            await page.goto("https://www.google.com", timeout=30000, wait_until="networkidle")
            await page.wait_for_timeout(random.randint(2000, 4000))
            
            # Check if there's a consent dialog and handle it
            try:
                consent_button = page.locator('button:has-text("Accept all"), button:has-text("I agree"), button:has-text("Accept")')
                if await consent_button.count() > 0:
                    await consent_button.first.click()
                    await page.wait_for_timeout(2000)
            except:
                pass
            
            # Find and use the search box
            search_selectors = [
                'textarea[name="q"]',
                'input[name="q"]',
                'input[title="Search"]',
                '#searchbox input',
                '.gLFyf'
            ]
            
            search_box = None
            for selector in search_selectors:
                try:
                    search_box = await page.wait_for_selector(selector, timeout=5000)
                    if search_box:
                        break
                except:
                    continue
            
            if not search_box:
                print("Could not find Google search box")
                return []
            
            # Type the query with human-like behavior
            await search_box.click()
            await page.wait_for_timeout(random.randint(500, 1000))
            
            # Clear any existing text and type query
            await search_box.fill("")
            await page.wait_for_timeout(200)
            for char in query:
                await search_box.type(char)
                await page.wait_for_timeout(random.randint(50, 150))
            
            await page.wait_for_timeout(random.randint(1000, 2000))
            
            # Submit search
            await page.keyboard.press("Enter")
            await page.wait_for_timeout(3000)
            
            # Wait for results to load
            await page.wait_for_load_state("networkidle", timeout=30000)
            
            # Use optimized selector order based on past performance
            result_selectors = self.get_optimized_selectors()
            print(f"Using optimized selector order based on performance")
            
            results = []
            selector_success_count = {}
            
            for i, selector in enumerate(result_selectors):
                try:
                    # Reduce timeout for selectors we know are less likely to work
                    timeout = 5000 if i < 2 else 3000  # First 2 get more time
                    
                    await page.wait_for_selector(selector, timeout=timeout)
                    result_links = await page.query_selector_all(selector)
                    
                    if result_links:
                        print(f"✓ Found {len(result_links)} results with selector: {selector}")
                        selector_success_count[selector] = len(result_links)
                        
                        # Update performance tracking
                        self.update_selector_performance(selector, True, len(result_links))
                        
                        # Process results from this selector
                        processed_count = 0
                        for j, link in enumerate(result_links[:self.max_results]):
                            try:
                                url = await link.get_attribute('href')
                                title_element = await link.query_selector('h3') or link
                                title = await title_element.inner_text() if title_element else "No title"
                                
                                if url and title and url.startswith('http'):
                                    # Skip Google's own pages
                                    if any(domain in url.lower() for domain in ['google.com', 'youtube.com', 'maps.google']):
                                        continue
                                    
                                    print(f"Processing result {len(results)+1}: {title[:50]}...")
                                    content = await self._scrape_page_content_safe(page.context, url)
                                    if not content or len(content) < 50:
                                        content = self._scrape_with_requests(url)
                                    
                                    results.append({
                                        'title': title.strip(),
                                        'url': url,
                                        'content': content[:1000] if content else "Content not available",
                                        'rank': len(results) + 1,
                                        'selector_used': selector  # Track which selector worked
                                    })
                                    processed_count += 1
                                    
                                    if len(results) >= self.max_results:
                                        break
                                        
                            except Exception as e:
                                print(f"Error processing result {j+1}: {e}")
                                continue
                        
                        # If we got good results from this selector, we can stop trying others
                        if processed_count >= 3:  # Got at least 3 good results
                            print(f"✓ Selector '{selector}' provided {processed_count} good results, stopping search")
                            break
                            
                except Exception as e:
                    print(f"Selector {selector} failed: {e}")
                    # Update performance tracking for failed selector
                    self.update_selector_performance(selector, False, 0)
                    continue
            
            # Log selector performance for future optimization
            if selector_success_count:
                print(f"Selector performance: {selector_success_count}")
            
            if not results:
                print("No Google results found with any selector")
            
            return results
            
        except Exception as e:
            print(f"Google search failed: {e}")
            return []

    async def _search_bing(self, page, query: str) -> List[Dict]:
        """Enhanced Bing search as fallback"""
        try:
            search_url = f"https://www.bing.com/search?q={query.replace(' ', '+')}"
            print(f"Searching Bing: {search_url}")
            
            await page.goto(search_url, timeout=30000, wait_until="networkidle")
            await page.wait_for_timeout(3000)
            
            # Handle cookie consent if present
            try:
                accept_button = page.locator('#bnp_btn_accept, button:has-text("Accept")')
                if await accept_button.count() > 0:
                    await accept_button.first.click()
                    await page.wait_for_timeout(2000)
            except:
                pass
            
            selectors_to_try = [
                '.b_algo h2 a',
                '.b_title a',
                'h2 a[href^="http"]',
                '.b_algo a[href^="http"]'
            ]
            
            results = []
            for selector in selectors_to_try:
                try:
                    await page.wait_for_selector(selector, timeout=10000)
                    result_links = await page.query_selector_all(selector)
                    
                    if result_links:
                        print(f"Found {len(result_links)} Bing results with selector: {selector}")
                        
                        for i, link in enumerate(result_links[:self.max_results]):
                            try:
                                url = await link.get_attribute('href')
                                title = await link.inner_text()
                                
                                if url and title and url.startswith('http'):
                                    # Skip Microsoft's own pages and ads
                                    if any(domain in url.lower() for domain in ['microsoft.com', 'msn.com', 'bing.com']):
                                        continue
                                    
                                    print(f"Processing Bing result {len(results)+1}: {title[:50]}...")
                                    content = await self._scrape_page_content_safe(page.context, url)
                                    if not content or len(content) < 50:
                                        content = self._scrape_with_requests(url)
                                    
                                    results.append({
                                        'title': title.strip(),
                                        'url': url,
                                        'content': content[:1000] if content else "Content not available",
                                        'rank': len(results) + 1
                                    })
                                    
                                    if len(results) >= self.max_results:
                                        break
                                        
                            except Exception as e:
                                print(f"Error processing Bing result {i+1}: {e}")
                                continue
                        
                        if results:
                            return results
                            
                except Exception as e:
                    print(f"Bing selector {selector} failed: {e}")
                    continue
            
            return results
            
        except Exception as e:
            print(f"Bing search failed: {e}")
            return []

    async def _scrape_page_content_safe(self, context, url: str) -> str:
        """Scrape content with proper error handling"""
        try:
            return await self._scrape_page_content(context, url)
        except Exception as e:
            print(f"Playwright scraping failed for {url}: {e}")
            return ""

    def _scrape_with_requests(self, url: str) -> str:
        """Fallback scraping using requests library"""
        try:
            headers = {
                'User-Agent': random.choice(self.user_agents),
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate',
                'Connection': 'keep-alive',
                'DNT': '1',
                'Upgrade-Insecure-Requests': '1'
            }
            
            response = requests.get(url, headers=headers, timeout=15, allow_redirects=True)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove unwanted elements
            for element in soup(['script', 'style', 'nav', 'header', 'footer', 'aside', 'iframe', 'noscript']):
                element.decompose()
            
            # Try to find main content
            main_content = None
            content_selectors = [
                'main', 'article', '.content', '.main-content', '#content', 
                '.post-content', '.entry-content', '.article-content', 
                '.page-content', '[role="main"]'
            ]
            
            for selector in content_selectors:
                main_element = soup.select_one(selector)
                if main_element:
                    main_content = main_element.get_text(strip=True, separator=' ')
                    break
            
            if not main_content:
                # Fallback to body content
                body = soup.find('body')
                if body:
                    main_content = body.get_text(strip=True, separator=' ')
                else:
                    main_content = soup.get_text(strip=True, separator=' ')
            
            # Clean up the content
            lines = main_content.split('\n')
            cleaned_lines = []
            for line in lines:
                line = line.strip()
                if line and len(line) > 10 and not line.startswith(('©', 'Cookie', 'Privacy')):
                    cleaned_lines.append(line)
            
            content = ' '.join(cleaned_lines)
            
            if len(content) < 50:
                return f"Limited content available from {url}"
            
            return content[:3000]
            
        except requests.RequestException as e:
            print(f"Requests scraping failed for {url}: {e}")
            return f"Failed to retrieve content from {url}"
        except Exception as e:
            print(f"General scraping error for {url}: {e}")
            return f"Error processing content from {url}"

    async def _scrape_page_content(self, context, url: str) -> str:
        """Scrape content from a specific URL using browser context"""
        page = None
        try:
            page = await context.new_page()
            await page.goto(url, timeout=20000, wait_until='domcontentloaded')
            await page.wait_for_timeout(2000)
            
            html_content = await page.content()
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Remove unwanted elements
            for script in soup(["script", "style", "nav", "header", "footer", "aside", "iframe"]):
                script.extract()
            
            # Try to find main content
            main_content = None
            content_selectors = [
                'main', 'article', '.content', '.main-content', '#content', 
                '.post-content', '.entry-content', '.article-content'
            ]
            
            for selector in content_selectors:
                main_element = soup.select_one(selector)
                if main_element:
                    main_content = main_element.get_text()
                    break
            
            if not main_content:
                main_content = soup.get_text()
            
            # Clean up content
            lines = (line.strip() for line in main_content.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            content = '\n'.join(chunk for chunk in chunks if chunk and len(chunk) > 10)
            
            return content[:3000]
            
        except Exception as e:
            print(f"Error scraping {url}: {e}")
            return ""
        finally:
            if page:
                await page.close()


class ContentProcessor:
    """Processes and summarizes scraped content"""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv('GOOGLE_API_KEY')
        if self.api_key:
            try:
                genai.configure(api_key=self.api_key)
            except Exception as e:
                print(f"Error configuring Gemini API: {e}")
                self.api_key = None
        else:
            print("No GOOGLE_API_KEY found. Using fallback summarization.")

    def summarize_results(self, query: str, results: List[Dict]) -> str:
        """Summarize search results"""
        if not results:
            return "No results found for your query."

        # Combine all content
        combined_content = ""
        for result in results:
            combined_content += f"Source: {result['title']}\nURL: {result['url']}\n{result['content']}\n\n"
        
        if self.api_key:
            return self._summarize_with_gemini(query, combined_content)
        else:
            return self._simple_summarize(query, results)

    def _summarize_with_gemini(self, query: str, content: str) -> str:
        """Summarize using Gemini API"""
        prompt = f"""
        You are a helpful assistant that summarizes web search results.
        Based on the following content scraped from the web, provide a concise and comprehensive answer to the user's original query.

        Original Query: "{query}"

        Scraped Content:
        {content[:20000]}  # Limit to avoid token limits

        Please provide a well-structured summary that directly answers the query.
        Include specific details and cite sources when relevant.
        """
        try:
            model = genai.GenerativeModel('gemini-1.5-flash')
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            print(f"Gemini API error: {e}")
            return self._simple_summarize(query, [])

    def _simple_summarize(self, query: str, results: List[Dict]) -> str:
        """Enhanced rule-based summarization"""
        if not results:
            return "No results found."

        summary = f"Based on {len(results)} sources, here's what I found about '{query}':\n\n"
        
        useful_results = []
        for result in results:
            # Filter out results with minimal content
            if (len(result['content']) > 100 and 
                "Unable to retrieve content" not in result['content'] and
                "Failed to retrieve content" not in result['content'] and
                "Content not available" not in result['content']):
                useful_results.append(result)
        
        if not useful_results:
            # If no good content, at least provide the titles and sources
            summary += "I found these relevant sources, though detailed content wasn't fully accessible:\n\n"
            for i, result in enumerate(results, 1):
                summary += f"{i}. **{result['title']}**\n"
                summary += f"   Source: {result['url']}\n\n"
            return summary
        
        # Summarize useful results
        for i, result in enumerate(useful_results, 1):
            summary += f"**{i}. {result['title']}**\n"
            
            # Extract key sentences from content
            sentences = result['content'].split('.')
            key_sentences = []
            
            # Look for sentences that might contain key information
            query_keywords = [word.lower() for word in query.split() if len(word) > 3]
            
            for sentence in sentences[:15]:  # Check first 15 sentences
                sentence = sentence.strip()
                if (len(sentence) > 30 and 
                    any(keyword in sentence.lower() for keyword in query_keywords)):
                    key_sentences.append(sentence)
                    if len(key_sentences) >= 3:  # Limit to 3 key sentences per source
                        break
            
            if key_sentences:
                summary += f"   {'. '.join(key_sentences)}.\n"
            else:
                # Fallback to first meaningful sentences
                meaningful_sentences = [s.strip() for s in sentences[:5] if len(s.strip()) > 30]
                if meaningful_sentences:
                    summary += f"   {meaningful_sentences[0]}.\n"
                else:
                    summary += f"   Content summary not available.\n"
            
            summary += f"   🔗 Source: {result['url']}\n\n"
        
        return summary


class StorageManager:
    """Manages query and result storage"""

    def __init__(self, db_path: str = "query_agent.db"):
        self.db_path = db_path
        self._init_database()
        self.similarity_engine = SimilarityEngine()

    def _init_database(self):
        """Initialize SQLite database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS queries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query TEXT NOT NULL,
                query_hash TEXT UNIQUE,
                embedding BLOB,
                results TEXT,
                summary TEXT,
                timestamp DATETIME,
                access_count INTEGER DEFAULT 1
            )
        ''')
        conn.commit()
        conn.close()

    def store_query_result(self, query: str, results: List[Dict], summary: str):
        """Store query result with embedding"""
        query_hash = hashlib.md5(query.lower().encode()).hexdigest()
        embedding = self.similarity_engine.get_embedding(query)
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO queries
            (query, query_hash, embedding, results, summary, timestamp, access_count)
            VALUES (?, ?, ?, ?, ?, ?, 1)
        ''', (
            query,
            query_hash,
            embedding.tobytes(),
            json.dumps(results),
            summary,
            datetime.now().isoformat()
        ))
        conn.commit()
        conn.close()

    def get_stored_queries(self) -> List[Dict]:
        """Get all stored queries with embeddings"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM queries ORDER BY timestamp DESC')
        rows = cursor.fetchall()
        conn.close()
        
        queries = []
        for row in rows:
            embedding = np.frombuffer(row[3], dtype=np.float32)
            queries.append({
                'id': row[0],
                'query': row[1],
                'query_hash': row[2],
                'embedding': embedding,
                'results': json.loads(row[4]),
                'summary': row[5],
                'timestamp': row[6],
                'access_count': row[7]
            })
        return queries

    def update_access_count(self, query_id: int):
        """Update access count for cached result"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('UPDATE queries SET access_count = access_count + 1 WHERE id = ?', (query_id,))
        conn.commit()
        conn.close()


class WebBrowserQueryAgent:
    """Main query agent orchestrating all components"""

    def __init__(self, api_key: Optional[str] = None):
        self.validator = QueryValidator()
        self.similarity_engine = SimilarityEngine()
        self.scraper = WebScraper()
        self.processor = ContentProcessor(api_key)
        self.storage = StorageManager()

    async def process_query(self, query: str) -> str:
        """Main query processing pipeline"""
        print(f"Processing query: '{query}'")

        # Validate query
        is_valid, reason = self.validator.is_valid(query)
        if not is_valid:
            return f"This is not a valid query. Reason: {reason}"
        print("✓ Query validated")

        # Check for similar queries
        stored_queries = self.storage.get_stored_queries()
        similar_query = self.similarity_engine.find_similar_query(query, stored_queries)
        
        if similar_query:
            print(f"✓ Found similar query: '{similar_query['query']}' (similarity: {similar_query['similarity_score']:.2f})")
            self.storage.update_access_count(similar_query['id'])
            return f"[CACHED RESULT - Similar to: '{similar_query['query']}']\n\n{similar_query['summary']}"
        
        print("✓ No similar queries found, performing web search")

        try:
            # Perform web search and scraping
            results = await self.scraper.search_and_scrape(query)
            print(f"✓ Scraped {len(results)} results")
            
            if not results:
                return "No search results found for your query. This might be due to search engine blocking or connectivity issues."

            # Summarize results
            summary = self.processor.summarize_results(query, results)
            print("✓ Results summarized")

            # Store for future queries
            self.storage.store_query_result(query, results, summary)
            print("✓ Results cached for future queries")
            
            return summary
            
        except Exception as e:
            return f"Error processing query: {str(e)}"


# CLI Interface
@click.group()
def cli():
    """Web Browser Query Agent - Intelligent web search with caching"""
    pass


@cli.command()
@click.option('--api-key', help='Gemini API key for better summarization')
def interactive(api_key):
    """Start interactive query session"""
    agent = WebBrowserQueryAgent(api_key)
    print("🔍 Web Browser Query Agent")
    print("Enter your queries (type 'quit' to exit)")
    print("-" * 50)
    
    while True:
        try:
            query = input("\n> ").strip()
            if query.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            if not query:
                continue
            
            print("\nProcessing...")
            result = asyncio.run(agent.process_query(query))
            print("\n" + "="*50)
            print(result)
            print("="*50)
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


@cli.command()
@click.argument('query')
@click.option('--api-key', help='Gemini API key for better summarization')
def search(query, api_key):
    """Search for a single query"""
    agent = WebBrowserQueryAgent(api_key)
    result = asyncio.run(agent.process_query(query))
    print(result)


@cli.command()
def stats():
    """Show query statistics"""
    storage = StorageManager()
    queries = storage.get_stored_queries()
    
    if not queries:
        print("No queries stored yet.")
        return
    
    print(f"\n📊 Query Statistics")
    print(f"Total queries: {len(queries)}")
    print(f"Most recent: {queries[0]['query']}")
    
    top_queries = sorted(queries, key=lambda x: x['access_count'], reverse=True)[:5]
    print(f"\nTop accessed queries:")
    for i, q in enumerate(top_queries, 1):
        print(f"{i}. {q['query']} (accessed {q['access_count']} times)")


if __name__ == '__main__':
    cli()