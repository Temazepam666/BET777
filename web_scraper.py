import pandas as pd
import aiohttp
import asyncio
from bs4 import BeautifulSoup

# Asynchronous function to fetch page data
async def fetch(session, url):
    async with session.get(url) as response:
        return await response.text()

# Scraping data asynchronously from multiple pages
async def scrape_data():
    base_url = "https://example-sports-website.com/stats?page="
    all_data = []
    
    async with aiohttp.ClientSession() as session:
        tasks = []
        for page in range(1, 5):  # Define page range or customize it dynamically
            url = base_url + str(page)
            tasks.append(fetch(session, url))
        
        # Gather responses asynchronously
        pages_content = await asyncio.gather(*tasks)
        
        for content in pages_content:
            soup = BeautifulSoup(content, 'html.parser')
            for entry in soup.find_all('div', class_='stats-entry'):
                feature1 = entry.find('span', class_='feature1').text
                feature2 = entry.find('span', class_='feature2').text
                outcome = entry.find('span', class_='outcome').text
                all_data.append([feature1, feature2, outcome])
    
    # Convert to DataFrame and save
    df = pd.DataFrame(all_data, columns=['feature1', 'feature2', 'outcome'])
    df.to_csv('data/sports_data.csv', index=False)
    return df

# Run asynchronous scraper
def run_scraper():
    asyncio.run(scrape_data())
