import requests
import json
import pandas as pd
from datetime import datetime
import os
import time

class FirebaseDataFetcher:
    def __init__(self):
        self.base_url = "insert-url-here" # You can get it sa mismong Firebase Account
        self.api_key = "insert-api-key-here"
        self.output_dir = r"file-save-directory-inside-quotation"
        os.makedirs(self.output_dir, exist_ok=True)
    
    def fetch_data(self):
        url = f"{self.base_url}.json?auth={self.api_key}"
        response = requests.get(url)
        
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Error fetching data: {response.status_code}")
            print(response.text)
            return None
    
    def fetch_data_by_node(self, node_name):
        url = f"{self.base_url}/{node_name}.json?auth={self.api_key}"
        response = requests.get(url)
        
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Error fetching data from node {node_name}: {response.status_code}")
            print(response.text)
            return None
    
    def save_to_json(self, data, filename):
        if not data:
            print("No data to save.")
            return
        
        filepath = os.path.join(self.output_dir, filename)
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=4)
        print(f"Data saved to {filepath}")

    def continuous_fetch(self, interval=60, filename="sensor_data.json"):
        print(f"Starting continuous data fetching. Saving to {filename}")
        print(f"Fetching every {interval} seconds. Press Ctrl+C to stop.")
        
        try:
            while True:
                data = self.fetch_data()
                if data:
                    print("\nFetched Data:")
                    print(json.dumps(data, indent=4))
                    print("\n" + "="*50 + "\n")
                    self.save_to_json(data, filename)
                time.sleep(interval)
        except KeyboardInterrupt:
            print("\nStopping continuous data fetch...")

if __name__ == "__main__":
    fetcher = FirebaseDataFetcher()
    fetcher.continuous_fetch(interval=3)

