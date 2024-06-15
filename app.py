# for testing you can use : curl -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" --data-binary @file name.json -v
# Set your Google Cloud credentials environment variable 

from flask import Flask, request, jsonify
import subprocess
import json
import os
import csv

app = Flask(__name__)

@app.route('/scrap', methods=['POST'])
def scrap():
    try:
        url = request.json.get('url')

        write_json_data([url])

        os.chdir('articlescraper')
        command = ['scrapy', 'crawl', 'article', '-O', 'article.json']
        subprocess.run(command, capture_output=True, text=True)

        os.chdir('..')

        article_data = read_json_data('articlescraper/article.json')[0]

        # Read data from CSV file
        with open("./articles.csv", "r", newline="") as csvfile:
            reader = csv.reader(csvfile)
            data = list(reader)

        data.append([0, article_data.get("title"), article_data.get("platform"), article_data.get("article_text")])
        del data[1]
        # Modify "id" column in each row
        for i, row in enumerate(data):
            if i == 0:  # Skip header row (if it exists)
                continue
            row[0] = i

        # Write updated data to CSV file
        with open("./articles.csv", "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(data)

        command = ['python3', 'feature_extraction.py']
        subprocess.run(command, capture_output=True, text=True)

        processed_data = read_json_data('./articles.json').get("instances")[-1]

        return jsonify(processed_data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
# Function to read JSON data
def read_json_data(url):
  try:
    with open(url, "r") as f:
      return json.load(f)
  except FileNotFoundError:
    print(f"Error: File '{url}' not found.")
    return None

# Function to write JSON data
def write_json_data(data):
  try:
    with open('./urls.json', "w") as f:
      json.dump(data, f, indent=4)  # Add indentation for readability (optional)
  except TypeError:
    print("Error: Data must be a valid JSON serializable object.")
    
if __name__ == '__main__':
    app.run(debug=True)