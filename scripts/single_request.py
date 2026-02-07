import json
import os
import time

import requests

# Updated configuration for external model testing
API_URL = ("YOUR_URL")
MODEL_ID = "gpt-oss-120b"
INPUT_FILE = "dummy_input.md"
OUTPUT_DIR = f"result_{MODEL_ID}"

# Optional: Add your API token if security is enabled on the server
API_TOKEN = "XXX"

USER_PROFILE = {
    "skills": [
        "Machine Learning",
        "Domain-Driven Design (DDD)",
        "Data Science",
        "Spring Boot",
        "Java",
        "Python",
        "Hibernate",
        "PostgreSQL",
        "R (Programming Language)",
        "Project Management"
    ],
    "experiences": [
        "4 Years of Software development"
    ],
    "qualifications": [
        "Domain Driven Design",
        "Bsc. Wirtschaftsinformatik"
    ]
}

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

headers = {
    "Content-Type": "application/json"
}
if API_TOKEN:
    headers["Authorization"] = f"Bearer {API_TOKEN}"

print(f"Starting evaluation using model: {MODEL_ID}")
print(f"Targeting API: {API_URL}")

if not os.path.exists(INPUT_FILE):
    print(f"Input file not found: {INPUT_FILE}")
    exit()

filename = os.path.basename(INPUT_FILE)

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    markdown_content = f.read()

payload = {
    "modelId": MODEL_ID,
    "inputText": markdown_content,
    "userProfile": USER_PROFILE
}

start_time = time.time()
try:
    response = requests.post(API_URL, headers=headers, data=json.dumps(payload), verify=False)
    end_time = time.time()
    processing_time_in_seconds = end_time - start_time

    print(f"Processing {filename}...")
    print("Status code:", response.status_code)

    if response.status_code == 200:
        result = {
            "modelId": MODEL_ID,
            "inputText": markdown_content,
            "result": response.json(),
            "processingTimeInSeconds": processing_time_in_seconds
        }
        print(json.dumps(result, indent=4, ensure_ascii=False))

        output_filename = os.path.splitext(filename)[0] + "_test.json"
        output_filepath = os.path.join(OUTPUT_DIR, output_filename)
        with open(output_filepath, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=4)
        print(f"\nResult saved to {output_filepath}")
    else:
        print("Error:", response.text)
except Exception as e:
    print(f"Critical error: {e}")
