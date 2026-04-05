import json
import os

def main():
    # Define test queries covering different topics that would likely appear in Wikipedia
    queries = [
        {"id": "q1", "text": "What is the capital of France?"},
        {"id": "q2", "text": "How does photosynthesis work?"},
        {"id": "q3", "text": "Who wrote Romeo and Juliet?"},
        {"id": "q4", "text": "History of artificial intelligence"},
        {"id": "q5", "text": "Causes of World War II"}
    ]
    
    # Define output path relative to the project root
    output_dir = os.path.join("data", "raw")
    output_path = os.path.join(output_dir, "queries.json")
    
    # Ensure directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the queries as a JSON file
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(queries, f, indent=4)
        
    print(f"Successfully saved {len(queries)} test queries to {output_path}")

if __name__ == "__main__":
    main()
