import httpx
import os
import json  # To load the JSON file

def extract_raw_urls(response_json, skip_premium=True):
    urls = []
    for image_data in response_json['results']:
        if skip_premium and image_data.get('premium', False):
            continue
        raw_url = image_data['urls']['raw']
        trimmed_url = raw_url.split('?')[0]
        urls.append((image_data['id'], trimmed_url))
    return urls

def download_images(keyword, urls):
    directory = f"./{keyword}"
    if not os.path.exists(directory):
        os.makedirs(directory)

    print(f"Starting download of {len(urls)} images for {keyword}...")
    for idx, (image_id, url) in enumerate(urls, start=1):
        image_response = httpx.get(url)
        if image_response.status_code == 200:
            image_path = os.path.join(directory, f"{image_id}.jpg")
            with open(image_path, 'wb') as image_file:
                image_file.write(image_response.content)
            print(f"{idx}. Downloaded {image_id} to {image_path}")
        else:
            print(f"{idx}. Failed to download image {image_id}")

# Example usage
if __name__ == "__main__":
    keyword = "art"
    
    # Step 1: Load the JSON response from the file
    json_file_path = f"./{keyword}.json"  # Replace with the actual path to your JSON file
    with open(json_file_path, "r", encoding="utf-8") as f:  # Specify utf-8 encoding
        all_results = json.load(f)  # Load the JSON data into a Python list

    if all_results:  # Ensure we have results
        urls = extract_raw_urls(all_results)  # Extract URLs, skipping premium images
        if urls:
            # Limit the total number of images to 100
            urls = urls[:100]
            print(f"Total images to be downloaded (excluding premium ones): {len(urls)}")
            download_images(keyword, urls)
        else:
            print("No non-premium images found for download.")
    else:
        print("No results found in the JSON file.")
