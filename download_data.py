import os
import csv
import requests
from tqdm import tqdm
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor
import threading

BASE_URL = "https://qipedc.moet.gov.vn"
videos_dir = "Dataset/Videos"
text_dir = "Dataset/Text"
os.makedirs(videos_dir, exist_ok=True)  
os.makedirs(text_dir, exist_ok=True)
csv_path = os.path.join(text_dir, "label.csv")
csv_lock = threading.Lock()


def fetch_dictionary_entries(group_size=4000, query_text=""):
    try:
        response = requests.post(
            f"{BASE_URL}/dictionary/getAll",
            data={"group": group_size, "text": query_text},
            timeout=60,
        )
        response.raise_for_status()
        payload = response.json()
    except Exception as exc:
        print(f"Failed to fetch dictionary data: {exc}")
        return []

    entries = payload.get("data", [])
    videos = []
    for entry in entries:
        video_id = entry.get("_id")
        label = entry.get("word")
        if not video_id or not label:
            continue
        video_url = f"{BASE_URL}/videos/{video_id}.mp4"
        videos.append({"label": label, "video_url": video_url})
    return videos

def csv_init():
    if not os.path.exists(csv_path):
        with open(csv_path, 'w', newline='', encoding= 'utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(["ID", "VIDEO", "LABEL"])

def add_to_csv(id, video, label):
    with csv_lock:
        with open(csv_path, 'a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow([id, video, label])

def download_video(video_data):
    video_url = video_data.get('video_url')
    label = video_data.get('label')
    filename = os.path.basename(urlparse(video_url).path)
    output_path = os.path.join(videos_dir, filename)
    if os.path.exists(output_path):
        print(f"Skip: {filename}")
        return
    try:
        print(f"Downloading: {filename}")
        response = requests.get(video_url, stream=True)
        response.raise_for_status()
        total_size = int(response.headers.get('content-length', 0))
        with open(output_path, 'wb') as file, tqdm(
            desc=f"Progess {filename}",
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
            ncols=100
        ) as bar:
            for data in response.iter_content(chunk_size=8192):
                size = file.write(data)
                bar.update(size)

        id = sum(1 for _ in open(csv_path, encoding='utf-8'))
        add_to_csv(id, filename, label)                  
        print(f"Completed: {filename}")
        print(f"Updated label.csv: {label}")

    except Exception as e:
        print(f"Error{filename}: {str(e)}")
        if os.path.exists(output_path):
            os.remove(output_path)  

def crawl_videos():
    print("CRAWLING VIDEOS")
    videos = fetch_dictionary_entries()
    if not videos:
        print("Unable to collect videos from API")
    return videos

def main():    
    videos = crawl_videos()
    if videos:
        print(f"Found {len(videos)} videos\n")
    
    print("STARTING DOWNLOAD VIDEOS")
    csv_init()
    
    if not videos:
        print("Videos not found")
        return
        
    with ThreadPoolExecutor(max_workers=3) as executor:
        executor.map(download_video, videos)
        
    print(f"DOWNLOAD COMPLETED {videos_dir}")

if __name__ == "__main__":
    main()
