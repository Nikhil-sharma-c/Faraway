import os
import io
import time
import hashlib
import re
import requests
import html
import json
from urllib.parse import urlsplit, urlunsplit

from PIL import Image
from tqdm import tqdm


# ============================================================
# CONFIG
# ============================================================

OUTPUT_DIR = "ExamCheatingDataset/candidates"

TARGETS = {
    "cheating": [
        "student secretly using mobile phone during exam hall",
        "student reading hidden cheat notes under desk during examination",
        "student copying answers from neighbor in exam room",
        "student caught cheating with notes during school examination",
        "CCTV security camera footage student cheating exam hall",
        "exam invigilator catching student cheating with phone",
        "student looking at unauthorized notes during test classroom",
        "student writing answers from hidden paper during exam",
        "surveillance photo cheating student examination center",
        "student using smartwatch to cheat during examination"
    ],

    "giving object": [
        "student secretly passing answer paper to another student in exam hall",
        "student handing cheating notes to classmate during examination",
        "CCTV footage students exchanging paper during exam",
        "student passing mobile phone to another student during test",
        "students sharing unauthorized notes inside examination room",
        "exam invigilator sees students passing object during examination"
    ]
}

IMAGES_PER_QUERY = 100
PEXELS_API_KEY = os.getenv("PEXELS_API_KEY")
UNSPLASH_ACCESS_KEY = os.getenv("UNSPLASH_ACCESS_KEY")

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 "
        "Chrome/120.0 Safari/537.36"
    )
}


# ============================================================
# CREATE DIRECTORIES
# ============================================================

for class_name in TARGETS:

    os.makedirs(
        os.path.join(
            OUTPUT_DIR,
            class_name
        ),
        exist_ok=True
    )


# ============================================================
# DOWNLOAD IMAGE
# ============================================================

def download_image(url):

    try:
        parts = urlsplit(url)
        clean_url = urlunsplit((parts.scheme, parts.netloc, parts.path, "", ""))
        response = requests.get(clean_url, headers=HEADERS, timeout=10)

        if response.status_code == 429:
            time.sleep(2)
            response = requests.get(clean_url, headers=HEADERS, timeout=10)

        if response.status_code != 200:
            return None

        image = Image.open(
            io.BytesIO(response.content)
        )

        # Convert to RGB
        image = image.convert("RGB")

        # Reject tiny images
        width, height = image.size

        if width < 200 or height < 200:
            return None

        return image

    except Exception:
        return None


# ============================================================
# SEARCH PEXELS, UNSPLASH, BING, OR OPENVERSE
# ============================================================

def search_images(query, limit=100):

    if PEXELS_API_KEY:
        try:
            response = requests.get(
                "https://api.pexels.com/v1/search",
                headers={"Authorization": PEXELS_API_KEY},
                params={"query": query, "per_page": min(limit, 80)},
                timeout=20,
            )
            response.raise_for_status()
            return [photo["src"]["large"] for photo in response.json().get("photos", [])]
        except (requests.RequestException, KeyError, TypeError, ValueError) as error:
            print(f"Pexels search failed: {query} -> {error}")
            return []

    if UNSPLASH_ACCESS_KEY:
        try:
            response = requests.get(
                "https://api.unsplash.com/search/photos",
                headers={"Authorization": f"Client-ID {UNSPLASH_ACCESS_KEY}"},
                params={"query": query, "per_page": min(limit, 30)},
                timeout=20,
            )
            response.raise_for_status()
            return [photo["urls"]["regular"] for photo in response.json().get("results", [])]
        except (requests.RequestException, KeyError, TypeError, ValueError) as error:
            print(f"Unsplash search failed: {query} -> {error}")
            return []

    # Bing's public image page provides a keyless fallback for small datasets.
    try:
        urls = []
        for offset in range(0, limit, 35):
            response = requests.get(
                "https://www.bing.com/images/search",
                headers=HEADERS,
                params={"q": query, "form": "HDRSC2", "first": offset + 1},
                timeout=20,
            )
            response.raise_for_status()
            page = html.unescape(response.text)
            for match in re.findall(r"murl&quot;:&quot;(.*?)&quot;|murl\":\"(.*?)\"", page):
                url = next((value for value in match if value), "")
                if url.startswith(("http://", "https://")):
                    urls.append(url.replace("\\/", "/"))
            if len(set(urls)) >= limit:
                break
        if urls:
            return list(dict.fromkeys(urls))[:limit]

    except (requests.RequestException, ValueError) as error:
        print(f"Bing search failed: {query} -> {error}")

    # Openverse provides a keyless public API when neither commercial image
    # API key is configured and includes openly licensed image sources.
    try:
        response = requests.get(
            "https://api.openverse.org/v1/images/",
            headers=HEADERS,
            params={
                "q": query,
                "page_size": min(limit, 100),
            },
            timeout=20,
        )
        response.raise_for_status()
        return [item["url"] for item in response.json().get("results", []) if item.get("url")][:limit]
    except (requests.RequestException, KeyError, TypeError, ValueError) as error:
        print(f"Openverse search failed: {query} -> {error}")
        return []

# ============================================================
# HASH IMAGE
# ============================================================

def image_hash(image):

    return hashlib.md5(
        image.tobytes()
    ).hexdigest()


# ============================================================
# MAIN SCRAPER
# ============================================================

def scrape_class(class_name, queries):

    output_path = os.path.join(
        OUTPUT_DIR,
        class_name
    )
    os.makedirs(output_path, exist_ok=True)

    existing_hashes = set()

    # Load existing images
    for filename in os.listdir(output_path):

        filepath = os.path.join(
            output_path,
            filename
        )

        try:

            image = Image.open(filepath)

            existing_hashes.add(
                image_hash(image)
            )

        except Exception:
            pass


    counter = len(existing_hashes)

    print("\n================================")
    print(f"CLASS: {class_name}")
    print("================================")


    for query in queries:

        print(
            f"\nSearching: {query}"
        )

        urls = search_images(
            query,
            IMAGES_PER_QUERY
        )

        print(
            f"Found {len(urls)} candidates"
        )


        for url in tqdm(urls):

            image = download_image(url)

            if image is None:
                continue

            h = image_hash(image)

            # Duplicate
            if h in existing_hashes:
                continue

            existing_hashes.add(h)

            counter += 1

            filename = (
                f"{class_name.replace(' ', '_')}"
                f"_{counter:05d}.jpg"
            )

            filepath = os.path.join(
                output_path,
                filename
            )

            os.makedirs(output_path, exist_ok=True)
            image.save(
                filepath,
                "JPEG",
                quality=90
            )

            time.sleep(0.1)


    print(
        f"\nSaved {counter} images "
        f"for {class_name}"
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    for class_name, queries in TARGETS.items():

        scrape_class(
            class_name,
            queries
        )

    print("\n================================")
    print("SCRAPING COMPLETE")
    print("================================")