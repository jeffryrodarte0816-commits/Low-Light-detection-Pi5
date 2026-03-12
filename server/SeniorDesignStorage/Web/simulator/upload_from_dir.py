"""Upload images from a directory to the server and attach GPS coords.

Behavior:
- For each image file in the specified directory, POST to /upload with multipart form.
- If --device is provided and no --lat/--lon are provided, the script will query
  /telemetry/{device}/latest and attach those coordinates to the upload when present.
"""
import os
import time
import argparse
import requests

DEFAULT_URL = os.getenv('SERVER_URL', 'http://localhost:8000')

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}


def find_images(path):
    for fname in os.listdir(path):
        lower = fname.lower()
        if any(lower.endswith(ext) for ext in IMAGE_EXTS):
            yield os.path.join(path, fname)


def get_latest_coords(server, device):
    try:
        r = requests.get(f"{server.rstrip('/')}/telemetry/{device}/latest", timeout=5)
        if r.status_code == 200:
            j = r.json()
            return j.get('latitude'), j.get('longitude')
    except Exception:
        return None, None
    return None, None


def upload_file(server, filepath, device=None, lat=None, lon=None):
    url = f"{server.rstrip('/')}/upload"
    files = {'file': open(filepath, 'rb')}
    data = {}
    if device:
        data['device_id'] = device
    if lat is not None and lon is not None:
        data['latitude'] = str(lat)
        data['longitude'] = str(lon)
    try:
        r = requests.post(url, files=files, data=data, timeout=10)
        return r.status_code, r.text
    finally:
        files['file'].close()


def run(directory, device=None, interval=1.0, server=DEFAULT_URL, once=False, lat=None, lon=None):
    uploaded = set()
    while True:
        for path in find_images(directory):
            if path in uploaded:
                continue
            use_lat = lat
            use_lon = lon
            if (use_lat is None or use_lon is None) and device:
                fetched_lat, fetched_lon = get_latest_coords(server, device)
                if fetched_lat is not None and fetched_lon is not None:
                    use_lat, use_lon = fetched_lat, fetched_lon

            code, text = upload_file(server, path, device=device, lat=use_lat, lon=use_lon)
            print(path, code, text)
            if code == 200 or code == 201:
                uploaded.add(path)
            time.sleep(interval)
        if once:
            break
        time.sleep(1.0)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--dir', required=True, help='Directory containing images to upload')
    p.add_argument('--device', help='Device id to attach to uploads (will attempt to use latest telemetry)')
    p.add_argument('--interval', type=float, default=1.0, help='Delay between uploads')
    p.add_argument('--server', default=DEFAULT_URL, help='Server base URL')
    p.add_argument('--once', action='store_true', help='Upload once then exit')
    p.add_argument('--lat', type=float, help='Force latitude for uploads')
    p.add_argument('--lon', type=float, help='Force longitude for uploads')
    args = p.parse_args()
    run(args.dir, device=args.device, interval=args.interval, server=args.server, once=args.once, lat=args.lat, lon=args.lon)
