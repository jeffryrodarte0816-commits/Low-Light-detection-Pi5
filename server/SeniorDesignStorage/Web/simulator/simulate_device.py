import os
import time
import random
import argparse
import requests

DEFAULT_URL = os.getenv('SERVER_URL', 'http://localhost:8000')


def random_walk(start, step=0.0001):
    lat, lon = start
    lat += random.uniform(-step, step)
    lon += random.uniform(-step, step)
    return lat, lon


def run(device_id, interval, start_lat, start_lon):
    pos = (start_lat, start_lon)
    url = DEFAULT_URL.rstrip('/') + '/telemetry'
    print(f"Posting to {url} as {device_id}")
    try:
        while True:
            pos = random_walk(pos, step=0.0005)
            payload = {
                'device_id': device_id,
                'latitude': pos[0],
                'longitude': pos[1]
            }
            try:
                r = requests.post(url, json=payload, timeout=5)
                print(time.strftime('%Y-%m-%d %H:%M:%S'), r.status_code, r.text)
            except Exception as e:
                print('post error', e)
            time.sleep(interval)
    except KeyboardInterrupt:
        print('Stopped')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--device', default='jetson-nano-1')
    p.add_argument('--interval', type=float, default=2.0)
    p.add_argument('--lat', type=float, default=37.4219999)
    p.add_argument('--lon', type=float, default=-122.0840575)
    args = p.parse_args()
    run(args.device, args.interval, args.lat, args.lon)
