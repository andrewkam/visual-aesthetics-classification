import sys
import re
import os
import requests
import billboard
import pylast
import json
import urllib.request
from urllib.error import HTTPError
from lxml import html


with open('config.json', 'r') as f:
    config = json.load(f)

LASTFM_API_KEY = config['LASTFM']['API_KEY']
LASTFM_API_SECRET = config['LASTFM']['API_SECRET']

network = pylast.LastFMNetwork(api_key=LASTFM_API_KEY,
                               api_secret=LASTFM_API_SECRET)


def getChart(genre):
    chart = billboard.ChartData(genre)
    artists = []

    for song in chart:
        # rank = song.rank
        artist_name = re.sub('Featuring.*| [xX] .*| & .*', '', song.artist)
        artist_name = artist_name.strip()

        artists.append(artist_name)

    return artists


def getImageUrls(artists):
    urls = []

    for artist_name in artists:
        search = network.search_for_artist(artist_name)
        results = search.get_next_page()
        artist = results[0]
        url_image = artist.get_url() + '/+images'
        urls.append(url_image)

    return urls


def getImages(genre, artists):
    urls = getImageUrls(artists)
    url_base = 'https://lastfm-img2.akamaized.net/i/u/avatar300s/'

    image_dir = './images/' + genre + '/'
    if not os.path.exists(image_dir):
        os.mkdir(image_dir)

    artist_count = len(artists)

    log_file = 'import_' + genre + '.txt'
    file = open(log_file, 'w')

    for i, (artist_name, url_image_list) in enumerate(zip(artists, urls)):
        print(f'({str(i+1).zfill(2)}/{str(artist_count).zfill(2)}) '
              f'{artist_name}')

        artist_name_dir = artist_name.replace(' ', '_').lower()
        artist_name_dir = artist_name_dir.replace('/', '_')

        if 'soundtrack' in artist_name_dir:
            continue

        r = requests.get(url_image_list)
        page_source = r.content
        tree = html.fromstring(page_source)
        results = tree.xpath("//img[@class='image-list-image']/@src")

        artist_image_dir = image_dir + artist_name_dir
        if not os.path.exists(artist_image_dir):
            os.mkdir(artist_image_dir)

        for i, path in enumerate(results):
            url_image = url_base + path[49:]  # + '.jpeg#' + path[49:]
            image_filename = '%s/%s-%02d.jpeg' % (artist_image_dir,
                                                  artist_name_dir,
                                                  i+1)

            file.write(image_filename + ' ' + url_image + '\n')

            if not os.path.isfile(image_filename):
                try:
                    urllib.request.urlretrieve(url_image, image_filename)
                except HTTPError as e:
                    print(image_filename, 'Error code:', e.code)
                    continue

    file.close()


def main():
    genre = sys.argv[1]

    artists = getChart(genre)  # ie. 'pop-songs'
    getImages(genre, artists)


main()
