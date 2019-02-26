import sys
import json
from os import listdir
from os.path import isfile, isdir, join
from random import shuffle
from dd_client import DD

with open('config.json', 'r') as f:
    config = json.load(f)

DD_PATH = config['IMAGEDIR']['DEEPDETECT']
LOCAL_PATH = config['IMAGEDIR']['LOCAL']
IMG_COUNT = 2000
IMG_BATCH = 50


def predict(service, chart, image_filenames):
    # setting up DD client
    host = 'localhost'
    sname = config['REPO'][service]['NAME']
    dd = DD(host)
    dd.set_return_format(dd.RETURN_PYTHON)

    parameters_input = {}
    parameters_mllib = {}
    parameters_output = {"best": 5,
                         "template": "{{#body}}{{#predictions}} "
                                     "{ \"index\": {\"_index\": \"images\", \"_type\": \"img\" } }\n "
                                     "{ \"uri\": \"{{uri}}\", "
                                     "\"chart\": \"" + chart + "\", "
                                     # "\"artist\": \"" + artist + "\", "
                                     "\"categories\": [ {{#classes}} "
                                     "{ \"category\": \"{{cat}}\", "
                                     "\"score\":{{prob}} } "
                                     "{{^last}},{{/last}}{{/classes}} ] }\n "
                                     "{{/predictions}}{{/body}} \n",
                         "network": {"url": "host.docker.internal:9200/images/_bulk",
                                     "http_method": "POST"}}

    predict = dd.post_predict(sname,
                              image_filenames,
                              parameters_input,
                              parameters_mllib,
                              parameters_output)


def eval_artist(service, chart, artist):
    dd_dir = join(DD_PATH, chart, artist)
    local_dir = join(LOCAL_PATH, chart, artist)

    image_filenames = []

    for file in sorted(listdir(local_dir)):
        if isfile(join(local_dir, file)):
            image_filenames.append(join(dd_dir, file))

    predict(service, chart, image_filenames[:3])


def eval_chart(service, chart):
    chart_dir = join(LOCAL_PATH, chart)

    image_filenames = []
    artists = sorted(listdir(chart_dir))

    for artist in artists:
        if isdir(join(chart_dir, artist)):
            dd_dir = join(DD_PATH, chart, artist)
            local_dir = join(LOCAL_PATH, chart, artist)

            for file in sorted(listdir(local_dir)):
                if isfile(join(local_dir, file)):
                    image_filenames.append(join(dd_dir, file))

    shuffle(image_filenames)

    for batch_no, index_start in enumerate(range(0, IMG_COUNT, IMG_BATCH)):
        print(f'{batch_no+1}/{int(IMG_COUNT/IMG_BATCH)}')
        index_stop = index_start + IMG_BATCH
        if index_stop > IMG_COUNT:
            index_stop = IMG_COUNT

        predict(service, chart, image_filenames[index_start:index_stop])


def main():
    arg_count = len(sys.argv)

    if 3 <= arg_count <= 4:
        service = sys.argv[1]
        chart = sys.argv[2]

        if arg_count == 3:
            eval_chart(service, chart)
        elif arg_count == 4:
            artist = sys.argv[3]
            eval_artist(service, chart, artist)
    else:
        print('Usage: eval_artist_images service chart [artist]')
        exit()


main()
