import sys
import json
from os import listdir
from os.path import isfile, isdir, join
from dd_client import DD

with open('config.json', 'r') as f:
    config = json.load(f)

DD_PATH = config['IMAGEDIR']['DEEPDETECT']
LOCAL_PATH = config['IMAGEDIR']['LOCAL']


def eval_artist(service, chart, artist):
    dd_dir = join(DD_PATH, chart, artist)
    local_dir = join(LOCAL_PATH, chart, artist)

    image_filenames = []

    for file in sorted(listdir(local_dir)):
        if isfile(join(local_dir, file)):
            image_filenames.append(join(dd_dir, file))

    # setting up DD client
    host = 'localhost'
    sname = config['REPO'][service]['NAME']
    dd = DD(host)
    dd.set_return_format(dd.RETURN_PYTHON)

    parameters_input = {}
    parameters_mllib = {}
    parameters_output = {"best": 1,
                         "template": "{{#body}}{{#predictions}} "
                                     "{ \"index\": {\"_index\": \"images\", \"_type\": \"img\" } }\n "
                                     "{ \"uri\": \"{{uri}}\", "
                                     "{{#classes}} "
                                     "\"chart\": \"" + chart + "\", "
                                     "\"category\": \"{{cat}}\", "
                                     "\"score\":{{prob}} } "
                                     "{{/classes}}\n "
                                     "{{/predictions}}{{/body}} \n",
                         "network": {"url": "host.docker.internal:9200/images/_bulk",
                                     "http_method": "POST"}}

    predict = dd.post_predict(sname,
                              image_filenames,
                              parameters_input,
                              parameters_mllib,
                              parameters_output)


def eval_chart(service, chart):
    chart_dir = join(LOCAL_PATH, chart)

    for artist in sorted(listdir(chart_dir)):
        if isdir(join(chart_dir, artist)):
            print(artist)
            eval_artist(service, chart, artist)


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
