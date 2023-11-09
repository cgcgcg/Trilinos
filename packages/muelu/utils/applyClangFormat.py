#!/usr/bin/env python3

from sys import argv
import requests
from tempfile import NamedTemporaryFile
from subprocess import Popen
from pathlib import Path

assert len(argv) == 2

commentID = int(argv[1])

url = "https://api.github.com/repos/cgcgcg/Trilinos/issues/comments/" + str(commentID)
headers = {"Accept": "application/vnd.github.inertia-preview+json"}
response = requests.get(url, headers=headers)

response_body = response.json()['body']
patch = response_body.splitlines()[1:-2]

try:
    tf = NamedTemporaryFile('w', delete=False)
    for line in patch:
        tf.write(line+'\n')
    tf.close()

    Popen(['git', 'apply', tf.name]).wait()

finally:
    Path(tf.name).unlink()
