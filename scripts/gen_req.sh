#!/bin/sh
uv pip compile requirements.in --universal --no-deps --no-strip-extras --no-annotate | sed "s/==/>=/g" > requirements.txt
