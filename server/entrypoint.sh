#!/bin/bash
args=$*
openscout $args &
openscout-face-engine $args &
openscout-object-engine $args &
openscout-ocr-engine $args
