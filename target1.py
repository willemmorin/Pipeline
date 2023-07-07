#!/usr/bin/env python
import math
from math import cos, sin, pi

import numpy as np
from PIL import Image, ImageDraw
import networkx as nx
import random
from collections import Counter
from itertools import combinations
import matplotlib.pyplot as plt
import sys
import wave
import math
import struct
import argparse
from itertools import *
from base64 import urlsafe_b64encode
import gzip
from io import BytesIO
import numpy as np
from rdf import Triple, URIRef, Literal
import datetime
from datetime import timedelta


""" Generate literals for all members with the same assignment. Literal values
cluster closely together. We cannot use the same values for all members because
some graph learning models merge literals with the same value into a single
node, which would add information to the graph's structure."""

def gen_string(vocab, assignment, nclusters):
    # two random words plus 10 subsequent words that are unique to this
    # cluster
    i = np.random.randint(len(vocab))
    j = np.random.randint(len(vocab))
    w = (assignment+1) * len(vocab)//(nclusters+1)
    return vocab[i] + ' ' +  vocab[j] + ' '.join([vocab[w+i] for i in range(10)])

def gen_anyURI(vocab, assignment, nclusters):
    # six subsequent words as (sub)domain that are unique to this
    # cluster, plus two random words as pages 
    i = np.random.randint(len(vocab))
    j = np.random.randint(len(vocab))
    w = (assignment+1) * len(vocab)//(nclusters+1)
    return "https://%s.org/%s#%s" % ('.'.join([vocab[w+i] for i in range(6)]),
                                     vocab[i],
                                     vocab[j])

def gen_gYear(assignment, nclusters, min_year=-2000, max_year=2000):
    return gen_integer(assignment, nclusters, min_year, max_year)

def gen_date(assignment, nclusters, min_year=1970, max_year=2000):
    # random year and day because we want to test the cyclic aspect
    # of months here. Use gYear to test signals from years
    year = np.random.randint(min_year, max_year)
    day = np.random.randint(1, 28)

    month = gen_integer(assignment, nclusters, min_value=1, max_value=12)
    # rotate months around 12-month cycle to capture non-linear relation
    month += (12/nclusters)//2
    if month > 12:
        month = month%12

    return "%d-%.2d-%.2d" % (year, month, day)

def gen_dateTime(assignment, nclusters, min_year=1970, max_year=2000):
    # random date to emphasize the time part
    year = "%d-%.2d-%.2d" % (np.random.randint(min_year, max_year),
                             np.random.randint(1, 12),
                             np.random.randint(1, 28))

    hh = gen_integer(assignment, nclusters, min_value=0, max_value=23)
    # rotate hours around 24-hour cycle to capture non-linear relation
    hh += (24/nclusters)//2
    hh = hh%24

    # random minutes and seconds as we want to focus on hours; however, the same
    # principle applies here
    mm = np.random.randint(1, 60)
    ss = np.random.randint(1, 60)

    return "%sT%.2d:%.2d:%.2d" % (year, hh, mm, ss)

def gen_integer(assignment, nclusters, min_value=-9e5, max_value=9e5):
    return int(np.round(gen_float(assignment, nclusters, min_value, max_value)))

def gen_float(assignment, nclusters, min_value=-9e5, max_value=9e5, offset=0):
    # random value around cluster centre
    rnge = (max_value-min_value)/nclusters
    mu = min_value - rnge/2 + (assignment+1) * rnge
    sigma = (rnge/2)/3  # map cluster range over 3 std devs

    value = sigma * np.random.randn() + mu + offset
    while value < min_value or value > max_value:
        value = sigma * np.random.randn() + mu + offset

    return value

def gen_boolean(assignment):
    return [True, False][assignment%2]

def gen_image(assignment, nclusters, size=(200, 200)):
    # image with white bg and with <assignment+1> black vertical lines
    # line location and number depends on cluster
    num_lines = 3 + assignment
    width = size[0]//num_lines
    im = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(im)
    for i in range(assignment+1):
        mid = (i+1)*size[0]//(assignment+2)
        draw.line([(mid, 0), (mid, size[1])], fill="black", width=width)

    # add some noise (max 1%)
    pixels = im.load()
    num_noisy_pixels = np.random.randint(1, int(.01*size[0]*size[1]))
    for _ in range(num_noisy_pixels):
        i = np.random.randint(size[0])
        j = np.random.randint(size[1])

        R = int(np.random.rand() * 255)
        G = int(np.random.rand() * 255)
        B = int(np.random.rand() * 255)
        pixels[i, j] = (R, G, B)

    return im

def gen_wktLiteral(assignment, nclusters, minimal_points=3):
    # target clusters get a unique shape 
    npoints = max(minimal_points, assignment+3)

    # points equally distributed around origin
    points = [np.array([nclusters * sin((2*pi*i)/npoints),
                        nclusters * cos((2*pi*i)/npoints)])
              for i in range(npoints)]

    # randomize by rotation around center as we want the shape to matter
    r = np.random.rand()
    a = 2*pi*r
    for point in points:
        x, y = point
        point[0] = y * sin(a) + x * cos(a)
        point[1] = y * cos(a) - x * sin(a)

    # centres equally distributed around origin to prevent location from
    # being a signal.
    centre_x = np.random.randn()/10
    centre_y = np.random.randn()/10

    # translate to new centres
    for point in points:
        x, y = point
        point[0] = centre_x + x
        point[1] = centre_y + y

    points_str = ["{} {}".format(x, y) for x, y in points]
    points_str.append(points_str[0])  # close polygon
    return "POLYGON ((" +\
            ", ".join(points_str) +\
            "))"

# audio generation adapted from the works by Zach Denton: https://zach.se/generate-audio-with-python/
def sine_wave(frequency=440.0, framerate=44100, amplitude=0.5):
    # generate repeating sine waves but use a lookup table instead of recalculating each time
    period = int(framerate / frequency)
    if amplitude > 1.0: amplitude = 1.0
    if amplitude < 0.0: amplitude = 0.0
    lookup_table = [float(amplitude) * math.sin(2.0*math.pi*float(frequency)*(float(i%period)/float(framerate))) for i in range(period)]
    return (lookup_table[i%period] for i in count(0))


def white_noise(amplitude=0.5):
    # generate random white noise
    
    return (float(amplitude) * random.uniform(-1, 1) for _ in count(0))

def compute_samples(channels, nsamples=None):
    # combining channels to get samples
    
    return islice(zip(*(map(sum, zip(*channel)) for channel in channels)), nsamples)

def gen_audio(assignment, framerate, amplitude, nsamples=None):
    # generate 1 second audio clip of a binaural beat (can extend for longer clips)
    
    channels = []
    for i in range(1):
        freq1 = int(1459/(assignment+1))
        if freq1<20:
            freq1 = 20
        freq2 = freq1 + 20
        channels.append((sine_wave(freq1, amplitude=0.1), white_noise(amplitude=0.001)))
        channels.append((sine_wave(freq2, amplitude=0.1), white_noise(amplitude=0.001)))

    channels = tuple(channels)    
    samples = compute_samples(channels)
    count = 0
    wav = []
    max_amplitude = 32767

    for sample in samples:
        if count> (framerate - 1):
            break
        x = struct.pack('h',int(sample[0] * max_amplitude))
        y = struct.pack('h',int(sample[1] * max_amplitude))
        samp = b''.join([x,y])
        wav.append(samp)
        count += 1
        
    waves = b''.join(wav)
     
    return waves

# A binaural beat is an auditory illusion perceived when two different pure-tone sine waves, both with frequencies lower than 1500 Hz, with less than a 40 Hz difference between them, are presented to a listener dichotically (one through each ear).