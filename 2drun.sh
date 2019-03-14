#!/bin/bash

./out/lightning2d > dmp/cur && python 2dgrid.py dmp/cur && feh dmp/2d_out.png
