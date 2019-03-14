#!/bin/bash

./out/lightning > dmp/cur && python 3dgrid.py dmp/cur && feh dmp/3d_out.png
