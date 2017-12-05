#!/bin/bash

pip install python-chess numpy tensorflow scikit-learn;

cd mychessmoves;
python setup.py install;
cd ..;

apt-get install liblzma-dev libpython2.7-dev;
pip install backports.lzma;

wget https://sites.google.com/site/gaviotachessengine/download/gaviota-1.0-linux.tar.gz;
gunzip gaviota-1.0-linux.tar.gz;
tar -xvf gaviota-1.0-linux.tar;
mkdir Gaviota;
mv gaviota-1.0-linux/gtb/* Gaviota;
rm -r gaviota-1.0-linux;
rm gaviota.tar;

mkdir Models;
mkdir dataset;
