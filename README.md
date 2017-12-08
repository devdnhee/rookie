# Rookie

## Requirements
- Python2.7
- pip

## Installation
`git clone https://github.com/devdnhee/rookie/`

Go to the directory and execute

`sudo python setup.py install`

Afterwards, download and extract the zipfile from https://drive.google.com/open?id=1eAZUC8NeWIk4l1UU8HVeL4HjnC0g9hsY and store the datasets (.epd file format) in the folder `dataset/`.

Test by executing (make sure you have at least 3 cpus available)
`python sim.py test -N 100 -I 2 -d 1`

which runs a new (small) simulation by letting the untrained engine play games against itself until it gained enough victories to learn something with SGD. It does this twice.
