# MVCTPy

This repository includes MVCTPy, a Python implementation of the ALNS method for solving the Multi-Vehicle Covering Tour Problen presented in my masters thesis: "Using Adaptive Large Neighborhood Search to Solve the Multi-Vehicle Covering Tour Problem".

The instances present in this repository are modified from the TSPLib instances
by [[1]](#1) in the way presented by [[2]](#2).

## Installation

The package is installed by cloning the repository to your machine

```bash
git clone https://github.com/jenstrolle/masters-thesis-code
``` 

## Usage

Running the script `run_instances.py` with --partial gives a number of runs of
the A-1-100-100-4 MVCTP instance according to the --runs flag. 
The instances which are run can be configured in the file. 
If the script is run without --partial, all instances are run once.

## Documentation

Documentation for MVCTPy is presented on [my
website](http://trolle.co/mvctpy/index.html).

## License

[MIT](https://choosealicense.com/licenses/mit/)


## References
<a id="1">[1]</a> 
Reinelt, G. (1991).
TSPLIB--A Traveling Salesman Problem Library
In: <it>ORSA Journal on Computing</it> 3.4, pp. 376-384

<a id="2">[2]</a> 
Hachicha, M. et al. (2000)
Heuristics for the multi-vehicle covering tour problem
In: <it>Computers & Operations Research</it> 27.1, pp. 29-42
