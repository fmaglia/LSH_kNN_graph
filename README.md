# LSH_kNN_graph
[[paper](https://arxiv.org/)] [[project](http://implab.ce.unipr.it/?page_id=)] 

The proposed method allows to create an approximate kNN graph in C++ for the diffusion application.
Then the retrieval is tested and the performance are the same or better than the ones obtained on the brute-force graph, but in less time (due to the reduction in the approximate kNN graph creation).


## Datasets
The original dataset files are converted in binary through the application of a simple C++ script:
- [Oxford5k](https://drive.google.com/)
- [Oxford105k](https://drive.google.com/) 

After downloading the dat files you need to create a folder called `dataset ` and then put in the uncompressed version.

## Installation
* Requirements:
  * G++ 5.4 or greater
  * openmp
  * cblas  
* Build:
` g++ -o LSH_sparse LSH_sparse.cpp -lstdc++fs -std=c++14 -fopenmp -O2 -lcblas`

## Test

`LSH kNN (δ = 6, L = 20, th = 5000):
./LSH_sparse 6 20 oxford5k false 5000 0 ResNet50`

`multi LSH kNN graph (δ = 6, L = 20, th = 5000, 80% of multi-probe LSH):
./LSH_sparse 6 20 oxford5k false 5000 80 ResNet50`


## Results

### Oxford5k
 
 In every test, the neighborhood is setted to 1.

| Configuration        | 1           | 10  | 100 | avg retrieval time |
| :------------- |:-------------:| :-----:| :---:| :---------:|
| δ = 16, L = 25, top500 | 90.50%   | 91.57% | 91.57% | 6 msec |
| δ = 15, L = 50, top500 | 98.36%   | 99.19% | 99.19% | 18 msec |
| δ = 15, L = 50, top10k | 99.20%   | 100%   | 100%   | 18 msec |


### Oxford105k

| Configuration        | 1           | 10  | 100 | avg retrieval time |
| :------------- |:-------------:| :-----:| :---:| :---------:|
| δ = 16, L = 100, top500 | 79.80%   | 80.80% | 80.80% | 60 msec |
| δ = 16, L = 100, top10k | 97.40%   | 98.50%   | 98.50%   | 100 msec |



## Reference

<pre>@article{magliani2018efficient,
  title={Efficient Nearest Neighbors Search for Large-Scale Landmark Recognition},
  author={Magliani, Federico and Fontanini, Tomaso and Prati, Andrea},
  journal={arXiv preprint arXiv:1806.05946},
  year={2018}
}</pre>
