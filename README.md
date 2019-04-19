# LSH_kNN_graph
[[paper](https://arxiv.org/pdf/1904.08668.pdf)] [[project](http://implab.ce.unipr.it/?page_id=1006)] 

The proposed method allows to create an approximate kNN graph in C++ for the diffusion application.
Then the retrieval is tested and the performance are the same or better than the ones obtained on the brute-force graph, but in less time (due to the reduction in the approximate kNN graph creation).


## Datasets
The original dataset files are converted in binary through the application of a simple C++ script:
- [Oxford5k](https://drive.google.com/file/d/1AZo4h175YaRT3RlXEC6RUch7X49MZ7pk/view?usp=sharing)
- [Oxford105k](https://drive.google.com/file/d/1dbWtHkPSxlzPJtSYPnORaIBPKXgoaxFs/view?usp=sharing) 

After downloading the dat files you need to create a folder called `dataset ` and then put in the uncompressed version.
Remember to modify the path in the C++ files.

## Installation
* Requirements:
  * G++ 5.4 or greater
  * openmp
  * cblas  
* Build:
` g++ -o LSH_sparse LSH_sparse.cpp -lstdc++fs -std=c++14 -fopenmp -O2 -lcblas`

## Test

LSH kNN (δ = 6, L = 20, th = 5000, using global descriptors):
`./LSH_sparse 6 20 oxford5k false 5000 0 ResNet50`

multi LSH kNN graph (δ = 6, L = 20, th = 5000, 80% of multi-probe LSH, using global descriptors):
`./LSH_sparse 6 20 oxford5k false 5000 80 ResNet50`

For the diffusion application the python script implemented in the [alzaman/paiss](https://github.com/almazan/paiss) github is used.

## Results

### Oxford5k
 
| Configuration        | LSH projection           | kNN graph creation | mAP |
| :------------- |:-------------:| :-----:| :---:|
| LSH kNN graph (δ = 6, L = 20) | 0.45 s   | 0.52 s | 90.97% |
| LSH kNN graph (δ = 8, L = 10) | 0.4 s   | 0.94 s | 88.98% | 
| multi LSH kNN graph (δ = 6, L = 2) | 0.29 s   | 1.54 s   | 91.13%   | 
| NN-descent (1) | -   | 55 s   | 83.81%   | 
| RP-div (2) | -   | 1.16 s   | 82.68%   | 
| brute-force | -   | 1.33 s   | 90.79%   | 


### Oxford105k

| Configuration        | LSH projection           | kNN graph creation | mAP |
| :------------- |:-------------:| :-----:| :---:|
| LSH kNN graph (δ = 6, L = 20) | 23 s   | 77 s | 92.50% |
| LSH kNN graph (δ = 8, L = 10) | 15 s   | 145 s | 90.79% | 
| multi LSH kNN graph (δ = 6, L = 4) | 5s   | 420 s   | 92.85%   | 
| brute-force | -   | 4733 s   | 91.45%   | 



## Reference

<pre>@article{magliani2019efficient,
  title={An Efficient Approximate kNN Graph Method for Diffusion on Image Retrieval},
  author={Magliani, Federico and McGuiness, Kevin and Mohedano, Eva and Prati, Andrea},
  journal={arXiv preprint arXiv:1904.08668},
  year={2019}
}

@inproceedings{dong2011efficient,
  title={Efficient k-nearest neighbor graph construction for generic similarity measures},
  author={Dong, Wei and Moses, Charikar and Li, Kai},
  booktitle={Proceedings of the 20th International Conference on World Wide Web},
  pages={577--586},
  year={2011},
  organization={ACM}
}

@inproceedings{sieranoja2018fast,
  title={Fast random pair divisive construction of kNN graph using generic distance measures},
  author={Sieranoja, Sami and Fr{\"a}nti, Pasi},
  booktitle={Proceedings of the 2018 International Conference on Big Data and Computing},
  pages={95--98},
  year={2018},
  organization={ACM}
}

</pre>
