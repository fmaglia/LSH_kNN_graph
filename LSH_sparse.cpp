#define NDEBUG

#include <iostream>
#include <vector>
#include <chrono>
#include <fstream>
#include <time.h>
#include <algorithm>
#include <random>
#include <mutex>
#include <omp.h>
#include <thread>
#include "utils.h"
#include <cstring>
#include <experimental/filesystem>
#include <boost/numeric/ublas/matrix_sparse.hpp>
#include <boost/numeric/ublas/io.hpp>
#include "/opt/intel/mkl/include/mkl.h"
//#include <cblas.h>
using namespace std;
namespace fs = std::experimental::filesystem;

int main(int argc, char* argv[])
{
    //std::cout<<"example: ./LSH_sparse 7 20 oxford5k true 4000 80 VGG16 "<<endl;

    /* Configuration variables*/
    int hash_dimension = stoi(argv[1]); //LSH -> bit (2^hash_dimension)
    float sigma = 1.0;
    int L = stoi(argv[2]);
    string dataset=argv[3];
    string home = "/media/federico/DCU";

    string reg = argv[4];
    bool regions = false;
    if (reg == "true" || reg == "True")
      regions = true;
    int max_number_elements = stoi(argv[5]);

    int multiLSH = stoi(argv[6]);
    int multiElements = 1;
    if (multiLSH == 0)
      std::cout << "LSH projections"<<endl;
    else{
      multiElements = round(multiLSH * hash_dimension/100);
      std::cout << "multi LSH projections ("<<multiLSH<<"% -> "<<multiElements<<")"<<endl;
    }

    /* End configuration variables */
    string network = argv[7];
    int globalVectorDimension = 512;
    if (network == "ResNet50" || network =="resnet50")
        globalVectorDimension = 2048;
    float threshold = 0.4;

    if (dataset =="oxford105k" && regions)
      home = "/media/federico/DCU";

    int hash_code = pow(2,hash_dimension);

   // if (dataset == "oxford105k" || dataset == "paris106k")
   //     threshold = 0.3;
    std::vector <string> fileTrainingSet;
    readTraining(home, dataset,fileTrainingSet, regions, network);
    std::vector <std::vector<int>> lsh_index (hash_code*L, std::vector <int>(0));

    auto tEncoding1 = std::chrono::high_resolution_clock::now();

    std::vector <std::vector<float>> descriptor_trainingSet;

    //gaussian distribution (mean = 0 and stddev = sigma) for the assignment of value of projection vector
    std::normal_distribution<double> distribution(0.0,sigma);
    std::random_device rd;
    std::mt19937 generator(rd()); //use 0 as a parameter for using VALGRIND (profiler), otherwise use rd()

    std::vector <std::vector<float>> projectionVector (hash_dimension*L, std::vector<float>(globalVectorDimension));

    for (unsigned int i=0; i<projectionVector.size(); ++i) {
        for (unsigned int j=0; j<projectionVector[i].size(); ++j) {
            projectionVector[i][j] = distribution(generator);
        }
    }

    std::vector <std::vector<int>> neighbor (0, std::vector <int>(0));
    int vicinato = 1;
    if (!multiLSH)
        vicinato = 0;
    for (int i=0; i < pow(2, hash_dimension); ++i) {
        string binary = calculateBinary(i, hash_dimension);
        std::vector <int> vicini;
        for (int v=1; v <= vicinato; ++v) {
            for (int j=0; j < hash_dimension; ++j) {
                calculateNeighbors(vicini, binary, j, v);
            }
        }
        vicini.insert(vicini.begin(),i);
        neighbor.push_back(vicini);
    }


    int trainingElements = 0;
    /* Reading database data */

    for (unsigned int i=0; i < fileTrainingSet.size(); i++) {
      std::cout << "Element "<<fileTrainingSet[i] << endl;
      std::ifstream fileStream(fileTrainingSet[i], std::ios::binary);

      string line;
      std::vector <float> descriptor_vector;
      float f;
      int counter = 0;
      while (fileStream.read(reinterpret_cast<char*>(&f), sizeof(float))){
          descriptor_vector.push_back(f);
          counter++;
          if (counter == globalVectorDimension) {
              descriptor_trainingSet.push_back(descriptor_vector);
	             //if (trainingElements == 0)
		             //std::cout<<descriptor_vector[0]<<" "<<descriptor_vector[1]<<endl;
                 omp_lock_t writelock;

                 omp_init_lock(&writelock);
           #pragma omp parallel num_threads(5)
          {
          #pragma omp for
          for (int hashTables = 0; hashTables < L; ++hashTables){
              int bucket_index = lsh_indexing(hash_dimension, descriptor_vector, projectionVector, hashTables);
              for (int j=0; j <= multiElements; j++){
                int result_index = neighbor[bucket_index][j];
                switch (hash_dimension) {
                    case 4: result_index += hashTables*16;
                    break;
                    case 5: result_index += hashTables*32;
                    break;
                    case 6: result_index += hashTables*64;
                    break;
                    case 7: result_index += hashTables*128;
                    break;
                    case 8: result_index += hashTables*256;
                    break;
                    case 9: result_index += hashTables*512;
                    break;
		                case 10: result_index += hashTables*1024;
		                break;
                    case 11: result_index += hashTables*2048;
                    break;
                    case 12: result_index += hashTables*4096;
                    break;
                    case 13: result_index += hashTables*8196;
                    break;
                    case 14: result_index += hashTables*16384;
                    break;
                    case 15: result_index += hashTables*32768;
                    break;
                }
                omp_set_lock(&writelock);

                lsh_index[result_index].push_back(trainingElements);
                omp_unset_lock(&writelock);

              }

            }
          }
          descriptor_vector.clear();
          counter = 0;
          trainingElements++;
          omp_destroy_lock(&writelock);
        }


      }

      fileStream.close();
  }


    auto tEncoding2 = std::chrono::high_resolution_clock::now();

    std::cout<<"Max number of elements per bucket "<<max_number_elements<<endl;


    float timeEncoding = (std::chrono::duration_cast<std::chrono::milliseconds>(tEncoding2 - tEncoding1).count());


    cout << "Encoding of "<<trainingElements<<" database images TERMINATED in "<<timeEncoding/1000<<" s"<< endl;

    cout << "*********Approximate kNN graphs creation through LSH sparse********"<<endl;
    cout << "Bits: "<<hash_dimension<<endl;
    cout << "Hash tables: "<<L<<endl;
    tEncoding1 = std::chrono::high_resolution_clock::now();

    std::vector <sparse_matrix> m;
    //std::vector<float> blacklist (trainingElements,0.0);

    //std::cout<<"Buckets "<<lsh_index.size()<<endl;

    omp_lock_t writelock;

    omp_init_lock(&writelock);

    #pragma omp parallel num_threads(5)
    {
    #pragma omp for
    for (unsigned int bucket=0; bucket < lsh_index.size(); bucket++){
        auto && b = lsh_index[bucket];
        //std::cout<<"Hash table "<<bucket/hash_code<<" bucket "<<bucket%hash_code<<" found "<<b.size()<<" elements"<<endl;

        // copy
         int N = b.size();
         if (N == 0)
            continue;
        /*if (N > max_number_elements){
          N = max_number_elements;
          //std::cout << "Bounded bucket elems "<<b.size()<<" to "<<max_number_elements<<endl;
        }*/

         auto A = new float[N * globalVectorDimension]; // A [N, 2048] row major
         //int elements = 0;
         for (unsigned int i = 0; i < N; i++) {
             auto index = b[i];
             auto dataptr = descriptor_trainingSet[index].data();
             std::memcpy(A + (globalVectorDimension * i), dataptr, globalVectorDimension * sizeof(float));
         }

         std::vector<float> C (N*N);
         cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
              N, N, globalVectorDimension, 1.0f, A, globalVectorDimension, A, globalVectorDimension, 0.0f, &C[0], N);

         for (unsigned int elem = 0; elem < N; elem++) {
            auto && b_elem = b[elem];
            for (unsigned int elem1=0; elem1 < N; elem1++) {
               auto && b_elem1 = b[elem1];

               float val = C[elem1 + elem*N];
               //std::cout <<"R "<<b_elem<<" C "<<b_elem1<<" val "<<val<<endl;

               if (val > threshold){

               //  ROW : b_elem   COL : b_elem1
               int row = b_elem;
               int col = b_elem1;

                float value;
                 if (row==col)
                    value = 0.0;
                 else
                    value = pow(val,10);

		             //std::cout <<"R "<<row<<" C "<<col<<" val "<<val<<endl;
                 //update_CRS(values, column_indices, row_pointer, value, row, col);

                 sparse_matrix element;
                 element.row = row;
                 element.col = col;
                 element.value = value;
                 omp_set_lock(&writelock);

                 m.push_back(element);

                 element.row = col;
                 element.col = row;
                 m.push_back(element);
                 omp_unset_lock(&writelock);

                 //graph.insert_element(row, col, val);
               }



               /*sparse_matrix element;
               element.row = row;
               element.col = col;
               element.value = val;
               update_COO(threshold, m, element);*/
               //update_COO_new(threshold, m, element);

            }

         }

         delete A;
       }


  }
  omp_destroy_lock(&writelock);


    tEncoding2 = std::chrono::high_resolution_clock::now();
    timeEncoding = (std::chrono::duration_cast<std::chrono::milliseconds>(tEncoding2 - tEncoding1).count());
    cout << "*********Final graph created in "<<timeEncoding/1000<<" s ***********"<< endl;



    tEncoding1 = std::chrono::high_resolution_clock::now();
    string path ="/media/federico/DCU/graph_"+dataset+"_hash_dim"+to_string(hash_dimension)+"_L"+to_string(L);
    if (regions)
      path += "_regions";

      //previous solution: FIND O(N) executed N*L times
      //actual solution:   SORT O(N logN)

      std::sort(m.begin(), m.end(),
                [](const sparse_matrix& a, const sparse_matrix& b) {
            if (a.row < b.row)
              return true;
            if (b.row < a.row)
              return false;
            return (a.col < b.col);
    });

    std::cout<<"Coordinate matrix shape "<<m.size()<<endl;

    // convert to CSR storage
    //compressed_matrix<float, row_major> g = graph;

    // Note: this must be called before using index1_data
    //g.complete_index1_data();

    //std::cout << "row pointer "<<g.index1_data().size()<<" values "<<g.value_data().size()<<endl;

    // Write CSR vectors
    //write_array_new(path+"_row_pointer", g.index1_data());
    //write_array_new(path+"_column_indices", g.index2_data());
    //write_array_new(path+"_values", g.value_data());

    //convert COO to CRS write_matrix
    std::vector <float> values;
    std::vector <int> column_indices;
    std::vector <int> row_pointer;

    int prev_row = -1;
    int prev_col = -1;
    bool new_row = false;
    for (unsigned int i=0; i < m.size(); i++){
      auto m_i = m[i];
      if (m_i.row != prev_row)
        new_row = false;
      if (m_i.row == prev_row && m_i.col == prev_col){
        continue;
      }
      else{
        values.push_back(m_i.value);
        column_indices.push_back(m_i.col);
        if (!new_row){
          row_pointer.push_back(values.size()-1);
          new_row = true;
          prev_row = m_i.row;
        }
        prev_col = m_i.col;
      }
    }

    row_pointer.push_back(values.size());

    write_array_new(path+"_values",values);
    write_array_new(path+"_column_indices",column_indices);
    write_array_new(path+"_row_pointer",row_pointer);

    std::cout<<"SIZES - values "<<values.size()<<" row_pointer "<<row_pointer.size()<<" column "<<column_indices.size()<<endl;
    //float graph_accuracy = values.size()*2*100/(trainingElements*trainingElements);
    float graph_accuracy = values.size()*2*100/(fileTrainingSet.size()*fileTrainingSet.size());
    //printf("Graph accuracy: %4f",graph_accuracy);
    std::cout<<"Graph accuracy on the overall graph: "<< graph_accuracy <<endl;

    tEncoding2 = std::chrono::high_resolution_clock::now();
    timeEncoding = (std::chrono::duration_cast<std::chrono::milliseconds>(tEncoding2 - tEncoding1).count());
    cout << "Writing terminated in "<<timeEncoding/1000<<" s using a threshold = "<<threshold<< endl;


    string command = "python3 /home/federico/paiss-master/script.py --hash approx_LSH_sparse --bit "+to_string(hash_dimension)+" --L "+to_string(L)+" --dataset "+dataset+" --network "+network;
    if (regions)
      command += " --regions True";
    system(command.c_str());

    return 0;

}
