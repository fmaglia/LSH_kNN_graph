#ifndef UTILS_H
#define UTILS_H

#include <iostream>
#include <vector>
#include <chrono>
#include <fstream>
#include <time.h>
#include <algorithm>
#include <math.h>
#include <experimental/filesystem>
namespace fs = std::experimental::filesystem;
using namespace std;

struct sparse_matrix {
    int row;
    int col;
    float value;
};

struct nn_descent{
  int col;
  float value;
};


float l2_norm_2vectors(std::vector<float> &v, std::vector<float> &u) {
    float accum = 0.;
    for (unsigned int i = 0; i < v.size(); ++i) {
        accum += (v[i] - u[i])*(v[i] - u[i]);
    }
    return sqrt(accum);
}

float dot_product(std::vector<float> &v, std::vector<float> &u){
    float result = 0.0;
    for (unsigned int i = 0; i < v.size(); ++i) {
        result += v[i] * u[i];
    }
    return result;
}

void readTraining(const string &home, const string &dataset, std::vector<string> &fileTrainingSet, const bool regions, const string network) {
    if (dataset == "oxford5k"){
      //fileTrainingSet.push_back(home+"/VGG16_ox.npy");
      //fileTrainingSet.push_back(home+"/VGG16.txt");
      if (regions)
        fileTrainingSet.push_back(home+"/"+network+"_ox_region");
      else
        fileTrainingSet.push_back(home+"/"+network+"_ox");
    }

    if (dataset == "oxford105k"){
      if (regions){
        //fileTrainingSet.push_back(home+"/"+network+"_ox_region");
        //fileTrainingSet.push_back(home+"/"+network+"_flickr100k_regions");
        fileTrainingSet.push_back(home+"/"+network+"_region_ox105k");
      }
      else{
        fileTrainingSet.push_back(home+"/"+network+"_ox105k");
        //fileTrainingSet.push_back(home+"/"+network+"_ox");
        //fileTrainingSet.push_back(home+"/"+network+"_flickr100k");
      }
    }

    if (dataset == "paris6k"){
      if (regions)
        fileTrainingSet.push_back(home+"/"+network+"_par_region");
      else
        fileTrainingSet.push_back(home+"/"+network+"_par");
    }

}

string calculateBinary(int number, int hash_dimension) {
    string result = "";
    int counter = 0;

    while (number>0) {
        result.insert(0,to_string(number%2));
        number /= 2;
        counter++;
    }

    while (counter<hash_dimension){
        result.insert(0,"0");
        counter++;
    }

    return result;


}

int calculateDecimal (string binary) {
    int result = 0;

    for (unsigned int i=0; i < binary.size(); ++i) {
        if ('1'==binary[i])
            result += pow(2,binary.size()-i-1);
    }

    return result;
}
string changeBit(string binary, int position) {

    if (binary[position]=='0')
        binary[position]='1';
    else
        binary[position]='0';

    return binary;
}

void calculateNeighbors (vector <int> & vicini, string binary, int position, int vicinato){

    if (vicinato==1) {
        vicini.push_back(calculateDecimal(changeBit(binary, position)));
    }
    /*else {
        for (unsigned int j=0; j<binary.size()-1; ++j){

            string newBinary = changeBit(binary, position);
            for (int v=1; v < vicinato; ++v) {
                if (j+v!=position)
                    newBinary = changeBit(newBinary, j+v);
                else if (j+v-1 >= 0)
                    newBinary = changeBit(newBinary, j+v-1);
                else if (j+v+1 < binary.size())
                    newBinary = changeBit(newBinary, j+v+1);
            }

            int value = calculateDecimal(newBinary);
            if (std::find(vicini.begin(), vicini.end(), value) == vicini.end())
                vicini.push_back(value);
        }

    }*/
    return;

}
/*
int lsh_indexing_new(const int hash_dimension, std::vector<float> &descriptor, std::vector<std::vector<float>> &projectionVector, const int iteration) {

    int result=0;

    float subresult = floor(abs(dot_product(descriptor,projectionVector[iteration]))/0.1);

    //cout << "iteration: "<<iteration<<" subresult: "<<subresult<<endl;


    //update hash for unique vector (using different hash tables)
    //result += iteration*pow(2,hash_dimension);
    switch (hash_dimension) {
        case 4: result += iteration*16;
        break;
        case 5: result += iteration*32;
        break;
        case 6: result += iteration*64;
        break;
        case 7: result += iteration*128;
        break;
        case 8: result += iteration*256;
        break;
        case 9: result += iteration*512;
        break;
    }

    return result;

}*/

int lsh_indexing(const int hash_dimension, std::vector<float> &descriptor, std::vector<std::vector<float>> &projectionVector, const int iteration) {

    int result=0;

    for (int j=0; j < hash_dimension; ++j) {
        float subresult = 0;

        for (unsigned int i=0; i < descriptor.size(); ++i) {
            //cout << "proj  "<<j+iteration*hash_dimension<<" on "<< hash_dimension*20<<" VLAD "<<i<<" on "<<descriptor.size()<< endl;
            subresult += descriptor[i]*projectionVector[j+iteration*hash_dimension][i];
        }
        if (subresult > 0) {
            switch (j)
            {
                case 0:  result += 1;
                break;
                case 1: result += 2;
                break;
                case 2: result += 4;
                break;
                case 3: result += 8;
                break;
                case 4: result += 16;
                break;
                case 5: result += 32;
                break;
                case 6: result += 64;
                break;
                case 7: result += 128;
                break;
                case 8: result += 256;
                break;
                case 9: result += 512;
                break;
		            case 10: result += 1024;
		            break;
                case 11: result += 2048;
                break;
                case 12: result += 4096;
                break;
                case 13: result += 8192;
                break;
                case 14: result += 16384;
                break;
                case 15: result += 32768;
                break;
            }
            //result += pow(2,j);
        }
        //cout << "iteration: "<<iteration<<" hash_dim "<<j<<" subresult: "<<subresult<<" result: "<<result<<endl;
    }

    //update hash for unique vector (using different hash tables)
    //result += iteration*pow(2,hash_dimension);
    /*switch (hash_dimension) {
        case 4: result += iteration*16;
        break;
        case 5: result += iteration*32;
        break;
        case 6: result += iteration*64;
        break;
        case 7: result += iteration*128;
        break;
        case 8: result += iteration*256;
        break;
        case 9: result += iteration*512;
        break;
    }*/

    return result;

}

void increase_next_row_pointers(const int &row, std::vector <int>& row_pointer){

  for (int unsigned i=row+1; i < row_pointer.size(); i++){
    if (row_pointer[i] >= 0)
      row_pointer[i]++;
  }
}

int find_prev_row(const int &row, std::vector <int>& row_pointer){
  for (int i=row-1; i >= 0; i--){
    if (row_pointer[i] >= 0)
      return i;
  }
  return row;
}

int find_next_row(const int &row, std::vector <int>& row_pointer){
  for (int unsigned i=row+1; i < row_pointer.size(); i++){
    if (row_pointer[i] >= 0)
      return i;
  }
  return row;
}

void insert_no_head(std::vector <float> &values, std::vector <int> &column_indices, std::vector<int>& row_pointer, const int &actual_row, const int &next_row, const int & col, const float & val){

  int s_index = row_pointer[actual_row];
  int f_index = row_pointer[next_row];
  bool inserted = false;

  if (s_index == f_index){
    //no next row
    f_index = values.size();
  }

  while (!inserted && s_index < f_index){
    if (col == column_indices[s_index]){
      inserted = true;
      break;
    }

    if (col < column_indices[s_index]){
      values.insert(values.begin()+s_index, val);
      column_indices.insert(column_indices.begin()+s_index, col);
      increase_next_row_pointers(actual_row, row_pointer);
      inserted = true;
      break;
    }

    s_index++;
  }

  if (!inserted){
    values.insert(values.begin() + s_index, val);
    column_indices.insert(column_indices.begin() + s_index, col);
    increase_next_row_pointers(actual_row, row_pointer);
    inserted = true;
  }


}

void update_CRS(std::vector <float>& values, std::vector<int> &column_indices, std::vector<int> &row_pointer, float& val, int &row, int& col){

  //new ROW
  if (row_pointer[row] == -1){

    int next_row = find_next_row(row, row_pointer);
    //int prev_row = find_prev_row(row, row_pointer);
    int index;
    if (next_row == row){
      //insert in tail
      values.push_back(val);
      column_indices.push_back(col);
      row_pointer[row] = values.size()-1;
    }
    else if (next_row != row) {
      //insert in the middle
      index = row_pointer[next_row];
      row_pointer[row] = index;
      values.insert(values.begin() + index, val);
      column_indices.insert(column_indices.begin() + index, col);
    }

    increase_next_row_pointers(row, row_pointer);

  }

  //add element to existing ROW
  else {

    insert_no_head(values, column_indices, row_pointer, row, find_next_row(row, row_pointer), col, val);

  }

  //cout << "Values size "<<values.size()<<endl;
}

bool find_same(const sparse_matrix &element, const std::vector<sparse_matrix> &m){
  bool result = false;
  for (auto & e : m){
      if (e.row == element.row && e.col == element.col){
        result = true;
        break;
      }

    }

  return result;
}

void update_COO(const float& threshold, std::vector <sparse_matrix> &m, sparse_matrix& element){
  if (element.value > threshold) {

    if (!find_same(element, m)){
      m.push_back(element);
    }

  }
  //cout << "Values size "<<values.size()<<endl;
}


/*
bool find_same_new(const sparse_matrix &elem, const std::vector<sparse_matrix> &m){
  if (std::find(m.begin(), m.end(), elem) == m.end())
  //if (std::find_if(m.begin(), m.end(), pred) == m.end())
    return false;
  else
    return true;
}

void update_COO_new(const float& threshold, std::vector <sparse_matrix>& m, sparse_matrix & element){
  if (element.value > threshold) {
    if (!find_same_new(element, m))
      m.push_back(element);
  }
}*/

void convert_COO_nn(const float &threshold, const std::vector <std::vector<nn_descent>>& nn, std::vector <float> &values, std::vector<int> &column_indices, std::vector<int> &row_pointer){

  cout << "Conversion to Cooordinate format with threshold = "<<threshold<<endl;
  for (unsigned int i=0; i < nn.size(); i++){
    for (unsigned int j=0; j < nn[i].size(); j++){
        if (nn[i][j].value > threshold){
          //cout << "Print -> row "<<i<<" col "<<nn[i][j].col<<" value "<<nn[i][j].value<<endl;
          values.push_back(nn[i][j].value);
          column_indices.push_back(nn[i][j].col);
          row_pointer.push_back(i);

        }
    }
  }

}

void convert_CRS(const float &threshold, const std::vector <std::vector<float>>& mat, std::vector <float> &values, std::vector<int> &column_indices, std::vector<int> &row_pointer){

  cout << "Conversion to Compressed Row Storage format with threshold = "<<threshold<<endl;
  bool new_row = false;

  int nnz = 0;
  for (unsigned int i=0; i < mat.size(); i++){
    new_row = false;
    for (unsigned int j=0; j < mat[i].size(); j++){
        if (mat[i][j] > threshold){
          values.push_back(mat[i][j]);
          column_indices.push_back(j);
          nnz++;

          if (!new_row){
            new_row = true;
            row_pointer.push_back(values.size()-1);
          }

        }
    }
  }
  row_pointer.push_back(nnz);

}

void convert_COO(const float &threshold, const std::vector <std::vector<float>>& mat, std::vector <float> &values, std::vector<int> &column_indices, std::vector<int> &row_pointer){

  cout << "Conversion to Cooordinate format with threshold = "<<threshold<<endl;
  for (unsigned int i=0; i < mat.size(); i++){
    for (unsigned int j=i; j < mat[i].size(); j++){
        if (mat[i][j] > threshold){
          values.push_back(mat[i][j]);
          column_indices.push_back(j);
          row_pointer.push_back(i);

        }
    }
  }

}

template<typename T>
void write_array(const string &path, const std::vector<T> &v){
  std::ofstream outFile(path, std::ofstream::binary);

  for (unsigned int i=0; i < v.size(); i++){
      outFile.write(reinterpret_cast<const char *>(&v[i]), sizeof(T));
  }
  outFile.close();
}

template<typename T>
void write_array_new(const string &path, const T &v){
  std::ofstream outFile(path, std::ofstream::binary);
  auto ptr = &(v[0]);
  auto bytes_per_element = sizeof(typename T::value_type);
  auto total_bytes = v.size() * bytes_per_element;
  outFile.write(reinterpret_cast<const char *>(ptr), total_bytes);
  outFile.close();
}

void write_matrix(const string &path, const int &trainingElements, const std::vector<std::vector<float>> &mat){
  std::ofstream outFile(path, std::ofstream::binary);
  //std::ofstream outFile("/media/eHD/federico/graph_"+dataset+"_hash_dim"+to_string(hash_dimension)+"_L"+to_string(L)+".txt");

  for (unsigned int i=0; i < mat.size(); i++){
    //outFile.write(reinterpret_cast<const char *>(&mat[i]), sizeof(float)*mat[i].size());

    for (unsigned int j=0; j < mat[i].size(); j++){
        //outFile << mat[i][j] << " ";
        outFile.write(reinterpret_cast<const char *>(&mat[i][j]), sizeof(float));
    }
    //outFile << "\n";
  }
  outFile.close();
}



#endif // UTILS_H
