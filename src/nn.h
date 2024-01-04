#ifndef CNN_H
#define CNN_H
#include <vector>
#include "matrix.h"
class NN{
public:
	NN(){
		c_kernels.clear();
		f_w.clear();
		f_b.clear();
		total_cnn_layers=0;
		total_ff_layers=0;
	}
  void CNN(const int& input_size,const int& kernel_size);
  void FFNN(const int& input_size,const int& output_size);
  Vector feed_fc(const Vector& x,const int& cnn_layer_number)const ;
  Vector feed_ff(const Vector& x,const int& ff_layer_number)const;
  double output(Vector x)const ;
  void get_parameters(const RealVector& pvec);

private:
	int num_params_;
	int total_cnn_layers;
	int total_ff_layers;
	std::vector<Vector> c_kernels;
	std::vector<Matrix> f_w;
	std::vector<Vector> f_b;

};

#endif 
