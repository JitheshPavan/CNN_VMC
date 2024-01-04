#include "nn.h"

void NN::CNN(const int& input_size,const int& kernel_size){
	c_kernels.push_back(Vector(kernel_size));
	total_cnn_layers++;
	num_params_ += kernel_size;
}
void NN::FFNN(const int& input_size,const int& output_size){
	f_w.push_back(Matrix(output_size,input_size));
	f_b.push_back(Vector(output_size));
	total_ff_layers++;
	num_params_ += output_size*(input_size +1);
}
void NN::get_parameters(const RealVector& pvec){
	int pos=0;
	if(total_cnn_layers>0){
		for(int i=0;i<total_cnn_layers;++i){
			int size=c_kernels[i].size();
			for(int j =0;j<size;++j){
				c_kernels[i][j]=pvec[pos++];
			}
		}
	}
	if(total_ff_layers>0){
		for(int i=0;i<total_ff_layers;++i){
			int size=f_w[i].cols()* f_w[i].rows();
			for(int j =0;j<size;++j){
				f_w(i)(j)=pvec[pos++];
			}
			size=f_b[i].size();
			for(int j=0;j<size;++j){
				f_b[i][j]=pvec[pos++];
			}

		}
	}
	if(!(pos==num_params_)){
		throw std::range_error("NN::get get_parameters range_error");
	}
}
Vector NN::feed_fc(const Vector& x,const int& cnn_layer_number)const {
	Vector kernel=c_kernels[cnn_layer_number];
	int kernel_size= kernel.size();
	int input_size= x.size();
	Vector output(input_size);
	for(int i=0;i<input_size;i++){
		double sum=0;
		for(int j=0;j<kernel_size;++j){
			int temp=i+j;
			while(temp>= input_size){
				temp=temp-input_size;
			}
			sum+= x(temp) * kernel(j);
		}
        output(i)= sum;
	}
	return output;
}
Vector NN::feed_ff(const Vector& x,const int& ff_layer_number) const{
	return x*f_w[ff_layer_number] + f_b[ff_layer_number];
}
double NN::output(Vector x)const {
	if(total_cnn_layers>0){
		for(int i=0;i<total_cnn_layers;++i){
			x= feed_fc(x,i);
		}
	}
	if(total_ff_layers>0){
		for(int i=0;i<total_ff_layers;++i){
			x=feed_ff(x,i);
		}
	}
	if(!(x.size()==1)){
		throw std::range_error("output is not a scalar");
	}
	return x(0);
}
