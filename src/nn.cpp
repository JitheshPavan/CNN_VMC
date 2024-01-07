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
	if(total_ff_layers>0){
		for(int i=total_ff_layers-1;i>=0;--i){
			int size=f_b[i].size();
			for(int j=0;j<size;++j){
				f_b[i][j]=pvec[pos++];
			}
			for(int j =0;j<f_w[i].cols();++j){
				for(int k=0;k<f_w[i].rows();++k){
					f_w[i](j,k)=pvec[pos++];
				}
			}
		}
	}
	if(total_cnn_layers>0){
		for(int i=total_cnn_layers;i>=0;--i){
			int size=c_kernels[i].size();
			for(int j =0;j<size;++j){
				c_kernels[i][j]=pvec[pos++];
			}
		}
	}

	if(!(pos==num_params_)){
		throw std::range_error("NN::get get_parameters range_error");
	}
}
void NN::get_vlayer(RealVector& sigma, const ivector& row) const
{
  for (int i=0; i<tot_sites_; i++){
    if(row(i) == 1) sigma(i) = 1;
    else sigma(i) = 0;
    if(row(i+tot_sites_) == 1) sigma(i+tot_sites_) = 1;
    else sigma(i+tot_sites_) = 0;
  }  
}
//-------------------------Feed-Forward-------------------------//
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
	return f_w[ff_layer_number]*x + f_b[ff_layer_number];
}
Vector NN::Sigmoid(const Vector& input) const
{
  return input.unaryExpr([](const double& x) {return 1.0/(1.0+std::exp(-x));});
}
double NN::output(Vector x)const {
	input_temp.clear();
	input_temp.push_back(x);
	if(total_cnn_layers>0){
		for(int i=0;i<total_cnn_layers;++i){
			x= feed_fc(x,i);
			input_temp.push_back(x);
			x=Sigmoid(x);
			input_temp.push_back(x);
		}
	}
	if(total_ff_layers>0){
		for(int i=0;i<total_ff_layers;++i){
			x=feed_ff(x,i);
			input_temp.push_back(x);
			x=Sigmoid(x);
			input_temp.push_back(x);
		}
	}
	if(!(x.size()==1)){
		throw std::range_error("output is not a scalar");
	}
	return x(0);
}

Vector NN::Sigmoidderivative(const Vector& input) const
{
  return input.unaryExpr([](const double& x) 
	{ double y = 1.0/(1.0+std::exp(-x)); return y*(1.0-y); });
}

void NN::get_derivatives(const RealVector& pvec, ComplexVector& derivatives, const ivector& row, const int& start_pos) const{
  	RealVector sigma;
  	sigma.resize(2*tot_sites_);
  	get_vlayer(sigma, row);

  	int total_layers= total_ff_layers+ total_cnn_layers;
  	output(sigma);
  	Vector error_signal= input_temp[total_layers-1];

  	int tracker=0;
  	if(total_ff_layers>0){	
  		for(int i=0;i<total_ff_layers;++i){
  			error_signal= Sigmoidderivative(input_temp[total_layers-(i+1)*2]).array() * error_signal.array();
  			for(int j=0;j<error_signal.size();j++){
  				derivatives(tracker++)= error_signal[j];
  			}
  			Matrix local_gradient=error_signal * input_temp[total_layers-(i+1)*2+1].transpose();
  			error_signal= f_w[total_ff_layers-i-1].transpose() * error_signal;
  			for (int i = 0; i < local_gradient.cols(); ++i){
    			for (int j = 0; j < local_gradient.rows(); ++j){
      				derivatives(tracker++) = local_gradient(i,j);
      			}
      		}
    	}
  	}
  	if(total_cnn_layers>0)
  	{
  		for(int i=0;i<total_cnn_layers;++i){
  			error_signal= Sigmoidderivative(input_temp[total_layers-(i+1)*2]).array() * error_signal.array();
  			for(int j=0;j<error_signal.size();j++){
  				derivatives(tracker++)= error_signal[j];
  			}

  		}	
  	}
}
 