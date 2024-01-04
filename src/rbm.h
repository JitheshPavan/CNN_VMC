// File: rbm.h
#ifndef RBM_H
#define RBM_H

#include <complex>
#include <Eigen/Eigenvalues>
#include "wavefunction.h"
#include "./constants.h"
#include "./matrix.h"
#include "./lattice.h"


enum class neuralnet_id {RBM, FFN};

class Rbm
{
public:
	Rbm() {}
	Rbm(const neuralnet_id& nn_id, const int& nsites, const int& start_pos_, const int& hidden_density)
  		{ init_nn( nn_id, nsites, start_pos_, hidden_density); }
	~Rbm() {}
	void init_nn(const neuralnet_id& nn_id, const int& nsites, const int& start_pos_, const int& hidden_density);
	//void init_params(void);
	void get_rbm_parameters(const RealVector& pvec);
	void get_visible_layer(RealVector& sigma, const ivector& row) const;
	void get_vlayer(RealVector& sigma, const ivector& row) const;
	std::complex<double> get_rbm_amplitudes(const ivector& row) const;
	void get_inlayer(RealVector& sigma, std::vector<int> up,std::vector<int> dn) const;
	const int& num_vparams(void) const { return num_vparams_; }
	void get_derivatives(const RealVector& pvec, ComplexVector& derivatives, const ivector& row, const int& start_pos) const;
	void compute_theta_table(const ivector& row);
    void update_theta_table(const int& spin, const int& tsite, const int& fsite) const;
    double sign_value(void) const;    
	//void get_derivatives(const RealVector& pvec,RealVector& derivatives,
	     //std::vector<int> up,std::vector<int> dn,const int& start_pos) const;
	//double get_rbm_amplitudes(const std::vector<int>& up,const std::vector<int>& dn) const     
private:
	neuralnet_id id_;
	Wavefunction wf_;
    int num_vunits_;
  	int num_hunits_;
  	int num_sign_params_;
  	int num_vparams_;
  	int tot_sites_;
  	int num_upspins;
  	int num_dnspins;
  	int start_pos;
  	int alpha;
  	double sign_bias_;
	RealMatrix kernel_;
	RealVector in_bias_;
	RealVector hl_bias_;
	RealVector vparams_;
	RealVector v_layer_;
	RealVector sign_weights_;
	mutable RealMatrix theta_;
	mutable double theta_sign_;
};
#endif
