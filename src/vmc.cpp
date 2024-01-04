/*---------------------------------------------------------------------------
* @Author: Amal Medhi, amedhi@mbpro
* @Date:   2019-03-20 13:07:50
* @Last Modified by:   Amal Medhi, amedhi@mbpro
* @Last Modified time: 2019-03-28 11:36:59
*----------------------------------------------------------------------------*/
// File: vmc.cpp

#include "vmc.h"
#include "matrix.h"
#include <fstream>

int VMC::init(void) 
{
  config.params();
  config.init(lattice_id::CHAIN,lattice_size(config.param[1]),wf_id::BCS,neuralnet_id::RBM);
  num_vparams = config.num_vparams();
  vparams.resize(num_vparams);
  num=2*num_vparams;
  n=num_vparams+num_vparams*(num_vparams+1)/2;
  HI =  0.0416; // set HI and LO according to your problem.
  LO = -0.0416;
  range = HI-LO;
  epsilon = 0.0001;
  step = 0.01;
  tol = config.param[5];
  //std::cout << tol << std::endl; getchar();
 
  sr_matrix.resize(num_vparams,num_vparams);
  U_.resize(num_vparams,num_vparams);
  I.resize(num_vparams,num_vparams);
  grads.resize(num_vparams);
  grad_log_psi.resize(num_vparams);
  lambda.resize(num_vparams);

  // run parameters
  num_samples = 2000;
  warmup_steps = 1000;
  iterations  = 2000;
  interval = 3;

  // observables
  energy.init("Energy");
  docc.init("double_occupancy");
  sign_value_.init("sign");
  //energy_grad.init("Energy_grad",2);
  u_triangular.init("u_triangular",n);
  f_values.init("f_values",num);

  return 0;
}

int VMC::run_simulation(void) 
{
  //set variational parameters
  //vparams.setOnes();
  /*double x;
  std::ifstream parm_file;  
  std::string filename;
  filename = "10.00.txt";
  parm_file.open(filename.c_str());
  int count = 0;
  if (!parm_file) {
    std::cout << "Unable to open file";
    exit(1); // terminate with error
  }    
  while (parm_file >> x) { 
    //std::cout << x << std::endl;
    vparams(count) = x;
    count ++;
  }*/ 
  vparams = (vparams + RealVector::Random(num_vparams,1.0))*range/2.;
  std::ofstream file("energy vs iterations.txt");
  std::ofstream vfile("vparams.txt");
  for (int nu=0; nu<iterations; ++nu){
    config.build(vparams);
    // warmup run
    config.init_state();
    for (int n=0; n<warmup_steps; ++n) {
      config.update_state();
    } 
    //std::cout << " warmup done\n";
    // measuring run
    int sample = 0;
    int skip_count = interval;
    int iwork_done = 0;
    // Initialize observables
    energy.reset();
    u_triangular.reset();
    f_values.reset();
    while (sample < num_samples) {
      if (skip_count == interval) {
        skip_count = 0;
        ++sample;
        int iwork = int((100.0*sample)/num_samples);
        if (iwork%10==0 && iwork>iwork_done) {
          iwork_done = iwork;
          //std::cout<<" done = "<<iwork<<"%\n";
        }
        // Make measurements
        //docc << config.Double_occupancy();
        config.get_gradlog_Psi(grad_log_psi);
        double E = config.get_energy();
        energy << E;
        u_triangular << config.product_grad_log_psi(grad_log_psi);
        f_values << config.measure_gradient(E,grad_log_psi);
      }
      config.update_state();
      skip_count++;
    }
    config.print_stats();
    //std::cout << docc.mean() << std::endl;
    double E_mean=energy.mean();
    //double sgn = sign_value_.mean();
    //std::cout<<E_mean<<std::endl;
    file << nu << "   " << E_mean << std::endl;
    std::cout << E_mean << std::endl;
    RealVector u_triangular_mean(n);
    RealVector f_values_mean(num);

    u_triangular_mean=u_triangular.mean_data();
    f_values_mean=f_values.mean_data();
    config.SR_matrix(sr_matrix,u_triangular_mean);
    for (unsigned i=0; i<num_vparams; ++i) sr_matrix(i,i) += 1.0E-4;
    config.finalize(E_mean,f_values_mean,grads);
    /*Eigen::VectorXd search_dir = sr_matrix.fullPivLu().solve(grads);
    vparams=vparams+step*search_dir;*/
    es_.compute(sr_matrix);
    U_ = es_.eigenvectors();
    lambda = es_.eigenvalues();
    //std::cout << U_ << std::endl; getchar();
    lambda_max = lambda(num_vparams-1);
    int count = num_vparams-2;
    for (int i = 0; i <num_vparams-1; ++i){
      double ratio = std::abs(lambda(count)/lambda_max);
      if (ratio < tol) break;
      count = count - 1;
    }

    for (int i = 0; i < num_vparams; ++i){
      double S_sum = 0.0;
      for (int j = 0; j < num_vparams; ++j){
        double U_sum = 0.0;
        for (int k = count+1; k < num_vparams; ++k){
          U_sum += U_(i,k)*U_(j,k)/(lambda(k));
        }
        S_sum += U_sum*grads(j);
      }
      //search_dir(i) = S_sum;
      //std::cout << S_sum << std::endl; getchar();
      vparams(i) = vparams(i) + step*S_sum;
    }
    //init();
    if (nu==iterations-1){
      vfile << vparams << std::endl;
    }
  }
  file.close();
  vfile.close();
  return 0;
}
