/*---------------------------------------------------------------------------
* @Author: Amal Medhi
* @Date:   2018-12-29 11:33:25
* @Last Modified by:   amedhi
* @Last Modified time: 2018-12-29 15:37:31
*----------------------------------------------------------------------------*/
#ifndef ACTIVATION_FUNC_H
#define ACTIVATION_FUNC_H

#include <iostream>
#include <stdexcept>
#include "constants.h"
#include "matrix.h"

namespace ann {

constexpr double pi(void) { return 3.1415926535897932384626433832795028841971693993751058209; }

class Activation
{
public:
  virtual ~Activation() {}
  virtual double function(const double& x) const = 0;
  virtual Vector function(const Vector& input) const = 0;
  virtual double derivative(const double& input) const = 0;
  virtual Vector derivative(const Vector& input) const = 0;
private:
};

class None : public Activation
{
public:
  None() {}
  ~None() {}
  double function(const double& x) const override { return x; }
  Vector function(const Vector& input) const override { return input; }
  double derivative(const double& x) const override { return 1.0; }
  Vector derivative(const Vector& input) const override { return Vector::Ones(input.size()); }
private:
};

class RELU : public Activation
{
public:
  RELU(const double& alpha=-1.0, const double& threshold=-1.0, 
  	const double&maxval=-1.0); 
  ~RELU() {}
  double function(const double& x) const override;
  Vector function(const Vector& input) const override;
  double derivative(const double& x) const override;
  Vector derivative(const Vector& input) const override;
private:
  bool default_{true};
  double alpha_{0.0};
  double threshold_{1.0};
  double maxval_{0.0};
};

class TANH : public Activation
{
public:
  TANH() {}
  ~TANH() {}
  double function(const double& x) const override;
  Vector function(const Vector& input) const override;
  double derivative(const double& x) const override;
  Vector derivative(const Vector& input) const override;
};

class Sigmoid : public Activation
{
public:
  Sigmoid() {}
  ~Sigmoid() {}
  double function(const double& x) const override;
  Vector function(const Vector& input) const override;
  double derivative(const double& x) const override;
  Vector derivative(const Vector& input) const override;
};

class ShiftedSigmoid : public Activation
{
public:
  ShiftedSigmoid() {}
  ~ShiftedSigmoid() {}
  double function(const double& x) const override;
  Vector function(const Vector& input) const override;
  double derivative(const double& x) const override;
  Vector derivative(const Vector& input) const override;
};

class LCOSH : public Activation
{
public:
  LCOSH() {}
  ~LCOSH() {}
  double function(const double& x) const override;
  Vector function(const Vector& input) const override;
  double derivative(const double& x) const override;
  Vector derivative(const Vector& input) const override;
};

class COSPI : public Activation
{
public:
  COSPI() {}
  ~COSPI() {}
  double function(const double& x) const override;
  Vector function(const Vector& input) const override;
  double derivative(const double& x) const override;
  Vector derivative(const Vector& input) const override;
};


} // end namespace ann

#endif