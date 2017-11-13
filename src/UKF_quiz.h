#ifndef UKF_QUIZ_H_
#define UKF_QUIZ_H_

#include <iostream>
#include "Eigen/Dense"

using Eigen::MatrixXd;
using Eigen::VectorXd;
using namespace std;

extern void test_sigma_point_generation(void);
extern void test_augmented_sigma_point_generation(void);
extern void test_predict_sigma_point(void);
extern void test_predict_mean_covariance(void);
extern void test_z_pred_s(void);
extern void test_kalman_gain(void);

#endif
