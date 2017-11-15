#include "ukf.h"
#include <iostream>
#include "Eigen/Dense"

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
    // if this is false, laser measurements will be ignored (except during init)
    use_laser_ = true;

    // if this is false, radar measurements will be ignored (except during init)
    use_radar_ = true;

    // initial state vector
    x_ = VectorXd(5);

    // initial covariance matrix
    P_ = MatrixXd(5, 5);

    // Todo : std_a_, std_yawdd_ parameter tuning
    // Process noise standard deviation longitudinal acceleration in m/s^2
    std_a_ = 30;

    // Process noise standard deviation yaw acceleration in rad/s^2
    std_yawdd_ = 30;

    // Laser measurement noise standard deviation position1 in m
    std_laspx_ = 0.15;

    // Laser measurement noise standard deviation position2 in m
    std_laspy_ = 0.15;

    // Radar measurement noise standard deviation radius in m
    std_radr_ = 0.3;

    // Radar measurement noise standard deviation angle in rad
    std_radphi_ = 0.03;

    // Radar measurement noise standard deviation radius change in m/s
    std_radrd_ = 0.3;

    /**
    TODO:

    Complete the initialization. See ukf.h for other member properties.

    Hint: one or more values initialized above might be wildly off...
    */

    // initially set to false, set to true in first call of ProcessMeasurement
    is_initialized_ = false;
    // State dimension
    n_x_ = 5;
    // Augmented state dimension
    n_aug_ = 7;
    // Sigma point spreading parameter
    lambda_ = 3 - n_aug_;
    time_us_ = 0;

    // initializing matrices
    x_ = VectorXd(n_x_);
    P_ = MatrixXd(n_x_, n_x_);
    Xsig_pred_ = MatrixXd(n_aug_, n_aug_);
}

UKF::~UKF() {}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
    /**
    TODO:

    Complete this function! Make sure you switch between lidar and radar
    measurements.
    */

    cout << "\n	sensor type : " << meas_package.sensor_type_ << "\n";
    cout << "\n	raw_measurements_ : " << meas_package.raw_measurements_ << "\n";
    cout << "\n	timestamp : " << meas_package.timestamp_ << "\n";

    /*****************************************************************************
     *  Initialization
     ****************************************************************************/
    if (!is_initialized_) {
        // state components
        float px;
        float py;
        float v = 0;
        float theta;
        float theta_d = 0;

        if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
            float rho = meas_package.raw_measurements_[0];  // range: radial
            float phi = meas_package.raw_measurements_[1];  // bearing:
            px = rho * cos(phi);
            py = rho * sin(phi);
            theta = atan2(py, px);

        } else if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
            // init state
            px = meas_package.raw_measurements_[0];
            py = meas_package.raw_measurements_[1];
            theta = atan2(py, px);
        }
        x_ << px, py, v, theta, theta_d;
        time_us_ = meas_package.timestamp_;

        // done initializing, no need to predict or update
        is_initialized_ = true;
        return;
    }

    /*****************************************************************************
     *  Prediction
     ****************************************************************************/

    /**
       * Update the state transition matrix F according to the new elapsed time.
        - Time is measured in seconds.
       * Update the process noise covariance matrix.
       * Use noise_ax = 9 and noise_ay = 9 for your Q matrix.
     */

    // compute the time elapsed between the current and previous measurements
    float dt = (meas_package.timestamp_ - time_us_) / 1000000.0;
    time_us_ = meas_package.timestamp_;

    Prediction(dt);

    /*****************************************************************************
     *  Update
     ****************************************************************************/

    /**
     * Use the sensor type to perform the update step.
     * Update the state and covariance matrices.
     */

    // Radar updates
    if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
        UpdateRadar(meas_package);
    }
    // Laser updates
    else {
        UpdateLidar(meas_package);
    }
}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) {
    /**
    TODO:

    Complete this function! Estimate the object's location. Modify the state
    vector, x_. Predict sigma points, the state, and the state covariance
    matrix.
    */

    cout << "\nPrediction() is called\n";
    cout << "	dt = " << delta_t << "\n";

    //    float dt_2 = dt * dt;
    //    float dt_3 = dt_2 * dt;
    //    float dt_4 = dt_3 * dt;
    //
    //    // Modify the F matrix so that the time is integrated
    //    ekf_.F_(0, 2) = dt;
    //    ekf_.F_(1, 3) = dt;
    //
    //    // set the process covariance matrix Q
    //    ekf_.Q_ = MatrixXd(4, 4);
    //    ekf_.Q_ << dt_4 / 4 * noise_ax, 0, dt_3 / 2 * noise_ax, 0, 0,
    //        dt_4 / 4 * noise_ay, 0, dt_3 / 2 * noise_ay, dt_3 / 2 * noise_ax,
    //        0, dt_2 * noise_ax, 0, 0, dt_3 / 2 * noise_ay, 0, dt_2 * noise_ay;
    //
    //    ekf_.Predict();
}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) {
    /**
    TODO:

    Complete this function! Use lidar data to update the belief about the
    object's position. Modify the state vector, x_, and covariance, P_.

    You'll also need to calculate the lidar NIS.
    */
}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package) {
    /**
    TODO:

    Complete this function! Use radar data to update the belief about the
    object's position. Modify the state vector, x_, and covariance, P_.

    You'll also need to calculate the radar NIS.
    */
}
