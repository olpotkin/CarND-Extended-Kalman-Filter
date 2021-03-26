#include "FusionEKF.h"
#include "tools.h"
#include "Eigen/Dense"
#include <iostream>


using Eigen::MatrixXd;
using Eigen::VectorXd;


// Constructor
FusionEKF::FusionEKF() {
  is_initialized_     = false;
  previous_timestamp_ = 0;

  // Initializing matrices
  // Measurement covariance matrix - laser
  R_laser_ = MatrixXd(2, 2);
  R_laser_ << 0.0225, 0.0,
              0.0,    0.0225;

  // Measurement covariance matrix - radar
  R_radar_ = MatrixXd(3, 3);
  R_radar_ << 0.09, 0.0,    0.0,
              0.0,  0.0009, 0.0,
              0.0,  0.0,    0.09;
  Hj_ = MatrixXd(3, 4);

  /// Finish initializing the FusionEKF.
  /// Set the process and measurement noises

  // Measurement matrix - laser
  H_laser_ = MatrixXd(2, 4);
  H_laser_ << 1, 0, 0, 0,
              0, 1, 0, 0;

  // State covariance matrix P_
  ekf_.P_ = MatrixXd(4, 4);
  ekf_.P_ <<  1, 0, 0,    0,
              0, 1, 0,    0,
              0, 0, 1000, 0,
              0, 0, 0,    1000;

  // The initial transition matrix F_
  ekf_.F_ = MatrixXd(4, 4);
  ekf_.F_ <<  1, 0, 1, 0,
              0, 1, 0, 1,
              0, 0, 1, 0,
              0, 0, 0, 1;

  noise_ax = 9.;
  noise_ay = 9.;
}


// Destructor
FusionEKF::~FusionEKF() {}


void FusionEKF::ProcessMeasurement(const MeasurementPackage& measurement_pack) {
  std::cout << "Initialization..." << std::endl;

  /// Initialization:
  /// Initialize the state ekf_.x_ with the first measurement.
  /// Create the covariance matrix.
  /// Remember: you'll need to convert radar from polar to cartesian coordinates.

  if (!is_initialized_) {
    // First measurement
    std::cout << "EKF: " << std::endl;
    ekf_.x_ = VectorXd(4);
    ekf_.x_ << 1, 1, 1, 1;

    if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
      std::cout << "Radar - step 1" << std::endl;
      // Convert radar from polar to cartesian coordinates and initialize state.
      double rho     = measurement_pack.raw_measurements_[0];
      double phi     = measurement_pack.raw_measurements_[1];
      double rho_dot = measurement_pack.raw_measurements_[2];

      double px = rho * cos(phi);
      double py = rho * sin(phi);

      // So while we can perfectly calculate px and py from phi, we cannot compute vx and vy from phi.
      // We will need yaw (which is introduced in UKF) to compute vx and vy.
      // So even from radar measurement, we can only compute px and py.
      // double vx = rho_dot * sin(phi);
      // double vy = rho_dot * cos(phi);
      double vx = 0;
      double vy = 0;

      ekf_.x_ << px, py, vx, vy;
    }
    else if (measurement_pack.sensor_type_ == MeasurementPackage::LASER) {
      std::cout << "Lidar - step 1" << std::endl;

      // Initialize state.
      ekf_.x_ << measurement_pack.raw_measurements_[0], measurement_pack.raw_measurements_[1], 0, 0;
    }

    // Special case initialisation problem (track 2)
    double eps = 0.000001F;
    if (std::fabs(ekf_.x_(0)) < eps && std::fabs(ekf_.x_(1)) < eps) {
      ekf_.x_(0) = eps;
      ekf_.x_(1) = eps;
    }

    // Print the initialization results
    std::cout << "EKF init: " << ekf_.x_ << std::endl;

    // Done initializing, no need to predict or update
    previous_timestamp_ = measurement_pack.timestamp_;
    is_initialized_ = true;

    return;
  }

  std::cout << "Prediction..." << std::endl;

  /// Prediction
  /// TODO:
  /// Update the state transition matrix F according to the new elapsed time.
  /// - Time is measured in seconds.
  /// Update the process noise covariance matrix.
  /// Use noise_ax = 9 and noise_ay = 9 for your Q matrix.

  // Compute the time elapsed between the current and previous measurements
  // dt - expressed in seconds
  double dt = (measurement_pack.timestamp_ - previous_timestamp_) / 1000000.0F;
  previous_timestamp_ = measurement_pack.timestamp_;

  ekf_.F_ = MatrixXd(4, 4);
  ekf_.F_ <<  1, 0, dt, 0,
              0, 1, 0,  dt,
              0, 0, 1,  0,
              0, 0, 0,  1;

  double dt2 = pow(dt, 2);
  double dt3 = pow(dt, 3);
  double dt4 = pow(dt, 4);

  ekf_.Q_ = MatrixXd(4, 4);
  ekf_.Q_ <<  dt4/4*noise_ax, 0,              dt3/2*noise_ax, 0,
              0,              dt4/4*noise_ay, 0,              dt3/2*noise_ay,
              dt3/2*noise_ax, 0,              dt2*noise_ax,   0,
              0,              dt3/2*noise_ay, 0,              dt2*noise_ay;

  std::cout << "ekf_.Predict()..." << std::endl;

  // Check if dt is above a certain threshold before predicting.
  // EKF prediction step does not handle dt=0 case gracefully!
  if (dt >= 0.000001) {
    ekf_.Predict();
  }

  std::cout << "ekf_.Predict() done!" << std::endl;

  /// Update:
  /// Use the sensor type to perform the update step.
  /// Update the state and covariance matrices.

  VectorXd x_new = measurement_pack.raw_measurements_;

  if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
    // Radar updates
    if ((x_new(0) != 0) || (x_new(1) != 0)) {
      Hj_     = tools.CalculateJacobian(ekf_.x_);
      ekf_.H_ = Hj_;
      ekf_.R_ = R_radar_;
      ekf_.UpdateEKF(x_new);

      std::cout << "Update RADAR done!" << std::endl;
    }
  }
  else {
    // Laser updates
    ekf_.R_ = R_laser_;
    ekf_.H_ = H_laser_;
    ekf_.Update(x_new);
    
    std::cout << "Update LIDAR done!" << std::endl;
  }

  // Print the output
  std::cout << "x_ = " << ekf_.x_ << std::endl;
  std::cout << "P_ = " << ekf_.P_ << std::endl;
}
