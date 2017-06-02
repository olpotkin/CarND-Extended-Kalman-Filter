#include "FusionEKF.h"
#include "tools.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/*
 * Constructor.
 */
FusionEKF::FusionEKF() {
  is_initialized_ = false;

  previous_timestamp_ = 0;

  // Initializing matrices
  R_laser_ = MatrixXd(2, 2);
  R_radar_ = MatrixXd(3, 3);
  H_laser_ = MatrixXd(2, 4);
  Hj_ = MatrixXd(3, 4);

  // Measurement covariance matrix - laser
  R_laser_ << 0.0225, 0,
              0     , 0.0225;

  // Measurement covariance matrix - radar
  R_radar_ <<   0.09, 0     , 0,
                0   , 0.0009, 0,
                0   , 0     , 0.09;

  /**
    * Finish initializing the FusionEKF.
    * Set the process and measurement noises
  */
  // Measurement matrix - laser
  H_laser_ <<   1, 0, 0, 0,
                0, 1, 0, 0;

  ekf_.Q_ = MatrixXd(4, 4);

  // State covariance matrix P
  ekf_.P_ = MatrixXd(4, 4);
  ekf_.P_ << 1, 0, 0,    0,
             0, 1, 0, 	 0,
             0, 0, 1000, 0,
		     0, 0, 0,    1000;

  // The initial transition matrix F_
	ekf_.F_ = MatrixXd(4, 4);
	ekf_.F_ << 1, 0, 1, 0,
		       0, 1, 0, 1,
		       0, 0, 1, 0,
		       0, 0, 0, 1;
	
	noise_ax = 9;
	noise_ay = 9;
}


/**
* Destructor.
*/
FusionEKF::~FusionEKF() {}

void FusionEKF::ProcessMeasurement(const MeasurementPackage &measurement_pack) {


  /*****************************************************************************
   *  Initialization
   ****************************************************************************/
  if (!is_initialized_) {
    /**
    TODO:
      * Initialize the state ekf_.x_ with the first measurement.
      * Create the covariance matrix.
      * Remember: you'll need to convert radar from polar to cartesian coordinates.
    */
    // first measurement
    cout << "EKF: " << endl;
    ekf_.x_ = VectorXd(4);
    ekf_.x_ << 1, 1, 1, 1;

    if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
      /**
      Convert radar from polar to cartesian coordinates and initialize state.
      */
    	double rho = measurement_pack.raw_measurements_[0];
		double phi = measurement_pack.raw_measurements_[1];
		double rho_dot = measurement_pack.raw_measurements_[2];
		ekf_.x_ << rho * cos(phi), rho * sin(phi), rho_dot * sin(phi), rho_dot * cos(phi);
    }
    else if (measurement_pack.sensor_type_ == MeasurementPackage::LASER) {
      /**
      Initialize state.
      */
    	ekf_.x_ << measurement_pack.raw_measurements_[0], measurement_pack.raw_measurements_[1], 0, 0;
    }

    // done initializing, no need to predict or update
    is_initialized_ = true;
    return;
  }

  /*****************************************************************************
   *  Prediction
   ****************************************************************************/

  /**
   TODO:
     * Update the state transition matrix F according to the new elapsed time.
      - Time is measured in seconds.
     * Update the process noise covariance matrix.
     * Use noise_ax = 9 and noise_ay = 9 for your Q matrix.
   */

  // Compute the time elapsed between the current and previous measurements
  float dt = (measurement_pack.timestamp_ - previous_timestamp_) / 1000000.0;	//dt - expressed in seconds
  previous_timestamp_ = measurement_pack.timestamp_;

  ekf_.F_ << 1, 0, dt, 0,
			 0, 1, 0,  dt,
			 0, 0, 1,  0,
			 0, 0, 0,  1;

  ekf_.Q_ << pow(dt, 4) / 4.0 * noise_ax, 0, pow(dt, 3) / 2.0 * noise_ax, 0,
		     0, pow(dt, 4) / 4.0 * noise_ay, 0, pow(dt, 3) / 2.0 * noise_ay,
		     pow(dt, 3) / 2.0 * noise_ax, 0, pow(dt, 2) * noise_ax, 0,
		     0, pow(dt, 3) / 2.0 * noise_ay, 0, pow(dt, 2) * noise_ay;

  if (dt > 0.001) {
		ekf_.Predict();
  }

  /*****************************************************************************
   *  Update
   ****************************************************************************/

  /**
   TODO:
     * Use the sensor type to perform the update step.
     * Update the state and covariance matrices.
   */

  if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
    // Radar updates
    Hj_ = tools.CalculateJacobian(ekf_.x_);
 	ekf_.H_ = Hj_;
	ekf_.R_ = R_radar_;
	ekf_.UpdateEKF(measurement_pack.raw_measurements_);
  } else {
    // Laser updates
	ekf_.H_ = H_laser_;
	ekf_.R_ = R_laser_;
	ekf_.Update(measurement_pack.raw_measurements_);
  }

  // print the output
  cout << "x_ = " << ekf_.x_ << endl;
  cout << "P_ = " << ekf_.P_ << endl;
}
