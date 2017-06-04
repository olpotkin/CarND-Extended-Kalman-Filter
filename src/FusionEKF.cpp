#include "FusionEKF.h"
#include "tools.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;


// Constructor.
FusionEKF::FusionEKF() {
    is_initialized_ = false;
    previous_timestamp_ = 0;

    // Initializing matrices
    
    // Measurement covariance matrix - laser
    R_laser_ = MatrixXd(2, 2);
    R_laser_ << 0.0225, 0,
                0, 0.0225;
    
    // Measurement covariance matrix - radar
    R_radar_ = MatrixXd(3, 3);
    R_radar_ << 0.09, 0, 0,
                0, 0.0009, 0,
                0, 0, 0.09;
    
    Hj_ = MatrixXd(3, 4);

    /**
     * Finish initializing the FusionEKF.
     * Set the process and measurement noises
     */
    
    // Measurement matrix - laser
    H_laser_ = MatrixXd(2, 4);
    H_laser_ << 1, 0, 0, 0,
                0, 1, 0, 0;

    // State covariance matrix P_
    ekf_.P_ = MatrixXd(4, 4);
    ekf_.P_ <<  1, 0, 0, 0,
                0, 1, 0, 0,
                0, 0, 1000, 0,
                0, 0, 0, 1000;

    // The initial transition matrix F_
    ekf_.F_ = MatrixXd(4, 4);
    ekf_.F_ <<  1, 0, 1, 0,
                0, 1, 0, 1,
                0, 0, 1, 0,
                0, 0, 0, 1;
  
    noise_ax = 9.;
    noise_ay = 9.;
}


// Destructor.
FusionEKF::~FusionEKF() {}


void FusionEKF::ProcessMeasurement(const MeasurementPackage &measurement_pack) {
    cout << "Initialization..." << endl;
    /*****************************************************************************
    *  Initialization
    ****************************************************************************/
    if (!is_initialized_) {
        /**
         * Initialize the state ekf_.x_ with the first measurement.
         * Create the covariance matrix.
         * Remember: you'll need to convert radar from polar to cartesian coordinates.
        */
        
        // First measurement
        cout << "EKF: " << endl;
        ekf_.x_ = VectorXd(4);
        ekf_.x_ << 1, 1, 1, 1;

        if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
            cout << "Radar - step 1" << endl;
            
            // Convert radar from polar to cartesian coordinates and initialize state.
            double rho = measurement_pack.raw_measurements_[0];
            double phi = measurement_pack.raw_measurements_[1];
            double rho_dot = measurement_pack.raw_measurements_[2];
            
            double px = rho * cos(phi);
            double py = rho * sin(phi);
            double vx = rho_dot * sin(phi);
            double vy = rho_dot * cos(phi);

            ekf_.x_ << px, py, vx, vy;
        }
        else if (measurement_pack.sensor_type_ == MeasurementPackage::LASER) {
            cout << "Lidar - step 1" << endl;
            
            // Initialize state.
            ekf_.x_ << measurement_pack.raw_measurements_[0], measurement_pack.raw_measurements_[1], 0, 0;
        }
        
        // Special case initialisation problem (track 2)
        float eps = 0.000001;
        if (fabs(ekf_.x_(0)) < eps and fabs(ekf_.x_(1)) < eps){
            ekf_.x_(0) = eps;
            ekf_.x_(1) = eps;
        }
        
        // Print the initialization results
        cout << "EKF init: " << ekf_.x_ << endl;
        
        // Done initializing, no need to predict or update
        previous_timestamp_ = measurement_pack.timestamp_;
        is_initialized_ = true;
        return;
    }
    
    cout << "Prediction..." << endl;
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
    // dt - expressed in seconds
    float dt = (measurement_pack.timestamp_ - previous_timestamp_) / 1000000.0;
    previous_timestamp_ = measurement_pack.timestamp_;

    ekf_.F_ = MatrixXd(4, 4);
    ekf_.F_ <<  1, 0, dt, 0,
                0, 1, 0, dt,
                0, 0, 1, 0,
                0, 0, 0, 1;
    
    float dt2 = pow(dt, 2);
    float dt3 = pow(dt, 3);
    float dt4 = pow(dt, 4);
    
    ekf_.Q_ = MatrixXd(4, 4);
    ekf_.Q_ <<  dt4/4*noise_ax, 0, dt3/2*noise_ax, 0,
                0, dt4/4*noise_ay, 0, dt3/2*noise_ay,
                dt3/2*noise_ax, 0, dt2*noise_ax, 0,
                0, dt3/2*noise_ay, 0, dt2*noise_ay;
    
    cout << "ekf_.Predict()..." << endl;
    
    ekf_.Predict();

    cout << "ekf_.Predict() done!" << endl;

    /*****************************************************************************
    *  Update
    ****************************************************************************/

    /**
    TODO:
     * Use the sensor type to perform the update step.
     * Update the state and covariance matrices.
    */
    
    VectorXd x_new = measurement_pack.raw_measurements_;
    if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
        // Radar updates
        if ((x_new(0) != 0) || (x_new(1) != 0)) {
            Hj_ = tools.CalculateJacobian(ekf_.x_);
            ekf_.H_ = Hj_;
            ekf_.R_ = R_radar_;
            ekf_.UpdateEKF(x_new);
            cout << "Update RADAR done!" << endl;
        }
    }
    else {
        // Laser updates
        ekf_.R_ = R_laser_;
        ekf_.H_ = H_laser_;
        ekf_.Update(x_new);
        cout << "Update LIDAR done!" << endl;
    }

    // print the output
    cout << "x_ = " << ekf_.x_ << endl;
    cout << "P_ = " << ekf_.P_ << endl;
}
