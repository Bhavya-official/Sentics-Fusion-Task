#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <map>
#include <cmath>
#include <random>
#include <algorithm>
#include <iomanip>
#include <chrono>
#include "json.hpp"  // Include nlohmann::json

using json = nlohmann::json;  // Define shorthand for nlohmann::json
using namespace std;

// Kalman Filter class for estimating the state of an object (e.g., heading angle)
class KalmanFilter {
public:
    KalmanFilter(double processNoise, double measurementNoise, double initialEstimate, double initialCovariance)
        : Q(processNoise), R(measurementNoise), x_hat(initialEstimate), P(initialCovariance) {}

    KalmanFilter() = default;

    // Function to update the filter with a new measurement (returns the updated state)
    double update(double z) {
        // Prediction step: predict the next state and its uncertainty
        double A = 1;
        double B = 0;
        double u = 0;
        
        double x_hat_pred = A * x_hat + B * u; // predicted state estimate
        double P_pred = A * P * A + Q; // predicted covariance estimate

        // Measurement update step: incorporate the new measurement to correct the prediction
        double K = P_pred / (P_pred + R);  // Kalman gain
        x_hat = x_hat_pred + K * (z - x_hat_pred);  // updated state estimate
        P = (1 - K) * P_pred;  // updated covariance estimate

        return x_hat;  // return the smoothed angle
    }

    // Function to set the Kalman filter parameters
    void set_parameters(double processNoise, double measurementNoise, double initialEstimate, double initialCovariance)
    {
        Q = processNoise; 
        R = measurementNoise; 
        x_hat = initialEstimate; 
        P = initialCovariance;
    }    

private:
    double Q;  // Process noise covariance
    double R;  // Measurement noise covariance
    double x_hat;  // Estimated state (heading angle)
    double P;  // Estimated covariance (uncertainty)
};

// Structure to represent a cluster of sensor data (e.g., multiple forklift detections)
struct Cluster {
    // std::chrono::system_clock::time_point f_timestamp;  // Timestamp of the cluster
    std::string f_timestamp;  // Timestamp of the cluster
    uint32_t f_id;  // Cluster ID
    vector<vector<float>> data;  // Data points in the cluster (e.g., [x, y, sensor_id])
    uint32_t heading;  // Estimated heading angle
    string status;  // Status of the object (e.g., 'moving', 'stopped')
    KalmanFilter filter_state;  // Kalman filter for state estimation (e.g., heading)

    // Constructor to initialize the Kalman filter with specific parameters
    Cluster() {
        double processNoise = 0.001;  // Small process noise
        double measurementNoise = 0.01;  // Measurement noise (tune based on your sensor)
        double initialEstimate = 0.0;  // Initial heading angle guess
        double initialCovariance = 1.0;  // Initial uncertainty

        // Create Kalman Filter object
        filter_state.set_parameters(processNoise, measurementNoise, initialEstimate, initialCovariance);
    }
};

// Function to extract sensor ID from camera ID (e.g., "cam_1" -> 1)
int extract_sensor_id(const string& cam_id) {
    size_t pos = cam_id.find('_');
    
    // If underscore exists, extract the part after the underscore
    if (pos != string::npos) {
        string id_str = cam_id.substr(pos + 1); // Get substring after '_'
        return stoi(id_str);  // Convert to integer
    } else {
        cerr << "Invalid cam_id format!" << endl;
    }

    return -1;
}

// Function to calculate Euclidean distance between two points
double calculate_distance(float x1, float y1, float x2, float y2) {
    return sqrt(pow(x1 - x2, 2) + pow(y1 - y2, 2));
}

// Function to parse a timestamp in JSON format (e.g., "2021-03-12 12:30:00.123")
std::chrono::system_clock::time_point parse_timestamp_json(const std::string& timestamp) {
    std::tm tm = {};
    std::stringstream ss(timestamp);
    
    // Parse the date and time part
    ss >> std::get_time(&tm, "%Y-%m-%d %H:%M:%S");

    // Extract milliseconds
    size_t ms_pos = timestamp.find('.');
    int milliseconds = 0;
    if (ms_pos != std::string::npos) {
        milliseconds = std::stoi(timestamp.substr(ms_pos + 1, 3)); // Get first 3 digits after decimal
    }

    // Convert std::tm to time_t and then to time_point
    auto time_point = std::chrono::system_clock::from_time_t(std::mktime(&tm));
    
    // Add the milliseconds to the time_point
    time_point += std::chrono::milliseconds(milliseconds);
    
    return time_point;
}

// Function to parse timestamp from CSV format (e.g., "2021-03-12T12:30:00.123")
std::chrono::system_clock::time_point parse_timestamp_csv(const std::string& timestamp) {
    std::tm tm = {};
    std::stringstream ss(timestamp);
    
    // Parse the date and time part
    ss >> std::get_time(&tm, "%Y-%m-%dT%H:%M:%S");

    // Extract milliseconds
    size_t ms_pos = timestamp.find('.');
    int milliseconds = 0;
    if (ms_pos != std::string::npos) {
        milliseconds = std::stoi(timestamp.substr(ms_pos + 1, 3)); // Get first 3 digits after decimal
    }

    // Convert std::tm to time_t and then to time_point
    auto time_point = std::chrono::system_clock::from_time_t(std::mktime(&tm));
    
    // Add the milliseconds to the time_point
    time_point += std::chrono::milliseconds(milliseconds);
    
    return time_point;
}

// Function to format a timestamp into a string with milliseconds
std::string format_timestamp(std::chrono::system_clock::time_point time_point) {
    // Convert time_point back to time_t
    std::time_t t = std::chrono::system_clock::to_time_t(time_point);
    
    // Convert time_t to tm struct
    std::tm tm = *std::localtime(&t);

    // Format and return the string with milliseconds
    std::stringstream ss;
    ss << std::put_time(&tm, "%Y-%m-%d %H:%M:%S");
    
    // Add milliseconds part
    auto milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(time_point.time_since_epoch()).count() % 1000;
    ss << "." << std::setw(3) << std::setfill('0') << milliseconds;
    
    return ss.str();
}

// Function to find the closest row in a CSV file based on timestamp comparison
std::string find_closest_row(const std::string& input_timestamp, const std::string& filename) {
    // Parse the input timestamp
    auto input_time_point = parse_timestamp_json(input_timestamp);
    
    // Open the CSV file
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Could not open the file!" << std::endl;
        return "";
    }

    std::string line;
    std::string closest_row;
    std::chrono::system_clock::time_point closest_timestamp;
    long long closest_diff = LLONG_MAX; // Start with a very large difference

    // Read the file line by line
    //Skip First Heading Line from CSV
    std::getline(file, line);

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string timestamp_str;
        
        // Read the timestamp in the row (first column in the CSV)
        std::getline(ss, timestamp_str, ',');

        // Parse the row's timestamp
        auto row_time_point = parse_timestamp_csv(timestamp_str);
        
        // Calculate the absolute difference between the timestamps
        long long diff;
        
        if (row_time_point > input_time_point)
        diff = std::chrono::duration_cast<std::chrono::microseconds>(row_time_point - input_time_point).count();
        else
        diff = std::chrono::duration_cast<std::chrono::microseconds>(input_time_point - row_time_point).count();

        // If this timestamp is closer, update the closest timestamp and row
        if (diff < closest_diff) {
            closest_diff = diff;
            closest_row = line;
            closest_timestamp = row_time_point;
        }
    }

    file.close();
    return closest_row;
}

// Function to extract heading and status from a row in the CSV file
void get_heading_and_status(const std::string& row, float& heading, std::string& status) {
    std::stringstream ss(row);
    std::string value;
    int column_index = 1;

    // Iterate through the columns in the row, split by commas
    while (std::getline(ss, value, ',')) {
        // When we reach the fourth value, return it
        if (column_index == 4) {
            heading = stof(value);  // Heading
        } else if (column_index == 5) {
            status = value;  // Status
            return;
        }
        column_index++;
    }
}

// Function to save clustered data into a CSV file
void save_cluster_data_to_csv(const string& filename, const vector<Cluster>& clusters) {
    // Open the CSV file for writing
    ofstream file(filename, std::ios::out | std::ios::trunc);
    
    // Check if the file is open
    if (!file.is_open()) {
        cerr << "Could not open the file for writing!" << endl;
        return;
    }

    // Write the header to the CSV file
    file << "f_timestamp,f_id,cluster_data,heading,status" << endl;

    // Iterate through the clusters and write their data to the file
    for (const auto& cluster : clusters) {
        // Format the timestamp as a string
        // std::string timestamp = format_timestamp(cluster.f_timestamp);
        std::string timestamp = cluster.f_timestamp;

        // Prepare the cluster data as a string (positions in the form of [x,y,sensor_id])
        stringstream cluster_data_stream;
        cluster_data_stream << "[";  // Start the cluster data list
        for (const auto& position : cluster.data) {
            cluster_data_stream << "[" << position[0] << "," << position[1] << "," << position[2] << "], ";
        }
        cluster_data_stream << "]";  // End the cluster data list
        string cluster_data_str = cluster_data_stream.str();

        // Write the cluster information to the file
        file << timestamp << ","  // Timestamp
             << cluster.f_id << ","  // Cluster ID
             << "\"" << cluster_data_str << "\","  // Cluster data
             << cluster.heading << ","  // Heading
             << "\"" << cluster.status << "\"" << endl;  // Status
    }

    // Close the file
    file.close();
}

// Main function that performs data fusion and clustering
int main() {
    // File names
    std::string json_file_name = "task_cam_data.json";
    std::string csv_file_name = "task_imu.csv"; 

    // Open JSON file for reading
    ifstream json_file(json_file_name);
    json root;
    vector<Cluster> fused_data;  // Container for storing fused data
    int cluster_f_id = 0;  // Cluster ID counter

    // Iterate through each entry in the root JSON object
    std::string line;
    while (getline(json_file, line))
    {
        std::stringstream ss(line);

        ss >> root;
        for (auto& entry : root.items()) {
            std::string timestamp = entry.value()["timestamp"];
            std::string cam_id = entry.key();
            int sensor_id = extract_sensor_id(cam_id);
            const json& cam_data = entry.value();

            // Extract object classes and positions
            const json& object_classes = cam_data["object_classes"];
            const json& object_positions_x_y = cam_data["object_positions_x_y"];
            
            // Iterate through the objects detected by the camera
            for (size_t i = 0; i < object_classes.size(); ++i) {
                if (object_classes[i] == "forklift") {
                    const json& position = object_positions_x_y[i];
                    float position_x = position[0];
                    float position_y = position[1];

                    bool clustered = false;

                    // Check if this object can be added to an existing cluster
                    for (auto& cluster : fused_data) {
                        for (const auto& pos : cluster.data) {
                            // If the position is close enough, add it to the cluster
                            if (calculate_distance(pos[0], pos[1], position_x, position_y) <= 2.0) {
                                cluster.data.push_back({position_x, position_y, sensor_id});
                                clustered = true;

                                // Find the closest heading and status for this object from the IMU data
                                const std::string row = find_closest_row(timestamp, csv_file_name);
                                float heading = 0;
                                get_heading_and_status(row, heading, cluster.status);
                                cluster.heading = cluster.filter_state.update(heading);

                                // Save the updated cluster data to CSV
                                save_cluster_data_to_csv("data_fusion.csv", fused_data);
                                break;
                            }
                        }
                        if (clustered) break;
                    }

                    // If the object wasn't clustered, create a new cluster
                    if (!clustered) {
                        Cluster new_cluster;
                        // new_cluster.f_timestamp = parse_timestamp_json(timestamp);
                        new_cluster.f_timestamp = timestamp;
                        new_cluster.f_id = ++cluster_f_id;
                        new_cluster.data.push_back({position_x, position_y, sensor_id});

                        // Find the closest heading and status for the new object from the IMU data
                        const std::string row = find_closest_row(timestamp, csv_file_name);
                        float heading = 0;
                        get_heading_and_status(row, heading, new_cluster.status);
                        new_cluster.heading = new_cluster.filter_state.update(heading);

                        // Add the new cluster to the fused data
                        fused_data.push_back(new_cluster);
                        save_cluster_data_to_csv("data_fusion.csv", fused_data);
                    }
                }
            }
        }
    }

    return 0;
}
