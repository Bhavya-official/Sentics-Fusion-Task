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
#include "json.hpp"  // Include nlohmann::json for JSON parsing

using json = nlohmann::json;
using namespace std;

// Kalman Filter class for estimating the state of an object (e.g., heading angle)
class KalmanFilter {
public:
    // Constructor initializing the Kalman filter with given noise values and initial estimates
    KalmanFilter(double processNoise, double measurementNoise, double initialEstimate, double initialCovariance)
        : Q(processNoise), R(measurementNoise), x_hat(initialEstimate), P(initialCovariance) {}

    // Default constructor
    KalmanFilter() = default;

    // Function to update the filter with a new measurement (returns the updated state)
    double update(double z) {
        double A = 1;  // State transition matrix
        double B = 0;  // Control input matrix
        double u = 0;  // Control input (not used here)
        
        double x_hat_pred = A * x_hat + B * u;  // predicted state estimate
        double P_pred = A * P * A + Q;  // predicted covariance estimate

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

// Structure to represent a cluster of sensor data
struct Cluster {
    uint64_t f_timestamp;  // Timestamp of the cluster
    uint32_t f_id;  // Cluster ID
    uint32_t num_entries;  // Number of entries in the cluster
    std::string cluster_data_str;  // String representing the data for the cluster
    uint32_t heading;  // Estimated heading angle
    string status;  // Status of the object (e.g., 'moving', 'stopped')
    KalmanFilter filter_state;  // Kalman filter for state estimation

    Cluster() {
        reset();  // Initialize the cluster
    }

    // Constructor to initialize the Kalman filter with specific parameters
    void reset() {
        f_timestamp = 0;
        f_id = 0;
        heading = 0;
        num_entries = 0;
        status = "";
        cluster_data_str = "";

        // Default parameters for the Kalman filter
        double processNoise = 0.001;
        double measurementNoise = 0.01;
        double initialEstimate = 0.0;
        double initialCovariance = 1.0;

        filter_state.set_parameters(processNoise, measurementNoise, initialEstimate, initialCovariance);  // Set filter parameters
    }
};

// Structure to represent a CSV row with sensor data
struct CSVRow {
    uint64_t timestamp;  // Timestamp of the row
    uint32_t id;  // ID of the object
    int yaw;  // Yaw angle of the object
    uint32_t heading;  // Heading angle
    bool state;  // State of the object (moving or stationary)
    float acceleration;  // Acceleration of the object
};

// Helper function to convert datetime to epoch time in microseconds
uint64_t convert_datetime_to_epoch(const string& datetime, const string& datetime_format) {
    std::istringstream ss(datetime);
    std::tm tm = {};
    ss >> std::get_time(&tm, datetime_format.c_str());

    uint64_t epoch_time_microseconds = 0;
    if (!ss.fail()) {
        std::time_t epoch_time = std::mktime(&tm);
        size_t dot_pos = datetime.find('.');
        if (dot_pos != std::string::npos) {
            std::string fractional_part = datetime.substr(dot_pos + 1);
            int fractional_value = std::stoi(fractional_part);
            int fractional_digits = fractional_part.length();
            if (fractional_digits == 3) fractional_value *= 1000;
            epoch_time_microseconds = epoch_time * 1000000 + fractional_value;
        }
    }
    return epoch_time_microseconds;
}

// Helper function to convert epoch time in microseconds to datetime string
std::string convert_epoch_to_datetime(uint64_t epoch_time_microseconds) {
    std::chrono::seconds sec(epoch_time_microseconds / 1000000);
    std::chrono::microseconds micros(epoch_time_microseconds % 1000000);

    std::time_t epoch_time_seconds = sec.count();
    std::tm* time_info = std::localtime(&epoch_time_seconds);

    std::ostringstream oss;
    oss << std::put_time(time_info, "%Y-%m-%d %H:%M:%S");
    oss << "." << std::setw(6) << std::setfill('0') << micros.count();

    return oss.str();
}

// Function to read CSV file into a vector of CSVRow structures
int read_csv(const std::string& filename, std::vector<CSVRow>& rows) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Unable to Open CSV File" << std::endl;
        return -1;  // Return error code if file can't be opened
    }

    std::string line;
    std::getline(file, line);  // Skip header row
    while (std::getline(file, line)) {
        std::istringstream ss(line);
        std::string timestampStr, idStr, yawStr, headingStr, stateStr, accelerationStr;

        // Parse each CSV column and convert to appropriate types
        std::getline(ss, timestampStr, ',');
        std::getline(ss, idStr, ',');
        std::getline(ss, yawStr, ',');
        std::getline(ss, headingStr, ',');
        std::getline(ss, stateStr, ',');
        std::getline(ss, accelerationStr, ',');

        CSVRow row;
        row.timestamp = convert_datetime_to_epoch(timestampStr, "%Y-%m-%dT%H:%M:%S");
        row.id = std::stoul(idStr);
        row.yaw = std::stoi(yawStr);
        row.heading = std::stoul(headingStr);
        row.state = (stateStr == "STANDING") ? 0 : 1;
        row.acceleration = std::stof(accelerationStr);

        rows.push_back(row);  // Add row to the vector
    }

    return 0;  // Return success
}

// Function to extract sensor ID from camera ID (e.g., "cam_1" -> 1)
int extract_sensor_id(const string& cam_id) {
    size_t pos = cam_id.find('_');
    if (pos != string::npos) {
        string id_str = cam_id.substr(pos + 1);
        return stoi(id_str);  // Extract integer after the underscore
    } else {
        cerr << "Invalid cam_id format!" << endl;
    }
    return -1;  // Return error if format is invalid
}

// Function to calculate distance between two timestamps
double calculate_distance(uint64_t t1, uint64_t t2) {
    return t2 - t1;
}

// Function to find the closest row in a CSV file based on timestamp comparison
CSVRow find_closest_row(const std::vector<CSVRow>& rows, uint64_t target_timestamp) {
    if (rows.empty()) throw std::invalid_argument("The vector is empty.");

    size_t left = 0;
    size_t right = rows.size() - 1;

    // Binary search to find the closest timestamp
    while (left < right) {
        size_t mid = left + (right - left) / 2;
        if (rows[mid].timestamp == target_timestamp) {
            return rows[mid];  // Found exact match
        } else if (rows[mid].timestamp < target_timestamp) {
            left = mid + 1;
        } else {
            right = mid;
        }
    }

    // Return the closest row
    if (left == 0) return rows[0];
    if (left == rows.size()) return rows[rows.size() - 1];

    uint64_t diff1 = std::abs(static_cast<int64_t>(rows[left].timestamp - target_timestamp));
    uint64_t diff2 = std::abs(static_cast<int64_t>(rows[left - 1].timestamp - target_timestamp));

    return (diff1 < diff2) ? rows[left] : rows[left - 1];  // Return closest row
}

// Function to save clustered data into a CSV file
void save_cluster_data_to_csv(ofstream& file, const Cluster& cluster) {
    if (!file.is_open()) {
        cerr << "Could not open the file for writing!" << endl;
        return;
    }

    std::string timestamp_str = convert_epoch_to_datetime(cluster.f_timestamp);
    file << "\"" << timestamp_str.c_str() << "\","
         << cluster.f_id << ",\""
         << cluster.cluster_data_str << "\","
         << cluster.heading << ",\""
         << cluster.status << "\"" << endl;
}

// Function to update the current cluster with new sensor data
void update_cluster(Cluster& last_cluster, const CSVRow& row, const json& cam_data, int sensor_id) {
    last_cluster.heading = last_cluster.filter_state.update(row.heading);  // Update heading using Kalman filter

    // Process object positions in camera data
    for (size_t i = 0; i < cam_data["object_classes"].size(); ++i) {
        const json& position = cam_data["object_positions_x_y"][i];
        float position_x = position[0];
        float position_y = position[1];

        last_cluster.f_timestamp += row.timestamp;  // Update timestamp
        last_cluster.cluster_data_str += "[" + std::to_string(position_x) + "," + std::to_string(position_y) + "," + std::to_string(sensor_id) + "],";  // Append data to cluster
        last_cluster.num_entries++;
    }
}

// Function to finalize the current cluster and save to CSV
void finalize_cluster(Cluster& last_cluster, int& cluster_f_id, const std::vector<CSVRow>& csv_rows, std::ofstream& fuse_file) {
    last_cluster.f_id = ++cluster_f_id;  // Increment and assign cluster ID
    uint64_t avg_timestamp = last_cluster.f_timestamp / last_cluster.num_entries;  // Calculate average timestamp
    CSVRow avg_time_row = find_closest_row(csv_rows, avg_timestamp);  // Find closest CSV row for average timestamp
    last_cluster.f_timestamp = avg_timestamp;
    last_cluster.status = avg_time_row.state == 0 ? "STANDING" : "DRIVING";  // Set status based on state

    if (!last_cluster.cluster_data_str.empty()) {
        last_cluster.cluster_data_str.pop_back();  // Remove trailing comma
    }
    last_cluster.cluster_data_str += "]";  // Close the data list

    save_cluster_data_to_csv(fuse_file, last_cluster);  // Save the cluster data to the CSV file
}

// Function to reset the current cluster
void reset_cluster(Cluster& last_cluster, const CSVRow& row, const json& cam_data, int sensor_id) {
    last_cluster.reset();  // Reset the cluster to initial state
    last_cluster.cluster_data_str = "[";  // Start new cluster data

    update_cluster(last_cluster, row, cam_data, sensor_id);  // Add first entry to the new cluster
}

// Function to process and fuse sensor data from JSON and CSV
void process_sensor_data(ifstream& json_file, const vector<CSVRow>& csv_rows, ofstream& fuse_file) {
    json root;
    uint64_t prev_timestamp = 0;
    bool is_first_entry = true;
    Cluster last_cluster;
    int cluster_f_id = 0;

    int line_no = 0;

    string line;
    while (getline(json_file, line)) {
        std::stringstream ss(line);
        ss >> root;  // Parse JSON data

        std::cout << "Processing Entry Number: " << ++line_no << std::endl;
        for (auto& entry : root.items()) {
            uint64_t timestamp = convert_datetime_to_epoch(entry.value()["timestamp"], "%Y-%m-%d %H:%M:%S");
            string cam_id = entry.key();
            int sensor_id = extract_sensor_id(cam_id);  // Extract sensor ID from camera ID
            const json& cam_data = entry.value();
        
            if (is_first_entry) {
                is_first_entry = false;
                prev_timestamp = timestamp;
            }
        
            const CSVRow& row = find_closest_row(csv_rows, timestamp);  // Find closest CSV row by timestamp
        
            // Update or finalize cluster based on timestamp distance
            if (calculate_distance(prev_timestamp, timestamp) < 2000000.0) {
                update_cluster(last_cluster, row, cam_data, sensor_id);
            } else {
                finalize_cluster(last_cluster, cluster_f_id, csv_rows, fuse_file);
                reset_cluster(last_cluster, row, cam_data, sensor_id);
                prev_timestamp = timestamp;
            }
        }
    }
    finalize_cluster(last_cluster, cluster_f_id, csv_rows, fuse_file);  // Finalize the last cluster
}

// Main function to load the files, process the sensor data, and save the fused output
int main() {
    std::string json_file_name = "task_cam_data.json";  // Input JSON file
    std::string csv_file_name = "task_imu.csv";  // Input CSV file
    std::string fused_csv_file = "data_fusion.csv";  // Output CSV file for fused data

    ifstream json_file(json_file_name);  // Open JSON file
    vector<CSVRow> csv_rows;  // Vector to hold CSV rows
    
    if (-1 == read_csv(csv_file_name, csv_rows)) {  // Read CSV and check for errors
        return -1;  // Return error if CSV read fails
    }

    ofstream fuse_file(fused_csv_file, std::ios::out | std::ios::trunc);  // Open output file for fused data

    fuse_file << "f_timestamp,f_id,cluster_data,heading,state" << std::endl;

    process_sensor_data(json_file, csv_rows, fuse_file);  // Process and fuse the sensor data

    fuse_file.close();  // Close the output file
    return 0;  // Successful execution
}
