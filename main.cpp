#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <eigen3/Eigen/Dense>
#include <omp.h>

namespace py = pybind11;

#pragma GCC visibility push(hidden)
class KMeans {
public:
    KMeans(int n_clusters, int max_iters=100, double tol=1e-4, std::string algorithm="lloyd", bool use_kmeans_pp=true)
        : n_clusters(n_clusters), max_iters(max_iters), tol(tol), algorithm(algorithm), use_kmeans_pp(use_kmeans_pp) {}

    void fit(const py::array_t<double>& X) {
        py::buffer_info buf = X.request();
        const double *ptr = static_cast<double *>(buf.ptr);
        const int N = buf.shape[0]; // number of samples
        const int D = buf.shape[1]; // dimension of each sample

        std::vector<std::vector<double>> data(N, std::vector<double>(D));
        std::vector<int> labels(N);

        // Copy data from numpy array
        #pragma omp parallel for
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < D; ++j)
                data[i][j] = ptr[i * D + j];

        // Initialize centroids using kmeans++ or random initialization
        Eigen::MatrixXd data_matrix = stdVecToEigen(data);
        Eigen::MatrixXd centroids;
        if (use_kmeans_pp)
            centroids = kmeansPlusPlus(data_matrix);
        else
            centroids = randomInitialization(data_matrix);

        centroids_old = Eigen::MatrixXd::Zero(n_clusters, D); // initialize centroids_old

        for (int iter = 0; iter < max_iters; ++iter) {
            // Assign each data point to the nearest centroid
            labels = assignLabels(data_matrix, centroids);

            // Update centroids
            centroids = updateCentroids(data_matrix, labels);

            // Check for convergence
            bool converged = checkConvergence(centroids);

            if (converged) break;
            centroids_old = centroids;
        }

        // Convert centroids to numpy array
        py::array_t<double> centroids_array({n_clusters, D});
        auto centroids_ptr = centroids_array.mutable_unchecked<2>();

        #pragma omp parallel for
        for (int i = 0; i < n_clusters; ++i)
            for (int j = 0; j < D; ++j)
                centroids_ptr(i, j) = centroids(i, j);

        this->centroids = centroids_array;

        // Convert labels to numpy array
        py::array_t<int> labels_array(N);
        auto labels_ptr = labels_array.mutable_unchecked<1>();

        for (int i = 0; i < N; ++i)
            labels_ptr(i) = labels[i];

        this->labels = labels_array;
    }

    py::array_t<int> getLabels() const {
        return labels;
    }

    py::array_t<double> getCentroids() const {
        return centroids;
    }

private:
    int n_clusters;
    int max_iters;
    double tol;
    std::string algorithm;
    bool use_kmeans_pp;
    py::array_t<int> labels;
    py::array_t<double> centroids;
    Eigen::MatrixXd centroids_old;

    Eigen::MatrixXd stdVecToEigen(const std::vector<std::vector<double>>& data) {
        Eigen::MatrixXd matrix(data.size(), data[0].size());

        #pragma omp parallel for
        for (size_t i = 0; i < data.size(); ++i)
            for (size_t j = 0; j < data[0].size(); ++j)
                matrix(i, j) = data[i][j];
        return matrix;
    }

    std::vector<int> assignLabels(const Eigen::MatrixXd& data, const Eigen::MatrixXd& centroids) {
        const int N = data.rows();
        std::vector<int> labels(N);

        #pragma omp parallel for
        for (int i = 0; i < N; ++i) {
            double min_dist = std::numeric_limits<double>::max();
            int label = 0;
            for (int j = 0; j < n_clusters; ++j) {
                double dist = (data.row(i) - centroids.row(j)).squaredNorm();
                if (dist < min_dist) {
                    min_dist = dist;
                    label = j;
                }
            }
            labels[i] = label;
        }

        return labels;
    }

    Eigen::MatrixXd updateCentroids(const Eigen::MatrixXd& data, const std::vector<int>& labels) {
        const int N = data.rows();
        const int D = data.cols();

        Eigen::MatrixXd new_centroids = Eigen::MatrixXd::Zero(n_clusters, D);
        std::vector<int> counts(n_clusters, 0);

        for (int i = 0; i < N; ++i) {
            int label = labels[i];
            new_centroids.row(label) += data.row(i);
            counts[label]++;
        }

        for (int i = 0; i < n_clusters; ++i) {
            if (counts[i] > 0) new_centroids.row(i) /= counts[i];
        }

        return new_centroids;
    }

    Eigen::MatrixXd randomInitialization(const Eigen::MatrixXd& data) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(0, data.rows() - 1);

        Eigen::MatrixXd centroids(n_clusters, data.cols());
        for (int i = 0; i < n_clusters; ++i) {
            centroids.row(i) = data.row(dis(gen));
        }

        return centroids;
    }

    Eigen::MatrixXd kmeansPlusPlus(const Eigen::MatrixXd& data) {
        int N = data.rows();
        int D = data.cols();

        Eigen::MatrixXd centroids(n_clusters, D);

        // Randomly choose the first centroid
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(0, N - 1);
        centroids.row(0) = data.row(dis(gen));

        // Choose the remaining centroids using kmeans++
        for (int i = 1; i < n_clusters; ++i) {
            Eigen::VectorXd min_dist_sq(N);
            for (int j = 0; j < N; ++j) {
                min_dist_sq[j] = (data.row(j) - centroids.row(0)).squaredNorm();
                for (int k = 1; k < i; ++k) {
                    min_dist_sq[j] = std::min(min_dist_sq[j], (data.row(j) - centroids.row(k)).squaredNorm());
                }
            }
            std::discrete_distribution<> weighted_dist(min_dist_sq.data(), min_dist_sq.data() + N);
            centroids.row(i) = data.row(weighted_dist(gen));
        }

        return centroids;
    }

    bool checkConvergence(const Eigen::MatrixXd& centroids) {
        if (algorithm == "lloyd") {
            return (centroids - centroids_old).norm() < tol;
        } else {
            // Elkan algorithm
            // To be continued..., just return true for convergence
            return true;
        }
    }
};

PYBIND11_MODULE(kmeans, m) {
    py::class_<KMeans>(m, "KMeans")
        .def(py::init<int, int, double, std::string, bool>())
        .def("fit", &KMeans::fit)
        .def("getLabels", &KMeans::getLabels)
        .def("getCentroids", &KMeans::getCentroids);
}