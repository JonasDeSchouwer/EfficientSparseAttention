#include <iostream>
#include <cassert>
// to locate this header file: execute torch.utils.cpp_extension.include_paths() in Python
#include <torch/extension.h>
#include <chrono>
#include <ctime>
#include <omp.h>


#define DEBUG 0
#define VERBOSE 0
#define PROFILING 1


void printFlatIntTensor(at::Tensor t) {
    assert(t.dim() == 1);
    for (int i = 0; i < t.size(0); i++) {
        std::cout << t.index({i}).item<int>() << " ";
    }
}

void printFlatFloatTensor(at::Tensor t) {
    assert(t.dim() == 1);
    for (int i = 0; i < t.size(0); i++) {
        std::cout << t.index({i}).item<float>() << " ";
    }
}


class BallTree;
using BallTreePtr = std::shared_ptr<BallTree>;
class BestMatches;

// A class to store the best k matches for this query (i.e. with the k highest scores).
// Format (score=inner_product, value=index)
// The matches are stored in a min-heap priority queue.
// The use of greater allows the worst scores to be stored at the root
using ScoreValuePair = std::pair<float, int>;
auto greater = [](ScoreValuePair left, ScoreValuePair right) {return left.first > right.first;};
class BestMatches {
public:
    BestMatches(int k) : k(k), pq(greater) {}

    // Add a new match to the priority queue
    void add(float score, int value) {
        if (pq.size() < k) {
            pq.push({score, value});
        } else if (score > pq.top().first) {
            pq.pop();
            pq.push(ScoreValuePair(score, value));
        }
    }

    // Returns the lower bound of the best k matches
    float lowerBound() {
        if (pq.size() < k) {
            return -INFINITY;
        } else {
            return pq.top().first;
        }
    }

    // Put the best k matches in the result tensor
    void putMatchesInResult(at::TensorAccessor<int,1> result_accessor) {
        assert(pq.size() == k);
        for (int i = k - 1; i >= 0; i--) {
            result_accessor[i] = pq.top().second;
            pq.pop();
        }
    }

private:
    int k;
    std::priority_queue<ScoreValuePair, std::vector<ScoreValuePair>, decltype(greater)> pq;
};

class BallTree {
public:
    BallTreePtr pLeft;       // the left child
    BallTreePtr pRight;      // the right child
    float radius;
    at::Tensor center;
    bool isLeafNode;
    at::Tensor data;      // (N,D) for leaf nodes, null for non-leaf nodes
    at::Tensor indices;   // (N,) for leaf nodes, null for non-leaf nodes
    int depth;

    // Constructor for empty node
    BallTree(at::Tensor center, float radius, int depth)
        : pLeft(nullptr), pRight(nullptr), radius(radius), center(center), isLeafNode(false), depth(depth) {}

    // The maximum inner product that a query q can have with any point in the ball
    // This is ⟨q, center⟩ + ‖q‖ * radius
    float maxInnerProduct(at::Tensor query, float query_norm) {
        return query.dot(center).item<float>() + query_norm * radius;
    }

    
    /**
     * Query the ball tree for the k highest inner products with query.
     *
     * This function searches the ball tree for the best matches to the given query tensor.
     * Uses a branch-and-bound algorithm to prune the search space
     *
     * @param query A (D,) tensor to search for matches.
     * @param bestMatchesSoFar A pointer to the BestMatches object that stores the best matches found so far.
     * @param k The number of best matches to find.
     * @return A pair consisting of (# nodes searched, # keys searched).
     */
    std::pair<int,int> query(at::Tensor query, float query_norm, BestMatches& bestMatchesSoFar, int k) {
        // for debugging
        #if DEBUG
            std::string indent = std::string(2*depth, ' ');
            std::cout << indent << "->";
            if (isLeafNode) {
                printFlatIntTensor(indices);
            }
            std::cout << std::endl;
        #endif
                
        // If this is a leaf node, search the data
        if (isLeafNode) {
            // Compute the inner products of the query with all points in the data
            int num_keys = indices.size(0);
            at::Tensor innerProducts = data.mm(query.unsqueeze(1)).squeeze(1);

            // Access the raw data pointers for faster access in the loop
            auto innerProducts_a = innerProducts.accessor<float, 1>();
            auto indices_a = indices.accessor<int, 1>();

            // Add the best matches to the bestMatchesSoFar object
            for (int i = 0; i < num_keys; i++) {

                float score = innerProducts_a[i];
                int value = indices_a[i];
                bestMatchesSoFar.add(score, value);
            }
            return {1, num_keys};
        
        // If this is not a leaf node, do a guided search
        // IPUB is short for Inner Product Upper Bound
        } else {
            int num_nodes_searched = 1;
            int num_keys_searched = 0;

            float left_IPUB = pLeft->maxInnerProduct(query, query_norm);
            float right_IPUB = pRight->maxInnerProduct(query, query_norm);

            if (left_IPUB < right_IPUB) {
                // swap the children so that the most promising child is pLeft
                std::swap(pLeft, pRight);
            }
            
            // Guided search: search the most promising child (now left) first
            // But only search if it can improve on the best matches so far
            if (left_IPUB > bestMatchesSoFar.lowerBound()) {
                auto [num_nodes_searched_, num_keys_searched_] =
                    pLeft->query(query, query_norm, bestMatchesSoFar, k);
                num_nodes_searched += num_nodes_searched_;
                num_keys_searched += num_keys_searched_;
            }
            if (right_IPUB > bestMatchesSoFar.lowerBound()) {
                auto [num_nodes_searched_, num_keys_searched_] =
                    pRight->query(query, query_norm, bestMatchesSoFar, k);
                num_nodes_searched += num_nodes_searched_;
                num_keys_searched += num_keys_searched_;
            }
            return {num_nodes_searched, num_keys_searched};
        }
    }

    // Print the ball tree
    void print() {
        std::string indent = std::string(2*depth, ' ');
        std::cout << indent << "(" << center << ", " << radius << ")" << std::endl;
        if (isLeafNode) {
            std::cout << indent << "  " << indices << std::endl;
        } else {
            pLeft->print();
            pRight->print();
        }
    }

    // Print the MIPs of the query with every descendant of the ball tree
    void printMIPs(at::Tensor query, float query_norm) {
        std::string indent = std::string(2*depth, ' ');
        std::cout << indent << "(" << radius << ", " << indices.size(0) << "): " << maxInnerProduct(query, query_norm) << std::endl;
        if (!isLeafNode) {
            pLeft->printMIPs(query, query_norm);
            pRight->printMIPs(query, query_norm);
        }
    }

    // Compute the total number of nodes
    int numNodes() {
        if (isLeafNode) {
            return 1;
        } else {
            return 1 + pLeft->numNodes() + pRight->numNodes();
        }
    }
};

std::pair<at::Tensor, at::Tensor> getTwoFarPoints(at::Tensor data);

BallTreePtr buildBallTree(at::Tensor data, at::Tensor indices, int maxLeafSize, int depth=0) {
    // data: (N, D)
    // indices: (N,)
    int N = data.size(0);
    assert (N == indices.size(0));

    at::Tensor center = data.mean(0);
    float radius = (data - center).pow(2).sum(1).sqrt().max().item<float>();
    BallTree currentNode(center, radius, depth);

    // for debugging
    #if DEBUG
        std::string indent = std::string(2*depth, ' ');
        std::cout << indent << indices.size(0) << ": ";
        if (false) {
            printFlatIntTensor(indices);
        }
        std::cout << radius << std::endl;
    #endif

    if (N <= maxLeafSize) {
        currentNode.isLeafNode = true;
        currentNode.data = data;
        currentNode.indices = indices;
    } else {
        // split the data into two parts
        auto [P, Q] = getTwoFarPoints(data);

        // put the data into the left or right child
        at::Tensor left_mask = (data - P).pow(2).sum(1) < (data - Q).pow(2).sum(1);
        at::Tensor right_mask = ~left_mask;

        // TODO: optimization: free data memory and only remember left_mask and right_mask

        // get data and indices for left and right children
        currentNode.pLeft = buildBallTree(data.index({left_mask}), indices.index({left_mask}), maxLeafSize, depth+1);
        currentNode.pRight = buildBallTree(data.index({right_mask}), indices.index({right_mask}), maxLeafSize, depth+1);
    }

    return std::make_shared<BallTree>(currentNode);
}



/*
get 2 points (P,Q) in data that are far apart, but can be computed efficiently

returns:
    (P, Q): at.Tensor, at.Tensor
*/
std::pair<at::Tensor, at::Tensor> getTwoFarPoints(at::Tensor data) {
    // pick a random point x in data
    at::Tensor rand_idx = at::randint(
        0, data.size(0), {1},
        at::TensorOptions()
            .dtype(at::ScalarType::Int)
            .device(data.device())
    );
    at::Tensor x = data.index({rand_idx});

    // find the point P in data that is farthest from x
    at::Tensor x_dists = (data - x).pow(2).sum(1);
    at::Tensor P_idx = x_dists.argmax();
    at::Tensor P = data.index({P_idx});
    at::Tensor P_dists = (data - P).pow(2).sum(1);

    // find the point Q in data that is farthest from P
    at::Tensor Q_idx = P_dists.argmax();
    at::Tensor Q = data.index({Q_idx});

    return std::make_pair(P, Q);
}


at::Tensor nearestKKeys(at::Tensor queries, at::Tensor keys, int k, int maxLeafSize) {
    /*
    queries: (B, H, Nq, kq_dim)
    keys: (B, H, Nk, kq_dim)
    k: int
    return: (B, H, Nq, k)
    */ 
    int B = queries.size(0);
    int H = queries.size(1);
    int Nq = queries.size(2);
    int Nk = keys.size(2);
    
    assert (queries.size(0) == keys.size(0));
    assert (queries.size(1) == keys.size(1));
    // in general, no reason why N would need to be the same for queries and keys
    assert (queries.size(3) == keys.size(3));

    if (queries.device() != keys.device()) {
        std::cerr << "queries and keys must be on the same device" << std::endl;
        exit(1);
    }
    if (queries.device() != at::Device(at::kCPU)) {
        std::cout << "It seems your keys and queries are not on the CPU. This is not recommended for efficiency reasons." << std::endl;
    }

    at::Tensor query_norms = queries.norm(2, -1);
    auto query_norms_a = query_norms.accessor<float, 3>();

    at::Tensor result = at::empty(
        {B, H, Nq, k},
        at::TensorOptions()
            .dtype(at::ScalarType::Int)
            .device(queries.device())
    );
    auto result_a = result.accessor<int, 4>();

    // declarations of profiling variables
    #if PROFILING
        std::chrono::duration<double> buildballtree_seconds, query_seconds;
    #endif
    #pragma omp parallel for collapse(2)
    for (int b = 0; b < B; b++) {
        for (int h = 0; h < H; h++) {
            // create ball tree of the keys
            #if PROFILING
                std::chrono::time_point<std::chrono::system_clock> start, end;
                start = std::chrono::system_clock::now();
            #endif
            at::Tensor indices = at::arange({Nk},
                at::TensorOptions()
                    .device(keys.device())
                    .dtype(at::ScalarType::Int)
            );
            BallTreePtr ballTree = buildBallTree(
                keys.index({b, h}),
                indices,
                maxLeafSize,
                0
            );
            #if PROFILING
                end = std::chrono::system_clock::now();
                buildballtree_seconds += end - start;
            #endif

            int num_threads;
            #pragma omp parallel
            {
                #pragma omp single
                num_threads = omp_get_num_threads();
            }

            // --- query the ball tree for every query ---
            std::vector<BestMatches> bestMatchesVector(num_threads, BestMatches(k));
            #if PROFILING
                start = std::chrono::system_clock::now();
            #endif
            #pragma omp parallel for
            for (int n = 0; n < Nq; n++) {
                int thread_id = omp_get_thread_num();
                #if VERBOSE
                    std::cout << "Querying (" << b << ", " << h << ", " << n << "): " << std::endl;
                #endif
                at::Tensor query = queries.select(0, b).select(0, h).select(0, n);
                float query_norm = query_norms_a[b][h][n];
                
                // the actual query
                auto& bestMatches = bestMatchesVector[thread_id]; // Access the corresponding BestMatches object
                auto [num_nodes_searched, num_keys_searched]
                    = ballTree->query(query, query_norm, bestMatches, k);
                
                // result processing
                #if VERBOSE
                    float kth_best_dot_product = bestMatches.lowerBound();
                #endif
                bestMatches.putMatchesInResult(result_a[b][h][n]);

                #if DEBUG
                    ballTree->printMIPs(query, query_norm);
                #endif
                #if VERBOSE
                    std::cout << "Succeeded" << std::endl;
                    std::cout << num_keys_searched << "/" << Nk << " keys searched, and " << num_nodes_searched << "/" << ballTree->numNodes() << " nodes searched" << std::endl;
                    std::cout << "Resulting k'th best dot product: " << kth_best_dot_product << std::endl;
                #endif
            }
            #if PROFILING
                end = std::chrono::system_clock::now();
                query_seconds += end - start;
            #endif
        }
    }

    #if PROFILING
        std::cout << "buildBallTree time: " << buildballtree_seconds.count() << "s" << std::endl;
        std::cout << "query time: " << query_seconds.count() << "s" << std::endl;
        std::cout << "total time: " << (buildballtree_seconds + query_seconds).count() << "s" << std::endl;
    #endif

    return result;
}


at::Tensor myDummyFunction(at::Tensor a, at::Tensor b) {
    return a+b;
}



void test_best_matches() {
    BestMatches bestMatches(3);
    std::cout << bestMatches.lowerBound() << std::endl;
    bestMatches.add(1.0, 1);
    bestMatches.add(8.0, 8);
    bestMatches.add(2.0, 2);
    bestMatches.add(9.0, 9);
    bestMatches.add(7.0, 7);
    bestMatches.add(5.0, 5);
    bestMatches.add(10.0, 10);
    bestMatches.add(4.0, 4);
    bestMatches.add(3.0, 3);
    bestMatches.add(6.0, 6);
    at::Tensor matches = at::empty({3}, at::TensorOptions().dtype(at::ScalarType::Int));
    auto matches_a = matches.accessor<int, 1>();
    bestMatches.putMatchesInResult(matches_a);
    assert(matches[0].item<int>() == 10);
    assert(matches[1].item<int>() == 9);
    assert(matches[2].item<int>() == 8);
    std::cout << matches << std::endl;
}

void test_nearestKKeys() {
    at::Tensor queries = at::randn({2, 3, 1000, 5});
    at::Tensor keys = at::randn({2, 3, 1000, 5});
    at::Tensor result = nearestKKeys(queries, keys, 3, 3);
}

void test_buildBallTree() {
    at::Tensor data = at::randn({1000, 5});
    at::Tensor indices = at::arange(
        {1000}, 
        at::TensorOptions()
            .device(at::Device(at::kCPU))
            .dtype(at::ScalarType::Int)
    );
    BallTreePtr ballTree = buildBallTree(data, indices, 3);
}


void run_all_tests() {
    test_best_matches();
    test_buildBallTree();
    test_nearestKKeys();
}



PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("nearestKKeys", &nearestKKeys, "Find the nearest k keys for each query",
                py::arg("queries"), py::arg("keys"), py::arg("k"), py::arg("maxLeafSize"));
    m.def("myDummyFunction", &myDummyFunction, "A dummy function that adds two tensors",
                py::arg("a"), py::arg("b"));
    m.def("run_all_tests", &run_all_tests, "Run all tests");
    m.def("test_best_matches", &test_best_matches, "Test the BestMatches class");
    m.def("test_nearestKKeys", &test_nearestKKeys, "Test the nearestKKeys function");
    m.def("test_buildBallTree", &test_buildBallTree, "Test the buildBallTree function");
}