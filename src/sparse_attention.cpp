#include <iostream>
#include <cassert>
// to locate this header file: execute torch.utils.cpp_extension.include_paths() in Python
#include <torch/extension.h>
#include <chrono>
#include <ctime>


bool DEBUG = false;
bool VERBOSE = false;
bool PROFILING = true;


void printFlatIntTensor(torch::Tensor t) {
    assert(t.dim() == 1);
    for (int i = 0; i < t.size(0); i++) {
        std::cout << t.index({i}).item<int>() << " ";
    }
}

void printFlatFloatTensor(torch::Tensor t) {
    assert(t.dim() == 1);
    for (int i = 0; i < t.size(0); i++) {
        std::cout << t.index({i}).item<float>() << " ";
    }
}


class BallTree;
using BallTreePtr = std::shared_ptr<BallTree>;
class BestMatches;
using BestMatchesPtr = std::shared_ptr<BestMatches>;

// A class to store the best k matches for this query (i.e. with the k highest scores).
// Format (score=inner_product, value=index)
// The matches are stored in a min-heap priority queue.
// The use of greater allows the worst scores to be stored at the root
using ScoreValuePair = std::pair<float, int>;
auto greater = [](ScoreValuePair left, ScoreValuePair right) { return left.first > right.first;};
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

    // Get the best k matches, as an (ordered) k-element torch tensor
    torch::Tensor getMatches() {
        torch::Tensor result = torch::zeros({k}, torch::kInt32);
        assert(pq.size() == k);
        for (int i = k - 1; i >= 0; i--) {
            result[i] = pq.top().second;
            pq.pop();
        }
        return result;
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
    torch::Tensor center;
    bool isLeafNode;
    torch::Tensor data;      // (N,D) for leaf nodes, null for non-leaf nodes
    torch::Tensor indices;   // (N,) for leaf nodes, null for non-leaf nodes
    int depth;

    // Constructor for empty node
    BallTree(torch::Tensor center, float radius, int depth)
        : pLeft(nullptr), pRight(nullptr), radius(radius), center(center), isLeafNode(false), depth(depth) {}

    // The maximum inner product that a query q can have with any point in the ball
    // This is ⟨q, center⟩ + ‖q‖ * radius
    float maxInnerProduct(torch::Tensor query, float query_norm) {
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
    std::pair<int,int> query(torch::Tensor query, float query_norm, BestMatchesPtr bestMatchesSoFar, int k) {
        // for debugging
        if (DEBUG) {
            std::string indent = std::string(2*depth, ' ');
            std::cout << indent << "->";
            if (isLeafNode) {
                printFlatIntTensor(indices);
            }
            std::cout << std::endl;
        }
                
        // If this is a leaf node, search the data
        if (isLeafNode) {
            // Compute the inner products of the query with all points in the data
            int num_keys = indices.size(0);
            torch::Tensor innerProducts = data.mm(query.unsqueeze(1)).squeeze(1);
            // Add the best matches to the bestMatchesSoFar object
            for (int i = 0; i < innerProducts.size(0); i++) {
                float score = innerProducts.index({i}).item<float>();
                int value = indices.index({i}).item<int>();
                bestMatchesSoFar->add(score, value);
            }
            return {1, num_keys};
        
        // If this is not a leaf node, do a guided search
        // IPUB is short for Inner Product Upper Bound
        } else {
            int num_nodes_searched = 1;
            int num_keys_searched = 0;

            float left_IPUB = pLeft->maxInnerProduct(query, query_norm);
            float right_IPUB = pRight->maxInnerProduct(query, query_norm);
            BallTreePtr most_promising_child;
            BallTreePtr least_promising_child;
            float most_promising_child_IPUB;
            float least_promising_child_IPUB;
            if (left_IPUB > right_IPUB) {
                most_promising_child = pLeft;
                least_promising_child = pRight;
                most_promising_child_IPUB = left_IPUB;
                least_promising_child_IPUB = right_IPUB;
            } else {
                most_promising_child = pRight;
                least_promising_child = pLeft;
                most_promising_child_IPUB = right_IPUB;
                least_promising_child_IPUB = left_IPUB;
            }
            
            // Guided search: search the most promising child first
            // But only search if it can improve on the best matches so far
            if (most_promising_child_IPUB > bestMatchesSoFar->lowerBound()) {
                auto [num_nodes_searched_, num_keys_searched_] =
                    most_promising_child->query(query, query_norm, bestMatchesSoFar, k);
                num_nodes_searched += num_nodes_searched_;
                num_keys_searched += num_keys_searched_;
            }
            if (least_promising_child_IPUB > bestMatchesSoFar->lowerBound()) {
                auto [num_nodes_searched_, num_keys_searched_] =
                    least_promising_child->query(query, query_norm, bestMatchesSoFar, k);
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
    void printMIPs(torch::Tensor query, float query_norm) {
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

std::pair<torch::Tensor, torch::Tensor> getTwoFarPoints(torch::Tensor data);

BallTreePtr buildBallTree(torch::Tensor data, torch::Tensor indices, int maxLeafSize, int depth=0) {
    // data: (N, D)
    // indices: (N,)
    int N = data.size(0);
    assert (N == indices.size(0));

    torch::Tensor center = data.mean(0);
    float radius = (data - center).pow(2).sum(1).sqrt().max().item<float>();
    BallTree currentNode(center, radius, depth);

    // for debugging
    if (DEBUG) {
        std::string indent = std::string(2*depth, ' ');
        std::cout << indent << indices.size(0) << ": ";
        if (false) {
            printFlatIntTensor(indices);
        }
        std::cout << radius << std::endl;
    }

    if (N <= maxLeafSize) {
        currentNode.isLeafNode = true;
        currentNode.data = data;
        currentNode.indices = indices;
    } else {
        // split the data into two parts
        auto [P, Q] = getTwoFarPoints(data);
        torch::Tensor P_dists = (data - P).pow(2).sum(1);
        torch::Tensor Q_dists = (data - Q).pow(2).sum(1);

        torch::Tensor left_mask = P_dists < Q_dists;
        torch::Tensor right_mask = ~left_mask;

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
    (P, Q): torch.Tensor, torch.Tensor
*/
std::pair<torch::Tensor, torch::Tensor> getTwoFarPoints(torch::Tensor data) {
    // pick a random point x in data
    torch::Tensor rand_idx = torch::randint(0, data.size(0), {1});
    torch::Tensor x = data.index({rand_idx});

    // find the point P in data that is farthest from x
    torch::Tensor x_dists = (data - x).pow(2).sum(1);
    torch::Tensor P_idx = x_dists.argmax();
    torch::Tensor P = data.index({P_idx});
    torch::Tensor P_dists = (data - P).pow(2).sum(1);

    // find the point Q in data that is farthest from P
    torch::Tensor Q_idx = P_dists.argmax();
    torch::Tensor Q = data.index({Q_idx});

    return std::make_pair(P, Q);
}


torch::Tensor nearestKKeys(torch::Tensor queries, torch::Tensor keys, int k, int maxLeafSize) {
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

    torch::Tensor result = torch::zeros({B, H, Nq, k}, torch::kInt32);

    // declarations of profiling variables
    std::chrono::duration<double> buildballtree_seconds, query_seconds;
    std::chrono::time_point<std::chrono::system_clock> start, end;
    for (int b = 0; b < B; b++) {
        for (int h = 0; h < H; h++) {
            // create ball tree of the keys
            if (PROFILING) {
                start = std::chrono::system_clock::now();
            }
            BallTreePtr ballTree = buildBallTree(keys.index({b, h}), torch::arange({Nk}), maxLeafSize, 0);
            BestMatchesPtr bestMatches = std::make_shared<BestMatches>(k);
            if (PROFILING) {
                end = std::chrono::system_clock::now();
                buildballtree_seconds += end - start;
            }

            if (PROFILING) {
                start = std::chrono::system_clock::now();
            }
            // query the ball tree for every query
            for (int n = 0; n < Nq; n++) {
                if (VERBOSE) {
                    std::cout << "Querying (" << b << ", " << h << ", " << n << "): " << std::endl;
                }
                torch::Tensor query = queries.index({b, h, n});
                float query_norm = query.norm().item<float>();
                
                // the actual query
                auto [num_nodes_searched, num_keys_searched]
                    = ballTree->query(query, query_norm, bestMatches, k);
                
                // result processing
                float kth_best_dot_product = bestMatches->lowerBound();
                torch::Tensor matches = bestMatches->getMatches();
                result.index_put_({b, h, n}, matches);
                if (DEBUG) {
                    ballTree->printMIPs(query, query_norm);
                }
                if (VERBOSE) {
                    std::cout << "Succeeded" << std::endl;
                    std::cout << num_keys_searched << "/" << Nk << " keys searched, and " << num_nodes_searched << "/" << ballTree->numNodes() << " nodes searched" << std::endl;
                    std::cout << "Resulting k'th best dot product: " << kth_best_dot_product << std::endl;
                }
            }
            if (PROFILING) {
                end = std::chrono::system_clock::now();
                query_seconds += end - start;
            }
        }
    }

    if (PROFILING) {
        std::cout << "buildBallTree time: " << buildballtree_seconds.count() << "s" << std::endl;
        std::cout << "query time: " << query_seconds.count() << "s" << std::endl;
        std::cout << "total time: " << (buildballtree_seconds + query_seconds).count() << "s" << std::endl;
    }

    return result;
}


torch::Tensor myDummyFunction(torch::Tensor a, torch::Tensor b) {
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
    torch::Tensor matches = bestMatches.getMatches();
    assert(matches[0].item<int>() == 10);
    assert(matches[1].item<int>() == 9);
    assert(matches[2].item<int>() == 8);
    std::cout << matches << std::endl;
}

void test_nearestKKeys() {
    torch::Tensor queries = torch::randn({2, 3, 500, 5});
    torch::Tensor keys = torch::randn({2, 3, 1000, 5});
    torch::Tensor result = nearestKKeys(queries, keys, 3, 3);
}


void run_all_tests() {
    test_best_matches();
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
}