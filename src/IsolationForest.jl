# adapted from LibIsolationForest
# see: https://github.com/msimms/LibIsolationForest/blob/master/julia/IsolationForest.jl
module IsolationForest

Feature = AbstractVector{T} where T <: AbstractFloat
Dataset = AbstractVector{<:Feature}

# Tree node, used internally.
mutable struct Node
    featureName::Int
    splitValue::AbstractFloat
    left::Union{Node, Nothing}
    right::Union{Node, Nothing}
end

# Isolation Forest.
mutable struct Forest
    numTrees::Int
    subSamplingSize::Int
    numFeatures::Int
    trees::AbstractArray{Union{Node, Nothing}}
    function Forest(numTrees, subSamplingSize, numFeatures, featureValues)
        forest = new(numTrees, subSamplingSize, numFeatures, [])
        for i = 1:forest.numTrees
            featureValues = deepcopy(featureValues)
            tree = create_tree(forest, featureValues, 0)
            if tree !== nothing
                push!(forest.trees, tree)
            end
        end
        return forest
    end
end

# Creates and returns a single tree. As this is a recursive function, depth indicates the current depth of the recursion.
function create_tree(forest::Forest, feature_values::Dataset, depth::Int)
    # Sanity check
    if forest.numFeatures <= 1
        return nothing
    end

    # If we've exceeded the maximum desired depth, then stop.
    if (forest.subSamplingSize > 0) && (depth >= forest.subSamplingSize)
        return nothing
    end

    # Randomly select a feature.
    randomly_selected_feature = rand(1:forest.numFeatures)

    # Randomly select a split value.
    feature_value_set = feature_values[randomly_selected_feature]
    feature_value_set_len = length(feature_value_set)

    if feature_value_set_len <= 1
        return nothing
    end
    split_value_index = rand(1:feature_value_set_len)
    split_value = feature_value_set[split_value_index]

    # Create a tree node to hold the split value.
    tree = Node(randomly_selected_feature, split_value, nothing, nothing)

    # Create two versions of the feature value set that we just used,
    # one for the left side of the tree and one for the right.
    temp_feature_values = feature_values

    # Create the left subtree.
    left_features = feature_value_set[1:split_value_index]

    temp_feature_values[randomly_selected_feature] = left_features
    tree.left = IsolationForest.create_tree(forest, temp_feature_values, depth + 1)

    # Create the right subtree.
    if split_value_index + 1 < feature_value_set_len
        right_features = feature_value_set[split_value_index + 1:feature_value_set_len]
        temp_feature_values[randomly_selected_feature] = right_features
        tree.right = IsolationForest.create_tree(forest, temp_feature_values, depth + 1)
    end

    return tree
end

# Scores the sample against the specified tree.
function score_sample_against_tree(tree::Node, features::Feature)
    depth = 0.0
    current_node = tree

    while current_node !== nothing
        found_feature = false

        # Find the next feature in the sample.
        for (current_feature_name, current_feature_value) in enumerate(features)

            # If the current node has the feature in question.
            if current_feature_name == current_node.featureName
                if current_feature_value < current_node.splitValue
                    current_node = current_node.left
                else
                    current_node = current_node.right
                end

                depth = depth + 1.0
                found_feature = true
                break
            end
        end

        # If the tree contained a feature not in the sample then take
        # both sides of the tree and average the scores together.
        if found_feature == false
            left_depth = depth + score_sample_against_tree(sample, current_node.left)
            right_depth = depth + score_sample_against_tree(sample, current_node.right)
            return (left_depth + right_depth) / 2.0
        end
    end

    return depth
end

# Scores the sample against the entire forest of trees. Result is the average path length.
function score_sample_against_forest(forest::Forest, features::Feature)
    num_trees = 0
    avg_path_len = 0.0

    for tree in forest.trees
        path_len = score_sample_against_tree(tree, features)
        if path_len > 0
            avg_path_len = avg_path_len + path_len
            num_trees = num_trees + 1
        end
    end

    if num_trees > 0
        avg_path_len = avg_path_len / num_trees
    end

    return avg_path_len
end

# Scores the sample against the entire forest of trees. Result is normalized so that values
# close to 1 indicate anomalies and values close to zero indicate normal values.
function H(i)
    return log(i) + 0.5772156649
end
function C(n)
    return 2 * H(n - 1) - (2 * (n - 1) / n)
end
function score_sample_against_forest_normalized(forest::Forest, features::Feature)

    # Compute the average path length for all valid trees.
    num_trees = 0
    avg_path_len = 0.0

    for tree in forest.trees
        path_len = score_sample_against_tree(tree, features)
        if path_len > 0
            avg_path_len = avg_path_len + path_len
            num_trees = num_trees + 1
        end
    end

    if num_trees > 0
        avg_path_len = avg_path_len / num_trees
    end

    # Normalize, per the original paper.
    score = 0.0
    if num_trees > 1.0
        exponent = -1.0 * (avg_path_len / C(num_trees))
        score = 2 ^ exponent
    end

    return score
end

end
