include("../IsolationForest.jl")

"""
    IForestDetector()

Determine the anomaly score of a sample based on their average path lengths on trees in a forest, see [1].

Parameters
----------
$(SCORE_UNSUPERVISED("IForest"))

References
----------
[1] Liu, Fei Tony, Kai Ming Ting, and Zhi-Hua Zhou. “Isolation Forest.” In 2008 Eighth IEEE International Conference on
Data Mining, 413–22. Pisa, Italy: IEEE, 2008.
"""
OD.@detector mutable struct IForestDetector <: UnsupervisedDetector
    num_trees::Int = 100
    sub_sampling_size::Int = 256
    normalize::Bool = false
end

struct IForestModel <: DetectorModel
    forest::IsolationForest.Forest
end

to_sample_set(X) = eachcol(X)
to_feature_set(X) = map(collect, eachrow(X))
make_scorer(forest, normalize::Bool) = normalize ?
    x -> IsolationForest.score_sample_against_forest_normalized(forest, x) :
    x -> IsolationForest.score_sample_against_forest(forest, x)

function OD.fit(detector::IForestDetector, X::Data; verbosity)::Fit
    num_features = size(X, 1)
    feature_values = to_feature_set(X)
    sample_values = to_sample_set(X)
    forest = IsolationForest.Forest(detector.num_trees, detector.sub_sampling_size, num_features, feature_values)
    score = make_scorer(forest, detector.normalize)
    return IForestModel(forest), score.(sample_values)
end

function OD.transform(detector::IForestDetector, model::IForestModel, X::Data)::Scores
    sample_values = to_sample_set(X)
    score = make_scorer(model.forest, detector.normalize)
    return score.(sample_values)
end
