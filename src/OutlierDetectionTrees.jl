module OutlierDetectionTrees
    using OutlierDetectionInterface
    using OutlierDetectionInterface:SCORE_UNSUPERVISED
    const OD = OutlierDetectionInterface

    include("models/IForest.jl")

    const UUID = "6470b2ab-4fe8-498e-808d-6badd5c3da38"
    const MODELS = [:IForestDetector]

    for model in MODELS
        @eval begin
            OD.@default_frontend $model
            OD.@default_metadata $model $UUID
            export $model
        end
    end
end
