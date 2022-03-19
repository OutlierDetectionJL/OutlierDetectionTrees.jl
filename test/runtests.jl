using OutlierDetectionTrees
using OutlierDetectionTest

test_meta.(eval.(OutlierDetectionTrees.MODELS))

data = TestData()
run_test(detector) = test_detector(detector, data)

run_test(IForestDetector())
