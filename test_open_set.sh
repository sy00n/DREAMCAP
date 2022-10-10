python tools/get_threshold.py test_config_open.json

echo "Get Uncertainty Threshold Finished!"

python tools/compare_openness.py test_config_open.json

echo "Open Set Evaluation and Comparison Finished!"