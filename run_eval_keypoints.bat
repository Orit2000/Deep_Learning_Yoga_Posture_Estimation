@echo off
python evaluate_test_set.py ^
  --model_type keypoints ^
  --model_path Results/best_model_weights_kp.pth ^
  --history_csv Results/training_history_kp.csv ^
  --output_npz Results/keypoints_preds_test.npz
pause
