# Training Result Summary

- Run timestamp: 20260627_173034
- Run mode: dev
- Device: cuda
- Baseline checkpoint: ./image-captioning-dev
- Exit checkpoint dir: ./checkpoint/intermediate_head_weights_dev
- Baseline best valid loss: 3.410998
- Exit best valid loss: 13.354266

## Baseline Test Metrics
- BLEU-1: 0.4795
- BLEU-2: 0.2870
- BLEU-3: 0.1416
- BLEU-4: 0.0806
- CIDEr: 0.1556
- METEOR: 0.1273

## Early Exit Test Metrics
- BLEU-1: 0.4795
- BLEU-2: 0.2870
- BLEU-3: 0.1416
- BLEU-4: 0.0806
- CIDEr: 0.1556
- METEOR: 0.1273

## Inference Timing
- Baseline avg latency: 1410.324 ms/image
- Early exit avg latency: 3951.275 ms/image
- Speedup: 0.357x
- Baseline avg ms/token: 44.073
- Early exit avg ms/token: 123.477
- Early exit avg layer: 12.000

## Output Files
- result.log
- baseline_step_log.csv
- baseline_epoch_log.csv
- baseline_test_metrics.csv
- baseline_test_predictions.csv
- exit_step_log.csv
- exit_epoch_log.csv
- exit_test_metrics.csv
- exit_test_predictions.csv
- exit_layer_usage.csv
- inference_timing.csv
- baseline_train_loss.png
- baseline_valid_loss.png
- exit_train_loss.png
- exit_valid_loss.png
