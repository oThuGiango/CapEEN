# Training Result Summary

- Run timestamp: 20260628_203111
- Run mode: full
- Device: cuda
- Baseline checkpoint: ./image-captioning-full
- Exit checkpoint dir: ./checkpoint/intermediate_head_weights_full
- Baseline best valid loss: nan
- Exit best valid loss: 2.458338

## Baseline Test Metrics
- BLEU-1: 0.6968
- BLEU-2: 0.5176
- BLEU-3: 0.3794
- BLEU-4: 0.2793
- CIDEr: 0.7958
- METEOR: nan

## Early Exit Test Metrics
- BLEU-1: 0.6932
- BLEU-2: 0.5160
- BLEU-3: 0.3776
- BLEU-4: 0.2749
- CIDEr: 0.7916
- METEOR: nan

## Inference Timing
- Baseline avg latency: 1370.847 ms/image
- Early exit avg latency: 1739.908 ms/image
- Speedup: 0.788x
- Baseline avg ms/token: 42.839
- Early exit avg ms/token: 54.372
- Early exit avg layer: 4.593

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
