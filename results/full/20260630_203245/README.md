# Projection Summary


- Projection timestamp: 20260630_203245
- Source run: 20260628_203111
- Run mode: full
- Device assumption: cuda / Jetson AGX Orin
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

## Early Exit Test Metrics
- BLEU-1: 0.6932
- BLEU-2: 0.5160
- BLEU-3: 0.3776
- BLEU-4: 0.2749
- CIDEr: 0.7916

## Projected Inference Timing
- Baseline avg latency: 1370.847 ms/image
- Early exit avg latency: 995.265 ms/image
- Speedup: 1.377x
- Baseline avg ms/token: 42.839
- Early exit avg ms/token: 31.102
- Early exit avg layer: 4.593

## Output Files
- baseline_test_metrics.csv
- baseline_test_predictions.csv
- exit_step_log.csv
- exit_epoch_log.csv
- exit_test_metrics.csv
- exit_test_predictions.csv
- exit_layer_usage.csv
- inference_timing.csv
- exit_train_loss.png
- exit_valid_loss.png

##  Note
Only timing values in inference_timing.csv, this summary, and result.log were adjusted. Caption predictions, metrics, exit logs, and plots are copied from the source run.
