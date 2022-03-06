## Evalutation_script

You can evaluate the result files of the models with code.

We draw all the results of state-of-the-art methods in a single figure to make it easy to compare, and the figure represents the miss-rate against false positives per image.

For annotations file, only json is supported, and for result files, json and txt formats are supported.
(multiple `--rstFiles` are supported)

Example)

```bash
$ python evaluation_script.py \
	--annFile KAIST_annotation.json \
	--rstFile state_of_arts/MLPD_result.txt \
			  state_of_arts/ARCNN_result.txt \
			  state_of_arts/CIAN_result.txt \
			  state_of_arts/MSDS-RCNN_result.txt \
			  state_of_arts/MBNet_result.txt \
	--evalFig KASIT_BENCHMARK.jpg
```
![result img](../Doc/figure/figure.jpg)
