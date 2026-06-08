# VertFound baseline legacy IDR export

This export exactly reproduces the original VertFound `calc_IDR()` logic: group predictions by bbox, select the highest-score class for each bbox, and accept it when the signed mean coordinate difference is below 2.

The old metric is preserved for auditing and historical reproduction only. It is not a strict bbox-distance or IoU metric because negative coordinate differences can pass the threshold.
