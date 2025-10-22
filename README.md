# cap-initial

## What this script expects (input schema)

It loads a single **Parquet** file:

```
/data/pgerard/tenet-media-cross-platform/ts-tk-tw/all_clustered_filtered.parquet
```

Your dataframe **must** contain these columns:

* `username` *(string)* — user identifier

  * (If your data uses `url-username`, set `user_col='url-username'` where applicable.)
* `cluster` *(int or string)* — cluster/sub-narrative id for each post
* `timestamp` *(datetime or parseable string)* — post time

The script derives:

* `time_bin` *(string)* — computed as 2-week bins:
  `df['time_bin'] = pd.to_datetime(timestamp).dt.to_period('2W').astype(str)`

> Minimal columns required: **`username`, `cluster`, `timestamp`**.
> If you only have `url-username`, either rename to `username` or pass `user_col='url-username'` when calling the temporal neighbor function.

---

## What it produces (outputs)

Saves NetworkX graphs (pickled) to:

```
save_dir = "/data/pgerard/tenet-media-cross-platform/ts-tk-tw/cap-networks-with-telegram/"
```

Files:

* `cap-full.gpickle` — full CAP graph (HNSW kNN over TF-IDF user-cluster vectors)
* `t-cap-full-decay_2w.gpickle` — temporal CAP graph aggregated over 2-week bins with exponential decay


## Configuration knobs (in-script defaults)

* **Paths**

  * `file_path` — input parquet
  * `save_dir` — where graphs are written

* **Temporal binning**

  * `2W` (two weeks). Change to `1W`, `M`, etc. if needed.


* **Temporal aggregation**

  * `method='decay'`, `lambda_=0.2` (higher λ → more weight on recent bins)

---

## Environment & dependencies

Python ≥ 3.9 recommended.

```bash
pip install pandas numpy networkx tqdm scipy scikit-learn faiss-cpu
```


---

## Quick start

1. Ensure your parquet has **`username`, `cluster`, `timestamp`**.
2. Update `file_path` and `save_dir` near the bottom of the script.
3. Run the script (e.g., `python cap_builder.py`).
4. Check `save_dir` for:

   * `cap-full.gpickle`
   * `t-cap-full-decay_2w.gpickle`
