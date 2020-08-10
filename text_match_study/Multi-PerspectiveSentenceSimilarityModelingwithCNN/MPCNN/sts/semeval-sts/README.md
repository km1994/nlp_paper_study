SemEval STS Task
================

Primarily based on the references in

http://alt.qcri.org/semeval2016/task1/index.php?id=data-and-tools

Per-year directories contain the datasets for these years, while
the ``all/`` contains a lot of symlinks.

Let's say you want to compare your model to 2016 entrants. To get
all the datasets before 2016 for your training set, load the
``all/201[0-5]*`` glob.

Standard train/val/test splits were created like this:

	cat data/sts/semeval-sts/all/201[-4].[^t]* >data/sts/semeval-sts/all/2015.train.tsv
	cat data/sts/semeval-sts/all/2014.tweet-news.test.tsv >data/sts/semeval-sts/all/2015.val.tsv
	cat data/sts/semeval-sts/all/2015.* >data/sts/semeval-sts/all/2015.test.tsv

Otherwise: Use scipy.stats.pearson.  The evaluation code is also
the same as in ../sick2014 - refer e.g. to the python example from
skip-thoughts, or in our own examples/ directory.

Model Comparison
----------------

For randomized models, 95% confidence intervals (t-distribution) are reported.

Because SemEval 2016 competition results weren't published at the test time,
we train on -2014 and test on 2015.  We use 2014.tweet-news as a validation
set.

Also NOTE THAT THESE RESULTS ARE OBSOLETE because they predate the f/bigvocab port.

| Model                    | train    | val      | ans.for. | ans.stud | belief   | headline | images   | t. mean  | settings
|--------------------------|----------|----------|----------|----------|----------|----------|----------|----------|---------
| termfreq TF-IDF #w       | 0.497085 | 0.651653 | 0.607226 | 0.676746 | 0.622920 | 0.725578 | 0.714331 | 0.669360 | ``freq_mode='tf'``
| termfreq BM25 #w         | 0.503736 | 0.656081 | 0.626950 | 0.690302 | 0.632223 | 0.725748 | 0.718185 | 0.678681 | (defaults)
| DLS@CU-S1                |          |          | 0.7390   | 0.7725   | 0.7491   | 0.8250   | 0.8644   | 0.8015   | STS2015 winner
|--------------------------|----------|----------|----------|----------|----------|----------|----------|----------|---------
| avg                      | 0.701518 | 0.634915 | 0.403326 | 0.654551 | 0.512077 | 0.670571 | 0.676907 | 0.583487 | (defaults)
|                          |±0.035813 |±0.010222 |±0.027295 |±0.009161 |±0.039178 |±0.011041 |±0.013377 |±0.134786 |
| DAN                      | 0.686774 | 0.672085 | 0.476526 | 0.687200 | 0.534313 | 0.697941 | 0.707563 | 0.620708 | ``inp_e_dropout=0`` ``inp_w_dropout=1/3`` ``deep=2`` ``pact='relu'``
|                          |±0.017556 |±0.006131 |±0.020705 |±0.006518 |±0.035623 |±0.006570 |±0.008592 |±0.119332 |
|--------------------------|----------|----------|----------|----------|----------|----------|----------|----------|---------
| rnn                      | 0.663181 | 0.613284 | 0.384119 | 0.608614 | 0.575296 | 0.606754 | 0.623750 | 0.559706 | (defaults)
|                          |±0.058655 |±0.057435 |±0.049856 |±0.040745 |±0.047644 |±0.067737 |±0.044167 |±0.110753 |
| cnn                      | 0.821512 | 0.696150 | 0.495722 | 0.658635 | 0.667512 | 0.689457 | 0.727084 | 0.647682 | ``inp_e_dropout=1/2`` ``dropout=1/2`` (FIXME)
|                          |±0.016233 |±0.002759 |±0.006732 |±0.006713 |±0.006377 |±0.006797 |±0.004054 |±0.098796 |
| rnncnn                   | 0.819834 | 0.705950 | 0.523081 | 0.699923 | 0.676170 | 0.717214 | 0.734250 | 0.670127 | (defaults)
|                          |±0.032854 |±0.005099 |±0.007836 |±0.005869 |±0.010037 |±0.007489 |±0.005823 |±0.094360 |
| attn1511                 | 0.712086 | 0.656483 | 0.429167 | 0.632170 | 0.628803 | 0.657264 | 0.668384 | 0.603158 | (defaults)
|                          |±0.033190 |±0.009479 |±0.019904 |±0.016477 |±0.015415 |±0.012070 |±0.023045 |±0.109596 |

These results are obtained like this:

	tools/train.py avg sts data/sts/semeval-sts/all/2015.train.tsv data/sts/semeval-sts/all/2015.val.tsv nb_runs=16
	tools/eval.py avg sts data/sts/semeval-sts/all/2015.train.tsv data/sts/semeval-sts/all/2015.val.tsv data/sts/semeval-sts/all/2015.test.tsv weights-sts-avg--69489c8dc3b6ce11-*-bestval.h5

(Note that the current toolchain does not include explicit support for
evaluation of the individual test splits.  FIXME)


Changes
-------

The original distribution puts gold standard and sentence pairs
in separate files.  To make ingestion easier, we paste them together
in .tsv files.

Not Included
------------

These datasets were not included:

  * STS2012-MSRvid (licence restriction)
  * Sample outputs
  * Raw STS2015 (at least for now; TODO?)
  * STS2015 sample baseline and per-forum stackoverflow data (aggregate
    data is included!)
  * correlation.pl scripts (incompatible with our tsv files)
