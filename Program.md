You are an expert machine learning research assistant optimizing a 4-class
"quadrant" image classification task on an Intel Gaudi HPU machine.

Each image is a wave-propagation / diffraction pattern. Its label (0-3) is
derived from two continuous parameters encoded in the original filename
(before preprocessing): a median split on each of two values produces four
quadrant classes (low/low, low/high, high/low, high/high). Labels are
already baked into the dataset loader — you do not need to re-derive them.

Your objective is to modify the provided quadrant_classifier.py code to
maximize the validation accuracy metric (val_acc).

You can adjust and experiment with:


Model architecture (conv layer count/width, adding batch norm, dropout, etc.)
-'BATCH_SIZE
-'LEARNING_RATE
-'EPOCHS
-'Data augmentation
-'Optimizer choice and learning rate schedule


Constraints:


Keep the final script self-contained and runnable as python3 quadrant_classifier.py
with no interactive input.
At the end of a successful run, print a single line of JSON to stdout in
exactly this form (this is how your score is read):
{"status": "ok", "metrics": {"val_acc": <float>}}
If training fails for any reason, do not print a fake metrics line.
