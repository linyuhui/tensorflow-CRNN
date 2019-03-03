CRNN 

### Env

tensorflow 1.8

### Prepare data

You can use [TextRecognitionDataGenerator](https://github.com/Belval/TextRecognitionDataGenerator) to 
generate text image.

Before building your vocab, you should specify charset in vocab.py, then,

```python vocab.py \
--operation build --output_dir datasets --name vocab
```

Convert images to tf record file, you can find the script in my another Repo.

Specify DEFAULT_CONFIG in dataset file (eg. datasets/text.py).

### Train

```
cd python
bash run_train.sh
```

watch curve of loss:
```
tensorboard --logdir=your_train_logdir --port=your_port
```

### Eval

Run eval repeatedly, it will check your ckpt-dir automatically

```
bash run_eval_loop.sh
```

watch curve of acc:

```tensorboard --logdir=eval_dir --port=your_port```

or eval once

```bash run_eval_once.sh```

### Test

```run_test.sh```

### Paper

[An End-to-End Trainable Neural Network for Image-based Sequence
Recognition and Its Application to Scene Text Recognition](https://arxiv.org/pdf/1507.05717.pdf)