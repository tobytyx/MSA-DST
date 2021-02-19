## MSA-DST Multi-Domain Self-Attention Dialogue State Tracking

## 简介
采用了以生成式模型为base解决多领域下的DST问题，模型为Seq2Seq结构，encoder采用Bert，decoder为标准transformer decoder结构。

在生成式的基础上，加入了判别任务，即Encoder的输出除了用在Decoder中，还会将\[CLS\]位置取出进行分类任务，确认有哪些槽位是需要做更新的。

## 项目结构
```console
├── create_data.py
├── data
│   └── multiwoz
│       ├── dev_dials.json
│       ├── gate_label.txt
│       ├── slot_map.json
│       ├── test_dials.json
│       ├── train_count.txt
│       └── train_dials.json
├── dataset.py
├── dependences
│   └── bert-base-uncased
│       ├── config.json
│       ├── pytorch_model.bin
│       └── vocab.txt
├── log.py
├── model.py
├── model_utils.py
├── output
├── README.md
├── special_tokens.txt
├── tokenizer.py
├── train.py
└── utils.py
```

## 运行方式
todo