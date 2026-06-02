# LSAR: Sparse Lexical Representation Learning for Efficient and Interpretable Audio Retrieval

## Data

Use JSONL files with Mimi audio codes and text fields.

- ASR: `audio_codes`, `transcript`
- Caption: `audio_codes`, `caption`
- SQA: `context_audio_codes`, `question`, `context`

An optional `teacher_sparse` field may provide precomputed SPLADE vectors as `{"indices": [...], "values": [...]}`. Use `--teacher_field teacher_sparse` to read it; otherwise, `train.py` encodes text with SPLADE-v3.

## Training

Datasets are configured with repeated `--train_data` and `--val_data` arguments. Each argument is a comma-separated specification:

```text
task=<asr|caption|sqa>,path=<jsonl_path>[,name=<name>][,audio=<audio_field>][,text=<text_field>][,question=<question_field>][,context=<context_field>][,group=<group_field>][,teacher=<teacher_field>][,start=<ratio>][,end=<ratio>]
```

Default fields are `audio_codes` + `transcript` for ASR, `audio_codes` + `caption` for Caption, and `context_audio_codes` + `question` + `context` for SQA. `group` is optional and is used to mark multiple texts from the same audio as positives.

```bash
python train.py \
  --train_data task=asr,path=/path/asr_train.jsonl \
  --train_data task=caption,path=/path/caption_train.jsonl \
  --train_data task=sqa,path=/path/sqa_train.jsonl \
  --val_data task=asr,path=/path/asr_val.jsonl \
  --val_data task=caption,path=/path/caption_val.jsonl \
  --val_data task=sqa,path=/path/sqa_val.jsonl
```

## Requirements

```bash
pip install torch sentence-transformers tqdm
```

## License

The source code in this repository is released under the GNU General Public License v3.0. See the `LICENSE` file for details.

The accompanying paper is licensed separately under CC-BY-NC-ND, according to the publication license selected for the paper.
