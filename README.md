# GPT2-rs

## About
Simple implementation (not optimized) of GPT2 inference in Rust. This is based on my previous implementation on Go lang.

## Model convertion
You will need to download the original weights of GPT2 from huggingface (you will also need the the `tokenizer.json`). And then run the following command.

```sh
python3 python/gpt2_model_converter.py /path/to/original_model_weights.pth /path/to/model.bin
```

## Build Me
```sh
cargo build --release
```

## Run Me
```sh
./target/release/gpt2-rs --tokenize /path/to/tokenizer.json --model /path/to/model.bin
```