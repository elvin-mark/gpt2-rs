mod gpt;
mod nn;
mod tokenizer;

use anyhow::Result;
use clap::Parser;

use gpt::load_gpt2_model_from_binary;
use tokenizer::Encoder;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(short, long, default_value = "./tokenizer.json")]
    tokenizer: String,

    #[arg(short, long, default_value = "./model.bin")]
    model: String,

    #[arg(
        short,
        long,
        default_value = "In physics, string theory is a theoretical framework in which the point-like particles of particle physics are replaced by one-dimensional objects called strings."
    )]
    prompt: String,

    #[arg(short = 'k', long, default_value_t = 5)]
    topk: usize,

    #[arg(short = 'n', long, default_value_t = 20)]
    num_tokens: usize,
}

fn main() -> Result<()> {
    let args = Args::parse();

    println!("Tokenizer Path: {}", args.tokenizer);
    println!("Model Path: {}", args.model);

    let mut encoder = Encoder::new(&args.tokenizer)?;
    let mut model = load_gpt2_model_from_binary(&args.model)?;

    let n_head = 12; // Hardcoded for GPT-2 base
    let n_ctx = 1024; // Hardcoded for GPT-2 base

    let input_ids = encoder.encode(&args.prompt);
    if input_ids.len() + args.num_tokens > n_ctx {
        anyhow::bail!("Prompt is too long!");
    }

    let output_ids = model.generate(input_ids, n_head, args.num_tokens, args.topk);
    let output_text = encoder.decode(&output_ids);

    println!("\n--- Generated Text ---");
    println!("{}", output_text);

    Ok(())
}
