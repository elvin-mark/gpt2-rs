use byteorder::{LittleEndian, ReadBytesExt};
use ndarray::{Array1, Array2};
use rand::rng;
use rand::seq::IndexedRandom;
use std::fs::File;
use std::io::{self, BufReader};

use crate::nn::{Attention, MLP, ffn, get_embedding, layer_norm, mha};

pub struct Block {
    pub mlp: MLP,
    pub attn: Attention,
    pub ln1_g: Array1<f32>,
    pub ln1_b: Array1<f32>,
    pub ln2_g: Array1<f32>,
    pub ln2_b: Array1<f32>,
}

pub struct GPT2Model {
    pub wte: Array2<f32>,
    pub wpe: Array2<f32>,
    pub blocks: Vec<Block>,
    pub ln_f_g: Array1<f32>,
    pub ln_f_b: Array1<f32>,
    pub lm_head: Option<Array2<f32>>, // Use Option for weight tying
}

fn transformer_block(x: &Array2<f32>, block: &Block, n_head: usize) -> Array2<f32> {
    let ln1_out = layer_norm(x, &block.ln1_g, &block.ln1_b, 1e-5);
    let mha_out = mha(&ln1_out, &block.attn, n_head);

    let x1 = x + &mha_out;

    let ln2_out = layer_norm(&x1, &block.ln2_g, &block.ln2_b, 1e-5);
    let ffn_out = ffn(&ln2_out, &block.mlp);

    &x1 + ffn_out
}

impl GPT2Model {
    pub fn forward(&self, inputs: &[u32], n_head: usize) -> Array2<f32> {
        let seq_len = inputs.len();
        if seq_len == 0 {
            return Array2::zeros((0, 0));
        }

        let tok_emb = get_embedding(&self.wte, inputs);

        let pos_indices: Vec<u32> = (0..seq_len as u32).collect();
        let pos_emb = get_embedding(&self.wpe, &pos_indices);

        let mut x = tok_emb + pos_emb;

        for block in &self.blocks {
            x = transformer_block(&x, block, n_head);
        }

        let ln_out = layer_norm(&x, &self.ln_f_g, &self.ln_f_b, 1e-5);

        if let Some(lm_head_w) = &self.lm_head {
            ln_out.dot(lm_head_w)
        } else {
            ln_out.dot(&self.wte.t())
        }
    }

    pub fn generate(
        &mut self,
        mut inputs: Vec<u32>,
        n_head: usize,
        n_tokens_to_generate: usize,
        top_k: usize,
    ) -> Vec<u32> {
        for i in 0..n_tokens_to_generate {
            println!("Generating token {}/{}", i + 1, n_tokens_to_generate);
            let logits = self.forward(&inputs, n_head);

            let last_logits = logits.row(logits.nrows() - 1);

            let mut sorted_logits: Vec<(u32, f32)> = last_logits
                .iter()
                .enumerate()
                .map(|(id, &val)| (id as u32, val))
                .collect();
            sorted_logits.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

            let top_k_logits = if top_k < sorted_logits.len() {
                &sorted_logits[0..top_k]
            } else {
                &sorted_logits
            };

            let mut rng = rng();
            let &(next_id, _) = top_k_logits.choose(&mut rng).unwrap();

            inputs.push(next_id);
        }
        let inputs_len = inputs.len();
        inputs
            .into_iter()
            .skip(inputs_len - n_tokens_to_generate)
            .collect()
    }
}

pub fn load_gpt2_model_from_binary(file_path: &str) -> anyhow::Result<GPT2Model> {
    const HIDDEN_DIM: usize = 768;
    const N_CTX: usize = 1024;
    const N_VOCAB: usize = 50257;
    const N_HEAD: usize = 12;
    const FC_DIM: usize = 3072;

    let file = File::open(file_path)?;
    let mut reader = BufReader::with_capacity(1024 * 1024, file); // 1MB buffer

    // Helper for bulk reading
    let mut read_floats = |count: usize| -> io::Result<Vec<f32>> {
        let mut data = vec![0.0f32; count];
        // Much faster: reads the whole block in one go
        reader.read_f32_into::<LittleEndian>(&mut data)?;
        Ok(data)
    };

    println!("Loading WTE...");
    let wte_data = read_floats(N_VOCAB * HIDDEN_DIM)?;
    let wte = Array2::from_shape_vec((N_VOCAB, HIDDEN_DIM), wte_data).unwrap();

    println!("Loading WPE...");
    let wpe_data = read_floats(N_CTX * HIDDEN_DIM)?;
    let wpe = Array2::from_shape_vec((N_CTX, HIDDEN_DIM), wpe_data).unwrap();

    let ln_f_g_data = read_floats(HIDDEN_DIM)?;
    let ln_f_g = Array1::from_vec(ln_f_g_data);

    let ln_f_b_data = read_floats(HIDDEN_DIM)?;
    let ln_f_b = Array1::from_vec(ln_f_b_data);

    let mut blocks = Vec::with_capacity(N_HEAD);
    for _ in 0..N_HEAD {
        let ln1_g = Array1::from_vec(read_floats(HIDDEN_DIM)?);
        let ln1_b = Array1::from_vec(read_floats(HIDDEN_DIM)?);
        let ln2_g = Array1::from_vec(read_floats(HIDDEN_DIM)?);
        let ln2_b = Array1::from_vec(read_floats(HIDDEN_DIM)?);

        let c_attn_w = Array2::from_shape_vec(
            (HIDDEN_DIM, HIDDEN_DIM * 3),
            read_floats(HIDDEN_DIM * HIDDEN_DIM * 3)?,
        )
        .unwrap();
        let c_attn_b = Array1::from_vec(read_floats(HIDDEN_DIM * 3)?);

        let c_proj_w_attn = Array2::from_shape_vec(
            (HIDDEN_DIM, HIDDEN_DIM),
            read_floats(HIDDEN_DIM * HIDDEN_DIM)?,
        )
        .unwrap();
        let c_proj_b_attn = Array1::from_vec(read_floats(HIDDEN_DIM)?);

        let c_fc_w =
            Array2::from_shape_vec((HIDDEN_DIM, FC_DIM), read_floats(HIDDEN_DIM * FC_DIM)?)
                .unwrap();
        let c_fc_b = Array1::from_vec(read_floats(FC_DIM)?);

        let c_proj_w_mlp =
            Array2::from_shape_vec((FC_DIM, HIDDEN_DIM), read_floats(FC_DIM * HIDDEN_DIM)?)
                .unwrap();
        let c_proj_b_mlp = Array1::from_vec(read_floats(HIDDEN_DIM)?);

        blocks.push(Block {
            mlp: MLP {
                c_fc_w,
                c_fc_b,
                c_proj_w: c_proj_w_mlp,
                c_proj_b: c_proj_b_mlp,
            },
            attn: Attention {
                c_attn_w,
                c_attn_b,
                c_proj_w: c_proj_w_attn,
                c_proj_b: c_proj_b_attn,
            },
            ln1_g,
            ln1_b,
            ln2_g,
            ln2_b,
        });
    }

    Ok(GPT2Model {
        wte,
        wpe,
        blocks,
        ln_f_g,
        ln_f_b,
        lm_head: None,
    })
}
