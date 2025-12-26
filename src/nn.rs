use ndarray::{Array1, Array2, Axis, s};

pub struct MLP {
    pub c_fc_w: Array2<f32>,
    pub c_fc_b: Array1<f32>,
    pub c_proj_w: Array2<f32>,
    pub c_proj_b: Array1<f32>,
}

pub struct Attention {
    pub c_attn_w: Array2<f32>,
    pub c_attn_b: Array1<f32>,
    pub c_proj_w: Array2<f32>,
    pub c_proj_b: Array1<f32>,
}

pub fn gelu(x: &Array2<f32>) -> Array2<f32> {
    const SQRT_2_OVER_PI: f32 = 0.79788456;

    // Clone once to get owned memory
    let mut output = x.clone();

    // Perform all math in a single pass over the data
    output.map_inplace(|val| {
        let v = *val;
        // Using v*v*v is even faster than powf(3)
        let inner = SQRT_2_OVER_PI * (v + 0.044715 * v * v * v);
        *val = 0.5 * v * (1.0 + inner.tanh());
    });

    output
}

pub fn softmax(x: &Array2<f32>) -> Array2<f32> {
    let mut result = Array2::zeros(x.dim());
    for (i, row) in x.axis_iter(Axis(0)).enumerate() {
        let max_val = row.fold(f32::NEG_INFINITY, |acc, &val| val.max(acc));
        let exp_row = (&row - max_val).mapv(f32::exp);
        let sum_exp = exp_row.sum();
        result.row_mut(i).assign(&(&exp_row / sum_exp));
    }
    result
}

pub fn layer_norm(x: &Array2<f32>, g: &Array1<f32>, b: &Array1<f32>, eps: f32) -> Array2<f32> {
    let mut out = Array2::zeros(x.dim());
    for (i, row) in x.axis_iter(Axis(0)).enumerate() {
        let mean = row.sum() / row.len() as f32;
        let variance =
            row.iter().map(|&val| (val - mean).powf(2.0)).sum::<f32>() / row.len() as f32;
        let std = (variance + eps).sqrt();

        let norm_row = (&row - mean) / std;
        out.row_mut(i).assign(&(&norm_row * g + b));
    }
    out
}

pub fn linear(x: &Array2<f32>, w: &Array2<f32>, b: &Array1<f32>) -> Array2<f32> {
    let mut z = x.dot(w);
    z += b;
    z
}

pub fn ffn(x: &Array2<f32>, mlp: &MLP) -> Array2<f32> {
    let x1 = linear(x, &mlp.c_fc_w, &mlp.c_fc_b);
    let x2 = gelu(&x1);
    let x3 = linear(&x2, &mlp.c_proj_w, &mlp.c_proj_b);
    x3
}

pub fn attention(
    q: &Array2<f32>,
    k: &Array2<f32>,
    v: &Array2<f32>,
    mask: &Array2<f32>,
) -> Array2<f32> {
    let mut score = q.dot(&k.t());
    let scale = (k.ncols() as f32).sqrt();
    score /= scale;

    score += mask;

    let smax = softmax(&score);

    smax.dot(v)
}

pub fn mha(x: &Array2<f32>, attn: &Attention, n_head: usize) -> Array2<f32> {
    let rows = x.nrows();

    let qkv = linear(x, &attn.c_attn_w, &attn.c_attn_b);

    let (q, k, v) = split_qkv(&qkv);

    let q_heads = split_into_heads(&q, n_head);
    let k_heads = split_into_heads(&k, n_head);
    let v_heads = split_into_heads(&v, n_head);

    let mask = create_causal_mask(rows);

    let head_outputs: Vec<Array2<f32>> = (0..n_head)
        .map(|h| attention(&q_heads[h], &k_heads[h], &v_heads[h], &mask))
        .collect();

    let concat_output = concatenate_heads(&head_outputs);

    linear(&concat_output, &attn.c_proj_w, &attn.c_proj_b)
}

pub fn split_qkv(qkv: &Array2<f32>) -> (Array2<f32>, Array2<f32>, Array2<f32>) {
    let (_, cols) = qkv.dim();
    let d_model = cols / 3;

    let q = qkv.slice(s![.., 0..d_model]).to_owned();
    let k = qkv.slice(s![.., d_model..2 * d_model]).to_owned();
    let v = qkv.slice(s![.., 2 * d_model..3 * d_model]).to_owned();

    (q, k, v)
}

pub fn split_into_heads(matrix: &Array2<f32>, n_head: usize) -> Vec<Array2<f32>> {
    let (_, cols) = matrix.dim();
    let head_dim = cols / n_head;

    let mut heads = Vec::with_capacity(n_head);
    for h in 0..n_head {
        let start_col = h * head_dim;
        let end_col = start_col + head_dim;
        heads.push(matrix.slice(s![.., start_col..end_col]).to_owned());
    }
    heads
}

pub fn concatenate_heads(heads: &[Array2<f32>]) -> Array2<f32> {
    if heads.is_empty() {
        return Array2::zeros((0, 0));
    }

    let (rows, head_dim) = heads[0].dim();
    let n_head = heads.len();
    let total_dim = n_head * head_dim;

    let mut result = Array2::zeros((rows, total_dim));

    for (h, head) in heads.iter().enumerate() {
        let start_col = h * head_dim;
        let end_col = start_col + head_dim;
        result.slice_mut(s![.., start_col..end_col]).assign(head);
    }
    result
}

pub fn get_embedding(emb_matrix: &Array2<f32>, indices: &[u32]) -> Array2<f32> {
    let emb_dim = emb_matrix.ncols();
    let seq_len = indices.len();

    let mut result = Array2::zeros((seq_len, emb_dim));

    for (i, &idx) in indices.iter().enumerate() {
        result.row_mut(i).assign(&emb_matrix.row(idx as usize));
    }
    result
}

pub fn create_causal_mask(seq_len: usize) -> Array2<f32> {
    let mut mask = Array2::zeros((seq_len, seq_len));

    for i in 0..seq_len {
        for j in 0..seq_len {
            if j > i {
                mask[[i, j]] = f32::NEG_INFINITY;
            }
        }
    }
    mask
}
