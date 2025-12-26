use regex::Regex;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::fs;

// TokenizerData represents the structure of tokenizer.json
#[derive(Debug, Serialize, Deserialize)]
pub struct TokenizerData {
    pub model: TokenizerModel,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct TokenizerModel {
    pub vocab: HashMap<String, u32>,
    pub merges: Vec<String>,
}

pub struct Encoder {
    pub encoder: HashMap<String, u32>,
    pub decoder: HashMap<u32, String>,
    pub byte_encoder: HashMap<u8, char>,
    pub byte_decoder: HashMap<char, u8>,
    pub bpe_ranks: HashMap<(String, String), u32>,
    pub cache: HashMap<String, String>,
    pub pat: Regex,
}

fn bytes_to_unicode() -> (HashMap<u8, char>, HashMap<char, u8>) {
    let mut bs: Vec<u8> = (b'!'..=b'~').collect();
    bs.extend(0xA1..=0xAC);
    bs.extend(0xAE..=0xFF);

    let mut cs: Vec<char> = bs.iter().map(|&b| b as char).collect();
    let mut n = 0;
    for b in 0..=255u8 {
        if !bs.contains(&b) {
            bs.push(b);
            cs.push(std::char::from_u32(0x100 + n).unwrap());
            n += 1;
        }
    }

    let byte_encoder: HashMap<u8, char> = bs.into_iter().zip(cs.into_iter()).collect();
    let byte_decoder: HashMap<char, u8> = byte_encoder.iter().map(|(&k, &v)| (v, k)).collect();

    (byte_encoder, byte_decoder)
}

fn get_pairs(word: &[String]) -> HashSet<(String, String)> {
    let mut pairs = HashSet::new();
    if word.len() < 2 {
        return pairs;
    }

    for i in 0..word.len() - 1 {
        pairs.insert((word[i].clone(), word[i + 1].clone()));
    }
    pairs
}

impl Encoder {
    pub fn new(tokenizer_path: &str) -> anyhow::Result<Self> {
        let file = fs::read_to_string(tokenizer_path)?;
        let tokenizer_data: TokenizerData = serde_json::from_str(&file)?;

        let encoder_map = tokenizer_data.model.vocab;
        let bpe_merges = tokenizer_data.model.merges;

        let mut bpe_ranks = HashMap::new();
        for (i, merge_str) in bpe_merges.into_iter().enumerate() {
            let parts: Vec<&str> = merge_str.split_whitespace().collect();
            if parts.len() == 2 {
                bpe_ranks.insert((parts[0].to_string(), parts[1].to_string()), i as u32);
            }
        }

        let decoder_map: HashMap<u32, String> =
            encoder_map.iter().map(|(k, &v)| (v, k.clone())).collect();
        let (byte_encoder, byte_decoder) = bytes_to_unicode();

        // let pat = Regex::new(
        //     r#"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"#,
        // )?;
        // Removed the (?!\S) look-ahead to make it compatible with the standard regex crate
        let pat = Regex::new(r#"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+"#)
            .unwrap();

        Ok(Encoder {
            encoder: encoder_map,
            decoder: decoder_map,
            byte_encoder,
            byte_decoder,
            bpe_ranks,
            cache: HashMap::new(),
            pat,
        })
    }

    pub fn bpe(&mut self, token: &str) -> String {
        if let Some(cached_val) = self.cache.get(token) {
            return cached_val.clone();
        }

        let mut word: Vec<String> = token.chars().map(|c| c.to_string()).collect();

        if word.len() < 2 {
            self.cache.insert(token.to_string(), token.to_string());
            return token.to_string();
        }

        let mut pairs = get_pairs(&word);

        while !pairs.is_empty() {
            let mut min_rank = u32::MAX;
            let mut bigram = None;
            let mut found = false;

            for pair in &pairs {
                if let Some(&rank) = self.bpe_ranks.get(pair) {
                    if rank < min_rank {
                        min_rank = rank;
                        bigram = Some(pair.clone());
                        found = true;
                    }
                }
            }

            if !found {
                break;
            }

            let (first, second) = bigram.unwrap();
            let mut new_word: Vec<String> = Vec::new();
            let mut i = 0;

            while i < word.len() {
                if let Some(j) = word[i..].iter().position(|s| s == &first) {
                    let j = i + j;
                    new_word.extend_from_slice(&word[i..j]);
                    i = j;
                } else {
                    new_word.extend_from_slice(&word[i..]);
                    break;
                }

                if i < word.len() - 1 && word[i] == first && word[i + 1] == second {
                    new_word.push(format!("{}{}", first, second));
                    i += 2;
                } else {
                    new_word.push(word[i].clone());
                    i += 1;
                }
            }
            word = new_word;
            if word.len() == 1 {
                break;
            }
            pairs = get_pairs(&word);
        }

        let result = word.join(" ");
        self.cache.insert(token.to_string(), result.clone());
        result
    }

    pub fn encode(&mut self, text: &str) -> Vec<u32> {
        if text.is_empty() {
            return Vec::new();
        }

        let mut bpe_tokens = Vec::new();
        let matches: Vec<_> = self
            .pat
            .find_iter(text)
            .map(|m| m.as_str().to_string())
            .collect();
        for token in matches {
            let encoded_token: String = token.bytes().map(|b| self.byte_encoder[&b]).collect();

            let bpe_result = self.bpe(&encoded_token);

            if !bpe_result.is_empty() {
                for bpe_token_str in bpe_result.split_whitespace() {
                    if let Some(&token_id) = self.encoder.get(bpe_token_str) {
                        bpe_tokens.push(token_id);
                    } else {
                        eprintln!("Warning: unknown token '{}'", bpe_token_str);
                    }
                }
            }
        }
        bpe_tokens
    }

    pub fn decode(&self, tokens: &[u32]) -> String {
        if tokens.is_empty() {
            return String::new();
        }

        let mut text = String::new();
        for &token_id in tokens {
            if let Some(token_str) = self.decoder.get(&token_id) {
                text.push_str(token_str);
            } else {
                eprintln!("Warning: unknown token ID {}", token_id);
            }
        }

        let decoded_bytes: Vec<u8> = text
            .chars()
            .filter_map(|r| self.byte_decoder.get(&r).copied())
            .collect();

        String::from_utf8_lossy(&decoded_bytes).to_string()
    }
}
