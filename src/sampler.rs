//! Token sampling strategies. Includes a small xorshift64 PRNG so the
//! crate doesn't need the `rand` dependency.

pub struct Rng {
    state: u64,
}

impl Rng {
    pub fn new(seed: u64) -> Self {
        Rng { state: if seed == 0 { 0xDEADBEEF } else { seed } }
    }

    /// xorshift64* — small, fast, good enough for sampling (not crypto).
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x.wrapping_mul(0x2545F4914F6CDD1D)
    }

    /// Uniform float in [0, 1)
    pub fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }
}

pub fn argmax(logits: &[f32]) -> usize {
    let mut best_i = 0;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v > best_v {
            best_v = v;
            best_i = i;
        }
    }
    best_i
}

/// Temperature scaling + top-k restriction + multinomial sampling.
/// temperature <= 0.0 falls back to greedy argmax.
pub fn sample_top_k(logits: &[f32], k: usize, temperature: f32, rng: &mut Rng) -> usize {
    if temperature <= 0.0 {
        return argmax(logits);
    }

    let mut indexed: Vec<(usize, f32)> = logits
        .iter()
        .enumerate()
        .map(|(i, &v)| (i, v / temperature))
        .collect();

    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    let k = k.min(indexed.len()).max(1);
    let top = &indexed[..k];

    // softmax over the top-k logits
    let max_logit = top.iter().map(|&(_, v)| v).fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = top.iter().map(|&(_, v)| (v - max_logit).exp()).collect();
    let sum: f32 = exps.iter().sum();

    let r = rng.next_f32() * sum;
    let mut acc = 0.0f32;
    for (i, &e) in exps.iter().enumerate() {
        acc += e;
        if acc >= r {
            return top[i].0;
        }
    }
    top[top.len() - 1].0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn argmax_finds_peak() {
        let logits = vec![0.1, 5.0, -2.0, 3.0];
        assert_eq!(argmax(&logits), 1);
    }

    #[test]
    fn zero_temperature_is_greedy() {
        let logits = vec![0.1, 5.0, -2.0, 3.0];
        let mut rng = Rng::new(42);
        assert_eq!(sample_top_k(&logits, 4, 0.0, &mut rng), 1);
    }

    #[test]
    fn top_k_only_returns_top_indices() {
        let logits = vec![0.0, 10.0, 9.0, -5.0, -5.0];
        let mut rng = Rng::new(1);
        for _ in 0..50 {
            let choice = sample_top_k(&logits, 2, 1.0, &mut rng);
            assert!(choice == 1 || choice == 2, "got {choice}, expected top-2 only");
        }
    }

    #[test]
    fn rng_is_deterministic_for_seed() {
        let mut a = Rng::new(7);
        let mut b = Rng::new(7);
        for _ in 0..10 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
    }
}
