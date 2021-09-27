use noisy_float::prelude::*;
use rand::distributions::WeightedIndex;
use rand::prelude::*;
use std::collections::HashMap;
const EPSILON: f64 = 1e-8;

// Cutoff is multiplier of baseline to stop at - e.g. 10%
// Might also want a min num steps
// Output is normalized to state 0, namely log n
fn coupled_entropy(transition: &[Vec<(usize, f64)>], cutoff: f64) -> Vec<(f64, Vec<Vec<f64>>)> {
    let size = transition.len();
    for row in transition {
        row.iter().for_each(|&(i, f)| {
            assert!(i < size);
            assert!(f > 0.0);
            assert!(f <= 1.0);
        });
        let row_sum: f64 = row.iter().map(|(_, f)| f).sum();
        assert!((row_sum - 1.0).abs() < EPSILON);
    }
    // dist[r][c] = p(original = r, coupled = c)
    let mut dist = vec![vec![0.0; size]; size];
    // Starts in state 0
    for i in 0..size {
        dist[0][i] = 1.0 / size as f64;
    }
    let mut entropies = vec![];
    loop {
        let entropy: f64 = dist
            .iter()
            .flat_map(|row| row.iter())
            .map(|&f| if f > 0.0 { -f * f.ln() } else { 0.0 })
            .sum();
        let norm_entropy = entropy / (size as f64).ln();
        entropies.push((norm_entropy, dist));
        if entropies.len() > size && norm_entropy - 1.0 < cutoff {
            return entropies;
        }
        let mut new_dist = vec![vec![0.0; size]; size];
        for (r, row) in entropies[entropies.len() - 1].1.iter().enumerate() {
            for (c, dist_p) in row.iter().enumerate() {
                if r == c {
                    for &(i, trans_p) in &transition[r] {
                        new_dist[i][i] += trans_p * dist_p;
                    }
                } else {
                    for &(i, trans_p_r) in &transition[r] {
                        for &(j, trans_p_c) in &transition[c] {
                            new_dist[i][j] += dist_p * trans_p_r * trans_p_c;
                        }
                    }
                }
            }
        }
        dist = new_dist;
    }
}
fn sample<T: Clone, R: Rng>(dist: &[f64], samples: u64, states: &[T], rng: &mut R) -> Vec<T> {
    let weighted = WeightedIndex::new(dist).unwrap();
    (0..samples)
        .map(|_| states[weighted.sample(rng)].clone())
        .collect()
}

fn cycle(n: usize) -> Vec<Vec<(usize, f64)>> {
    (0..n)
        .map(|i| vec![((i + n - 1) % n, 1.0 / 2.0), ((i + 1) % n, 1.0 / 2.0)])
        .collect()
}
fn cycles() {
    let many = (false, 20);
    let verbose = !many.0;
    let lower = if many.0 { 1 } else { many.1 };
    let upper = many.1;
    for k_base in lower..=upper {
        let k = k_base * 2 + 1;
        let transition = cycle(k);
        let result = coupled_entropy(&transition, 0.1);
        if verbose {
            println!("Cycle {};entropy;dist", k);
            for (i, (entropy, dist)) in result.iter().enumerate() {
                println!(
                    "{}; {:.5}; {:.5?}",
                    i,
                    entropy,
                    dist.iter()
                        .map(|row| row.iter().sum::<f64>())
                        .collect::<Vec<f64>>()
                );
            }
        } else {
            let (best_i, (best_entropy, _)) = result
                .iter()
                .enumerate()
                .max_by_key(|(_, (f, _))| n64(*f))
                .expect("Nonempty");
            println!("{} {} {}", k, best_i, best_entropy);
        }
    }
}

fn linear_diffuse(n: usize, k: usize) -> (Vec<Vec<(usize, f64)>>, Vec<usize>) {
    let mut states = vec![];
    let mut state_to_index = HashMap::new();
    for i in 0usize..1 << n {
        if i.count_ones() == k as u32 {
            state_to_index.insert(i, states.len());
            states.push(i);
        }
    }
    let mut transition = vec![];
    for &i in &states {
        let mut target = HashMap::new();
        for src in 0..n - 1 {
            let m1 = 1 << src;
            let m2 = 1 << (src + 1);
            let p1 = i & m1 > 0;
            let p2 = i & m2 > 0;
            let new = if p1 == p2 { i } else { i ^ m1 ^ m2 };
            *target.entry(new).or_insert(0) += 1;
        }
        let mut row = vec![];
        for (new, weight) in target {
            let index = state_to_index[&new];
            row.push((index, weight as f64 / (n - 1) as f64));
        }
        transition.push(row);
    }
    (transition, states)
}
fn diffusion() {
    let (transition, states) = linear_diffuse(8, 4);
    let mut rng = StdRng::seed_from_u64(0);
    let result = coupled_entropy(&transition, 0.1);
    let (best_i, _) = result.iter().enumerate().max_by_key(|(_, (f, _))| n64(*f)).unwrap();
    for base in (0..11) {
        let i = if base == 5 {
            
        }
        let rows: Vec<f64> = dist.iter().map(|row| row.iter().sum()).collect();
        let mut samples = sample(&rows, 10, &states, &mut rng);
        samples.sort();
        println!(
            "{:>3}; {:.5}; {}",
            i,
            entropy,
            samples
                .iter()
                .map(|i| format!("{:08b}", i))
                .collect::<Vec<String>>()
                .join(", ")
        );
    }
}

fn main() {
    let choice = 1;
    match choice {
        0 => cycles(),
        1 => diffusion(),
        _ => unimplemented!(),
    }
}
