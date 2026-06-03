#![allow(unused)]

/*
 * This code demos how mini_batches can be created by shuffling the indices of the data and then 
 * grouping them into batches. The code creates a vector of items, shuffles their indices, and then 
 * creates mini-batches based on the shuffled indices. Each mini-batch contains references to the 
 * original items in the data vector.
 */

use rand::prelude::SliceRandom;

#[derive(Debug)]
struct Item {
    foo: usize,
}

fn main() {
    let data = vec![
        Item { foo: 0 },
        Item { foo: 1 },
        Item { foo: 2 },
        Item { foo: 3 },
        Item { foo: 4 },
        Item { foo: 5 },
    ];

    let p_data = &data;
    let mini_batch_size = 2; // 2 or 3 - I don't handle remainder (of chunks_exact) in this function

    let mut indices: Vec<usize> = (0..p_data.len()).collect();

    for _ in 0..5 {
        indices.shuffle(&mut rand::rng());
        println!("indices = {:?}", &indices);

        let mini_batches = indices
            .chunks_exact(mini_batch_size)
            .map(|indices_batch| {
                indices_batch
                    .iter()
                    .map(|&i| &p_data[i])
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();

        println!("mini_batches = {:?}", &mini_batches);
    }
}
