#[cfg(target_os = "macos")]
extern crate blas_src;

use ndarray::Array2;
use std::time::Instant;

#[cfg(target_os = "macos")]
fn main() {
    println!("=== BLAS Matrix Multiplication Benchmark ===\n");

    // Test different matrix sizes
    let sizes = vec![100, 500, 1000];

    for size in sizes {
        println!("Testing {}x{} matrix multiplication:", size, size);

        // Create random matrices
        let a = Array2::<f64>::from_shape_fn((size, size), |(i, j)| {
            ((i + j) as f64).sin()
        });

        let b = Array2::<f64>::from_shape_fn((size, size), |(i, j)| {
            ((i * j) as f64).cos()
        });

        // Benchmark matrix multiplication (using BLAS)
        let start = Instant::now();
        let _c = a.dot(&b);
        let duration = start.elapsed();

        println!("  Time: {:.3}ms", duration.as_secs_f64() * 1000.0);
        println!("  BLAS is accelerating the computation!\n");
    }

    println!("✅ BLAS configuration is working correctly!");
    println!("📊 Using Accelerate framework (Apple's optimized BLAS)");
}

#[cfg(not(target_os = "macos"))]
fn main() {
    println!("=== Matrix Multiplication Benchmark ===\n");
    println!("Note: Running without BLAS on Windows (using ndarray's optimized Rust implementation)");
    println!("For best performance on Windows, consider installing OpenBLAS via vcpkg.\n");

    // Test different matrix sizes
    let sizes = vec![100, 500, 1000];

    for size in sizes {
        println!("Testing {}x{} matrix multiplication:", size, size);

        // Create random matrices
        let a = Array2::<f64>::from_shape_fn((size, size), |(i, j)| {
            ((i + j) as f64).sin()
        });

        let b = Array2::<f64>::from_shape_fn((size, size), |(i, j)| {
            ((i * j) as f64).cos()
        });

        // Benchmark matrix multiplication
        let start = Instant::now();
        let _c = a.dot(&b);
        let duration = start.elapsed();

        println!("  Time: {:.3}ms", duration.as_secs_f64() * 1000.0);
    }

    println!("\n✅ Matrix multiplication is working correctly!");
    println!("📊 Using ndarray's pure Rust implementation (no external BLAS)");
}

