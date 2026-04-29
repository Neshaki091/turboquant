use pyo3::prelude::*;

mod turboquant;

#[pymodule]
fn tq_native_lib(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Tự động sử dụng số luồng tối đa của hệ thống
    let _ = rayon::ThreadPoolBuilder::new().build_global();

    // TURBOQUANT CORE: SQ+QJL Quantization
    m.add_function(wrap_pyfunction!(turboquant::tq_scan, m)?)?;
    
    // Legacy support (optional but kept for compatibility within TQ classes)
    m.add_function(wrap_pyfunction!(turboquant::tq_master_scan, m)?)?;
    m.add_function(wrap_pyfunction!(turboquant::mse_score_simd, m)?)?;
    
    Ok(())
}
