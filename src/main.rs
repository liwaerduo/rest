//! # Rust-based Electronic-Structure Tool (REST)
//! 
//! ### Installation
//!   At present, REST can be compiled in Linux only  
//! 
//!   0) Prerequisites:  
//!     - libopenblas.so  
//!     - libcint.so  
//!     - libhdf5.so  
//!     - libxc.so  
//!   1) git clone git@github.com:igor-1982/rest_workspace.git rest_workspace  
//!   2) cd rest_workspace; cp Config.templet Config
//!   3) edit `Config` to make the prerequeisite libraries aformationed accessable via some global variables heading with `REST`.  
//!   4) bash Config; source $HOME/.bash_profile
//!   5-1) cargo build (--release) 
//! 
//! ### Usage
//!   - Some examples are provided in the folder of `rest/examples/`.  
//!   - Detailed user manual is in preparation.  
//!   - Basic usage of varying keywords can be found on the page of [`InputKeywords`](crate::ctrl_io::InputKeywords).
//! 
//! ### Features
//!   1) Use Gaussian Type Orbital (GTO) basis sets  
//!   2) Provide Density Functional Approximations (DFAs) at varying levels from LDA, GGA, Hybrid, to Fifth-rungs, including doubly hybrid approximations, like XYG3, XYGJ-OS, and random-phase approximation (RPA).  
//!   3) Provide some Wave Function Methods (WFMs), like Hartree-Fock approximation (HF) and Moller-Plesset Second-order perturbation (MP2)
//!   4) Provide analytic electronic-repulsive integrals (ERI)s as well as the Resolution-of-idensity (RI) approximation. The RI algorithm is the recommended choice.  
//!   5) High Share Memory Parallelism (SMP) efficiency
//! 
//! 
//! ### Development
//!   1) Provide a tensor library, namely [`rest_tensors`](https://igor-1982.github.io/rest_tensors/rest_tensors/). `rest_tensors` is developed to manipulate
//!    different kinds multi-rank arrays in REST. Thanks to the sophisticated generic, type, and trait systems, `rest_tensors` can be used as easy as `Numpy` and `Scipy` without losing the computation efficiency. 
//!   2) It is very easy and save to develop a parallel program using the Rust language. Please refer to [`rayon`](rayon) for more details.
//!   3) However, attention should pay if you want to use (Sca)Lapcke functions together in the rayon-spawn threads. 
//!    It is because the (Sca)Lapack functions, like `dgemm`, were compiled with OpenMP by default.  The competetion between OpenMP and Rayon threads 
//!    could dramatically deteriorate the final performance.  
//!    Please use [`utilities::omp_set_num_threads_wrapper`] to set the OpenMP treads number in the runtime.
//! 
//! 
//! 
//! ### Presentation
//! ![image](/home/igor/Documents/Package-Pool/rest_workspace/rest/figures/REST电子结构程序1.png) 
//! ![image](/home/igor/Documents/Package-Pool/rest_workspace/rest/figures/REST电子结构程序2.png) 
//! ![image](/home/igor/Documents/Package-Pool/rest_workspace/rest/figures/REST电子结构程序3.png) 
//! ![image](/home/igor/Documents/Package-Pool/rest_workspace/rest/figures/REST电子结构程序4.png) 
//! ![image](/home/igor/Documents/Package-Pool/rest_workspace/rest/figures/REST电子结构程序5-2.png) 
//! ![image](/home/igor/Documents/Package-Pool/rest_workspace/rest/figures/REST电子结构程序6-2.png) 
//! 
#![allow(unused)]
extern crate rest_tensors as tensors;
//extern crate rest_libxc as libxc;
extern crate chrono as time;
#[macro_use]
extern crate lazy_static;
use std::os::raw;
use std::{f64, fs::File, io::Write};
use std::path::PathBuf;
use pyo3::prelude::*;

mod geom_io;
mod basis_io;
mod ctrl_io;
mod dft;
mod utilities;
mod molecule_io;
mod scf_io;
mod initial_guess;
mod ri_pt2;
//mod grad;
mod ri_rpa;
mod isdf;
mod constants;
mod post_scf_analysis;
mod external_libs;
//use rayon;
#[global_allocator]
static GLOBAL: jemallocator::Jemalloc = jemallocator::Jemalloc;
//static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;
//use crate::grad::rhf::Gradient;
use crate::initial_guess::sap::*;

use anyhow;
//use crate::isdf::error_isdf;
use crate::dft::DFA4REST;
use crate::post_scf_analysis::mulliken::mulliken_pop;
//use crate::post_scf_analysis::{post_scf_correlation, print_out_dfa, save_chkfile};
use crate::scf_io::scf;
use time::{DateTime,Local};
use crate::molecule_io::Molecule;
//use crate::isdf::error_isdf;
//use crate::dft::DFA4REST;
use crate::post_scf_analysis::{post_scf_correlation, print_out_dfa, save_chkfile, rand_wf_real_space, cube_build, molden_build, post_ai_correction};


//pub use crate::initial_guess::sap::*;
//use crate::{post_scf_analysis::{rand_wf_real_space, cube_build, molden_build}, isdf::error_isdf, molecule_io::Molecule};



use num_traits::Float;
const TOLERANCE: f64 = 1e-5;



fn main() -> anyhow::Result<()> {

    let ctrl_file = utilities::parse_input().value_of("input_file").unwrap_or("ctrl.in").to_string();
    if ! PathBuf::from(ctrl_file.clone()).is_file() {
        panic!("Input file ({:}) does not exist", ctrl_file);
    }
    let mut mol = Molecule::build(ctrl_file)?;

    println!("Molecule_name: {}", &mol.geom.name);
    println!("mol.geom.elem: {}",format!("{:?}", &mol.geom.elem));
    
    println!("mol.geom.position: {}",format!("{:?}", &mol.geom.position));

    
    let tol = TOLERANCE / f64::sqrt(1.0 + mol.geom.position.size[1] as f64);
    println!("mol.geom.position.size[1] as f64: {}",mol.geom.position.size[1] as f64);
    println!("tol: {}", tol);

    
    let decimals = (-tol.log10()).floor() as i32;
    println!("decimals: {}", decimals);

    let mut atom: Vec<(&String, [f64;3])> = Vec::with_capacity(mol.geom.position.size[1]);

    for i in 0..mol.geom.position.size[1] {
        
        atom.push((&mol.geom.elem[0], [mol.geom.position.data[i * 3 + 0], mol.geom.position.data[i * 3 + 1], mol.geom.position.data[i * 3 + 2]]));
    }
    for i in 0..atom.len() {
        println!("Element: {}, Position: ({}, {}, {})", atom[i].0, atom[i].1[0], atom[i].1[1], atom[i].1[2]);
    }
    let atomtypes = atom_types(&atom, None, None);
    
    let rawsys = SymmSys::new(&atom);
    // print atomtypes
    for (key, value) in &rawsys.atomtypes {
        println!("Atom type: {}, Atom indices: {:?}", key, value);
    }

    Ok(())
}
use std::collections::HashMap;

fn atom_types(atoms: &[(&String, [f64;3])], basis: Option<HashMap<String, isize>>, magmom: Option<Vec<isize>>) -> HashMap<String, Vec<usize>> {
    let mut atmgroup = HashMap::new();

    // Iterate over atoms
    for (ia, a) in atoms.iter().enumerate() {
        let symbol = if a.0.to_uppercase().contains("GHOST") {
            format!("X{}", &a.0[5..])
        } else {
            a.0.clone()
        };

        if let Some(basis_map) = &basis {
            let stdsymb = std_symbol(&symbol);
            if basis_map.contains_key(&symbol) || basis_map.contains_key(&stdsymb) {
                let key = if basis_map.get(&symbol) == basis_map.get(&stdsymb) {
                    stdsymb
                } else {
                    symbol
                };
                atmgroup.entry(key).or_insert_with(Vec::new).push(ia);
            } else {
                atmgroup.entry(stdsymb).or_insert_with(Vec::new).push(ia);
            }
        } else {
            atmgroup.entry(symbol).or_insert_with(Vec::new).push(ia);
        }
    }

    if let Some(magmom_vec) = magmom {
        let mut atmgroup_new = HashMap::new();
        let suffix = vec![(-1, "d"), (0, "o"), (1, "u")].into_iter().collect::<HashMap<_, _>>();
        for (elem, idx) in atmgroup.iter() {
            let uniq_mag = magmom_vec.iter().filter(|&&m| idx.contains(&(magmom_vec.iter().position(|&x| x == m).unwrap()))).collect::<Vec<_>>();
            if uniq_mag.len() > 1 {
                for (i, &mag) in uniq_mag.iter().enumerate() {
                    let subgrp: Vec<_> = idx.iter().filter(|&i| magmom_vec[*i] == *mag).map(|&i| i).collect();
                    if !suffix.contains_key(&mag) {
                        panic!("Magmom should be chosen from [-1, 0, 1], but {} is given", mag);
                    }
                    atmgroup_new.insert(format!("{}_{}", elem, suffix[&mag]), subgrp);
                }
            } else {
                atmgroup_new.insert(elem.clone(), idx.clone());
            }
        }
        atmgroup = atmgroup_new;
    }

    atmgroup
}

// Helper function to get standard symbol
fn std_symbol(symbol: &str) -> String {
    // Placeholder implementation
    symbol.to_string()
}

struct SymmSys {
    atomtypes: HashMap<String, Vec<usize>>,
}

impl SymmSys {
    fn new(atoms: &[(&String, [f64;3])]) -> Self {
        let mut symm_sys = SymmSys {
            atomtypes: HashMap::new(),
        };
        symm_sys.init(atoms);
        symm_sys
    }
    fn init(&mut self, atoms: &[(&String, [f64;3])]) {
        self.atomtypes = atom_types(&atoms, None, None);
    }
}