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
        
        atom.push((&mol.geom.elem[i], [mol.geom.position.data[i * 3 + 0], mol.geom.position.data[i * 3 + 1], mol.geom.position.data[i * 3 + 2]]));
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
use ndarray::{Array, Array1, Array2, arr1, arr2, Axis, ViewRepr, s};
struct SymmSys {
    atomtypes: HashMap<String, Vec<usize>>,
    charge_center: Array1<f64>,
    atoms: Vec<Vec<f64>>,
    group_atoms_by_distance: Vec<Vec<usize>>,
   
}

impl SymmSys {
    fn new(atoms: &[(&String, [f64;3])]) -> Self {
        let mut symm_sys = SymmSys {
            atomtypes: HashMap::new(),
            charge_center: Array::zeros(3),
            atoms: Vec::new(),
            group_atoms_by_distance: Vec::new(),
        
        };
        
        symm_sys.init(atoms);
        symm_sys
    }
    fn init(&mut self, atoms: &[(&String, [f64;3])]) {
        self.atomtypes = atom_types(&atoms, None, None);
        let mut chg1 = (std::f64::consts::PI - 2.0) as f64;
        let mut coords = Vec::new();
        let mut idx = Vec::new();
        let mut fake_chgs = Vec::new();
        for (k, lst) in &self.atomtypes {
            idx.push(lst.clone());
            coords.push(lst.iter().map(|&i| atoms[i].1).collect::<Vec<_>>());
            // print coords
            println!("coords: {:?}", coords);
            let ksymb = _rm_digit(k);
            if ksymb != *k {
                fake_chgs.push(vec![chg1; lst.len()]);
                chg1 *= (std::f64::consts::PI - 2.0) as f64;

            } else if is_ghost_atom(k.as_str()) {
                if ksymb == "X" || ksymb.to_uppercase() == "GHOST" {
                    fake_chgs.push(vec![0.3; lst.len()]);
                } else if ksymb.starts_with('X') {
                    let charge = charge(&ksymb[1..]) as f64 + 0.3;
                    fake_chgs.push(vec![charge; lst.len()]);
                } else if ksymb.starts_with("GHOST") {
                    let charge = charge(&ksymb[5..]) as f64 + 0.3;
                    fake_chgs.push(vec![charge; lst.len()]);
                }
            } else {
                let charge = charge(ksymb.as_str());
                fake_chgs.push(vec![charge as f64; lst.len()]);
            }
        }
        
        
        let coords: Vec<f64> = coords
            .into_iter()
            .flat_map(|inner| inner.into_iter().flatten())
            .collect();
        let mut coords = Array2::<f64>::from_shape_vec((coords.len() / 3, 3), coords).unwrap();
        println!("coords: {:?}", coords);


        let fake_chgs: Vec<f64> = fake_chgs.into_iter().flatten().collect();
        let fake_chgs = Array1::<f64>::from_vec(fake_chgs);
        println!("fake_chgs: {:?}", fake_chgs);

        
        
        self.charge_center = einsum(&fake_chgs, &coords) / fake_chgs.sum();
        println!("charge_center: {:?}", self.charge_center);

        for (i ,coord) in coords.indexed_iter_mut() {
            *coord -= self.charge_center[i.1];
        }

        println!("coords: {:?}", coords);
        
        println!("idx: {:?}",idx);
        let idx = idx.into_iter().flatten().collect::<Vec<usize>>();
        let idx = argsort(&idx, |&a, b| a.cmp(&b));
        println!("idx: {:?}",idx);

        for &i in &idx {
            let chg = fake_chgs[i];
            let mut a = vec![chg];
            for ((j,k)) in coords.indexed_iter() {
                if j.0 == i {
                    a.push(*k);
                }
            }
            self.atoms.push(a);
        }
        println!("atoms: {:?}", self.atoms);
    
        let decimals = (-TOLERANCE.log10() as isize) - 1;
        for index in self.atomtypes.values() {
            let mut c:Vec<Vec<f64>> = Vec::new();
            for i in index {
                c.push(self.atoms[*i][1..].to_vec());
            }
            println!("c: {:?}", c);
            let a: Array2<f64> = Array2::from_shape_vec((c.len(), c[0].len()), c.into_iter().flatten().collect::<Vec<_>>()).unwrap();

            let norms = a.axis_iter(ndarray::Axis(0))
            .map(|row| {
                row.view()
                    .iter()
                    .map(|&x| x * x) 
                    .sum::<f64>()
                    .sqrt()
            })
            .collect::<Vec<_>>();
            let dist = around(&norms, decimals);
            println!("dist: {:?}", dist);

            let (u, idx) = get_unique_and_indices(&dist);
            println!("u: {:?}", u);
            println!("idx: {:?}", idx);
            
            
            for i in 0..u.len() {
                let mut a = Vec::new();
                for j in 0..index.len() {
                    if idx[j] == i {
                        a.push(index[j]);
                    }
                }
                self.group_atoms_by_distance.push(a);
            }

            println!("group_atoms_by_distance: {:?}", self.group_atoms_by_distance);

        }

        
        

    }
}

fn _rm_digit(symb: &str) -> String {
    symb.chars().filter(|c| c.is_alphabetic()).collect()
}
fn is_ghost_atom(symb_or_chg: impl Into<SymbolOrCharge>) -> bool {
    match symb_or_chg.into() {
        SymbolOrCharge::Int(i) => i == 0,
        SymbolOrCharge::Str(s) => s.contains("GHOST") || (s.starts_with('X') && s.chars().nth(1).map_or(false, |c| c.to_ascii_uppercase() != 'E')),
    }
}

/// Represents either an integer or a string.
enum SymbolOrCharge {
    Int(i32),
    Str(String),
}

impl From<i32> for SymbolOrCharge {
    fn from(i: i32) -> Self {
        SymbolOrCharge::Int(i)
    }
}

impl From<String> for SymbolOrCharge {
    fn from(s: String) -> Self {
        SymbolOrCharge::Str(s)
    }
}

impl From<&str> for SymbolOrCharge {
    fn from(s: &str) -> Self {
        SymbolOrCharge::Str(s.to_string())
    }
}

// Example elements proton map
lazy_static! {
    static ref ELEMENTS_PROTON: HashMap<String, i32> = {
        let mut m = HashMap::new();
        m.insert("X".to_string(), 0);
        m.insert("H".to_string(), 1);
        m.insert("HE".to_string(), 2);
        m.insert("LI".to_string(), 3);
        m.insert("BE".to_string(), 4);
        m.insert("B".to_string(), 5);
        m.insert("C".to_string(), 6);
        m.insert("N".to_string(), 7);
        m.insert("O".to_string(), 8);
        m.insert("F".to_string(), 9);
        m.insert("NE".to_string(), 10);
        m.insert("NA".to_string(), 11);
        m.insert("MG".to_string(), 12);
        m.insert("AL".to_string(), 13);
        m.insert("SI".to_string(), 14);
        m.insert("P".to_string(), 15);
        m.insert("S".to_string(), 16);
        m.insert("CL".to_string(), 17);
        m.insert("AR".to_string(), 18);
        m.insert("K".to_string(), 19);
        m.insert("CA".to_string(), 20);
        m.insert("SC".to_string(), 21);
        m.insert("TI".to_string(), 22);
        m.insert("V".to_string(), 23);
        m.insert("CR".to_string(), 24);
        m.insert("MN".to_string(), 25);
        m.insert("FE".to_string(), 26);
        m.insert("CO".to_string(), 27);
        m.insert("NI".to_string(), 28);
        m.insert("CU".to_string(), 29);
        m.insert("ZN".to_string(), 30);
        m.insert("GA".to_string(), 31);
        m.insert("GE".to_string(), 32);
        m.insert("AS".to_string(), 33);
        m.insert("SE".to_string(), 34);
        m.insert("BR".to_string(), 35);
        m.insert("KR".to_string(), 36);
        m.insert("RB".to_string(), 37);
        m.insert("SR".to_string(), 38);
        m.insert("Y".to_string(), 39);
        m.insert("ZR".to_string(), 40);
        m.insert("NB".to_string(), 41);
        m.insert("MO".to_string(), 42);
        m.insert("TC".to_string(), 43);
        m.insert("RU".to_string(), 44);
        m.insert("RH".to_string(), 45);
        m.insert("PD".to_string(), 46);
        m.insert("AG".to_string(), 47);
        m.insert("CD".to_string(), 48);
        m.insert("IN".to_string(), 49);
        m.insert("SN".to_string(), 50);
        m.insert("SB".to_string(), 51);
        m.insert("TE".to_string(), 52);
        m.insert("I".to_string(), 53);
        m.insert("XE".to_string(), 54);
        m.insert("CS".to_string(), 55);
        m.insert("BA".to_string(), 56);
        m.insert("LA".to_string(), 57);
        m.insert("CE".to_string(), 58);
        m.insert("PR".to_string(), 59);
        m.insert("ND".to_string(), 60);
        m.insert("PM".to_string(), 61);
        m.insert("SM".to_string(), 62);
        m.insert("EU".to_string(), 63);
        m.insert("GD".to_string(), 64);
        m.insert("TB".to_string(), 65);
        m.insert("DY".to_string(), 66);
        m.insert("HO".to_string(), 67);
        m.insert("ER".to_string(), 68);
        m.insert("TM".to_string(), 69);
        m.insert("YB".to_string(), 70);
        m.insert("LU".to_string(), 71);
        m.insert("HF".to_string(), 72);
        m.insert("TA".to_string(), 73);
        m.insert("W".to_string(), 74);
        m.insert("RE".to_string(), 75);
        m.insert("OS".to_string(), 76);
        m.insert("IR".to_string(), 77);
        m.insert("PT".to_string(), 78);
        m.insert("AU".to_string(), 79);
        m.insert("HG".to_string(), 80);
        m.insert("TL".to_string(), 81);
        m.insert("PB".to_string(), 82);
        m.insert("BI".to_string(), 83);
        m.insert("PO".to_string(), 84);
        m.insert("AT".to_string(), 85);
        m.insert("RN".to_string(), 86);
        m.insert("FR".to_string(), 87);
        m.insert("RA".to_string(), 88);
        m.insert("AC".to_string(), 89);
        m.insert("TH".to_string(), 90);
        m.insert("PA".to_string(), 91);
        m.insert("U".to_string(), 92);
        m.insert("NP".to_string(), 93);
        m.insert("PU".to_string(), 94);
        m.insert("AM".to_string(), 95);
        m.insert("CM".to_string(), 96);
        m.insert("BK".to_string(), 97);
        m.insert("CF".to_string(), 98);
        m.insert("ES".to_string(), 99);
        m.insert("FM".to_string(), 100);
        m.insert("MD".to_string(), 101);
        m.insert("NO".to_string(), 102);
        m.insert("LR".to_string(), 103);
        m.insert("RF".to_string(), 104);
        m.insert("DB".to_string(), 105);
        m.insert("SG".to_string(), 106);
        m.insert("BH".to_string(), 107);
        m.insert("HS".to_string(), 108);
        m.insert("MT".to_string(), 109);
        m.insert("DS".to_string(), 110);
        m.insert("RG".to_string(), 111);
        m.insert("CN".to_string(), 112);
        m.insert("NH".to_string(), 113);
        m.insert("FL".to_string(), 114);
        m.insert("MC".to_string(), 115);
        m.insert("LV".to_string(), 116);
        m.insert("TS".to_string(), 117);
        m.insert("OG".to_string(), 118);
        m.insert("GHOST".to_string(), 0);
        m
    };
}

fn charge(symb_or_chg: impl Into<SymbolOrCharge>) -> i32 {
    match symb_or_chg.into() {
        SymbolOrCharge::Int(i) => i,
        SymbolOrCharge::Str(s) => {
            let a = s.trim().to_uppercase();
            if a.starts_with("GHOST") || (a.starts_with('X') && a != "XE") {
                0
            } else {
                *ELEMENTS_PROTON.get(&_rm_digit(&a)).unwrap_or(&0)
            }
        },
    }
}
fn einsum(i: &Array1<f64>, ij: &Array2<f64>) -> Array1<f64> {
    let mut result = Array1::<f64>::zeros(ij.ncols());
    for col_idx in 0..ij.ncols() {
        let dot_product = ij.column(col_idx).dot(i);
        result[col_idx] = dot_product;
    }
    result
}
use std::cmp::Ordering;

fn argsort<T, F>(arr: &Vec<T>, cmp: F) -> Vec<usize>
where
    T: Ord,
    F: Fn(&T, &T) -> Ordering,
{
    let mut indices: Vec<usize> = (0..arr.len()).collect();
    indices.sort_unstable_by(|&a, &b| cmp(&arr[a], &arr[b]));
    indices
}
fn around(vec: &Vec<f64>, decimals: isize) -> Array1<f64> {
    let arr = Array1::from_iter(vec.iter().cloned());
    arr.mapv(move |x| {
        let factor = 10f64.powi(decimals as i32);
        (x * factor).round() / factor
    })
}
use ndarray::prelude::*;
use approx::AbsDiffEq;

#[derive(Debug)]
struct FloatWrapper(f64);

impl PartialEq for FloatWrapper {
    fn eq(&self, other: &Self) -> bool {
        
        self.0.abs_diff_eq(&other.0, epsilon()) || self.0 == other.0
    }
}

impl Eq for FloatWrapper {}

impl std::hash::Hash for FloatWrapper {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.0.to_bits().hash(state);
    }
}


fn epsilon() -> f64 {
    1e-10
}

fn get_unique_and_indices(arr: &Array1<f64>) -> (Vec<f64>, Vec<usize>) {
    let mut seen = std::collections::HashSet::new();
    let mut unique_values = Vec::new();
    let mut indices = Vec::new();

    for (i, &value) in arr.iter().enumerate() {
        if !seen.contains(&FloatWrapper(value)) {
            seen.insert(FloatWrapper(value));
            unique_values.push(value);
            indices.push(i);
        } else {
            
            if let Some(&last_index) = indices.last() {
                indices.push(last_index);
            } else {
                indices.push(0);  
            }
        }
    }

    (unique_values, indices)
}
