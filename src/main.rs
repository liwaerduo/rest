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

    let mut atom: Vec<(&String, [f64;3])> = Vec::with_capacity(mol.geom.position.size[1]);

    for i in 0..mol.geom.position.size[1] {
        
        atom.push((&mol.geom.elem[i], [mol.geom.position.data[i * 3 + 0], mol.geom.position.data[i * 3 + 1], mol.geom.position.data[i * 3 + 2]]));
    }
    for i in 0..atom.len() {
        println!("Element: {}, Position: ({}, {}, {})", atom[i].0, atom[i].1[0], atom[i].1[1], atom[i].1[2]);
    }
    
    
    let (gpname, orig, axes) = detect_symm(&atom, None);
    println!("===============================result=======================================================");
    println!("gpname: {:?}", gpname);
    println!("orig: {:?}", orig);
    println!("axes: {:?}",axes);
    let atom = shift_atom(atom, &orig, &axes);
    println!("atom: {:?}", atom);
    //println!("{:?} {:?}",gpname,symm_identical_atoms(&gpname, atom));
    Ok(())
}

fn detect_symm(atoms: &Vec<(&String, [f64;3])>, basis: Option<HashMap<String, isize>>) -> (String, Array1<f64>, Array2<f64>) {

    let tol = TOLERANCE / f64::sqrt(1.0 + atoms.len() as f64);
    println!("atoms.len(): {}",atoms.len() as f64);
    println!("tol: {}", tol);

    
    let decimals = (-tol.log10()).floor() as usize;
    println!("decimals: {}", decimals);

    let mut rawsys = SymmSys::new(&atoms);

    let (w1, u1) = rawsys.cartesian_tensor(1);
    println!("w1: {:?}, u1: {:?}", w1, u1);
    let mut axes = u1.t().to_owned();

    let charge_center = rawsys.charge_center.clone();

    fn allclose(w1: &Array1<f64>, tol: f64) -> bool {
        w1.iter().all(|&x| (x - 0.0).abs() <= tol)
    }
    println!("rawsys.has_icenter(){}",rawsys.has_icenter());
    if allclose(&w1, tol) {
        let gpname = "SO3".to_string();
        // return (gpname, charge_center, Array2::eye(3));
        println!("gpname: {:?}", gpname);
        println!("charge_center: {:?}", charge_center);
        let identity_matrix: Array2<f64> = Array2::eye(3);
        println!("Identity matrix:\n{:?}", identity_matrix);
        (gpname, charge_center, identity_matrix)
    } else if allclose(&w1.slice(s![..2]).to_owned(), tol) { // 线性分子
        let gpname = if rawsys.has_icenter() {
         "Dooh"
        } else {
            "Coov"
        };
        // return gpname, charge_center, axes
        println!("gpname: {:?}",gpname);
        println!("charge_center: {:?}", charge_center);
        println!("axes: {:?}", axes);
        (gpname.to_string(), charge_center, axes)
    } else {
        let (w1_degeneracy, w1_degen_values) = degeneracy(&w1, decimals);
        println!("w1_degeneracy: {:?}",w1_degeneracy);
        println!("w1_degen_values: {:?}",w1_degen_values);
        let (w2, u2) = rawsys.cartesian_tensor(2); // 
        println!("w2: {:?}", w2);
        println!("u2: {:?}", u2);
        let (w2_degeneracy, w2_degen_values) = degeneracy(&w2, decimals);
        println!("w2_degeneracy: {:?}",w2_degeneracy);
        println!("w2_degen_values: {:?}",w2_degen_values);
        
        let n: Option<i32> = None;
        let c2x:Option<Array1<f64>> = None;
        let mirrorx:Option<Array1<f64>> = None;
        fn contains(array: &Array1<usize>, value: usize) -> bool {
            array.iter().any(|&x| x == value)
        }
        fn convert_to_matrix(new_axes: Option<Vec<Array1<f64>>>) -> Option<Array2<f64>> {
            if let Some(axes) = new_axes {
                if axes.is_empty() {
                    return None;
                }
                let n = axes.len();
                let m = axes[0].len();
                let mut matrix = Array2::zeros((n, m));
                for (i, vec) in axes.into_iter().enumerate() {
                    matrix.row_mut(i).assign(&vec);
                }
                Some(matrix)
            } else {
                None
            }
        }
        if contains(&w1_degeneracy, 3) {
            println!("3 in w1_degeneracy:");
            // T, O, I
            let (w3, u3) = rawsys.cartesian_tensor(3);
            let (w3_degeneracy, w3_degen_values) = degeneracy(&w3, decimals);
    
            if contains(&w2_degeneracy, 5) && contains(&w3_degeneracy, 4) && w3_degeneracy.len() == 3 {
                if let (gpname, new_axes) = search_i_group(&mut rawsys) {
                    println!("Group: {:?}, New Axes: {:?}", gpname, new_axes);
                    return (gpname.unwrap(), charge_center, _refine(convert_to_matrix(new_axes).unwrap()));
                }
            } else if contains(&w2_degeneracy, 3) && w2_degeneracy.len() <= 3 {
                if let (gpname, new_axes) = search_ot_group(&mut rawsys) {
                    println!("Group: {}, New Axes: {:?}", gpname.clone().unwrap(), new_axes);
                    return (gpname.unwrap(), charge_center, _refine(convert_to_matrix(new_axes).unwrap()));
                }
            }
        } else if contains(&w1_degeneracy, 2) && w2_degeneracy.iter().any(|&x| x >= 2) {
            println!("2 in w1_degeneracy and numpy.any(w2_degeneracy[w2_degen_values>0] >= 2)");

            let view1 = w1.index_axis(Axis(0), 1).to_owned();
            let view2 = w1.index_axis(Axis(0), 2).to_owned();

            let diff = view1 - view2;
            let abs_diff = diff.mapv(f64::abs);
            let sum_diff = abs_diff.sum();

            if sum_diff < tol {
                
                let new_order = [1, 2, 0];
                let (rows, cols) = axes.dim();
                let new_shape = (rows, cols);

                let mut reordered_axes = Array::zeros(new_shape);

                for (i, &val) in axes.iter().enumerate() {
                    let old_index = i % axes.shape()[0];
                    let new_index = new_order[old_index];
                    let new_position = (i / axes.shape()[0], new_index);
                    reordered_axes[new_position] = val;
                }
            }       
            println!("axes{:?}",axes);
            axes = axes.select(Axis(0), &[1, 2, 0]);
            

            println!("search_c_highest start");
            println!("axes[2]{:?}", &axes.index_axis(Axis(0), 2).to_owned());
            let (c_highest, n) = rawsys.search_c_highest(Some(&axes.index_axis(Axis(0), 2).to_owned()));
            println!("search_c_highest end");
            println!("n{:?}",n);
            let n = if n == 1 { None } else {
                let c2x = rawsys.search_c2x(&axes.index_axis(Axis(0), 2).to_owned(), n);
                let mirrorx = rawsys.search_mirrorx(Some(&axes.index_axis(Axis(0), 2).to_owned()), n);
                Some((c2x, mirrorx))
            };
        } else {
            let n = -1; 
            
        }
        println!("n{:?}",n);
        if let None = n {
            println!("n is None");
            let (zaxis, n) = rawsys.search_c_highest(None);
            if n > 1 {
                if let Some(c2x) = rawsys.search_c2x(&zaxis, n) {
                    axes = _make_axes(&zaxis.view(), &c2x.view());
                } else if let Some(mirrorx) = rawsys.search_mirrorx(Some(&zaxis), n) {
                    axes = _make_axes(&zaxis.view(), &mirrorx.view());
                } else {

                    let identity_axes = vec![
                        array![1.0, 0.0, 0.0],
                        array![0.0, 1.0, 0.0],
                        array![0.0, 0.0, 1.0],
                    ];
                    for axis in identity_axes {
                        if !parallel_vectors(&axis.view(), &zaxis.view(), TOLERANCE) {
                            axes = _make_axes(&zaxis.view(), &axis.view());
                            break;
                        }
                    }
                }
            } else {

                if let Some(mirror) = rawsys.search_mirrorx(None, 1) {
                    let xaxis = array![1.0, 0.0, 0.0];
                    axes = _make_axes(&mirror.view(), &xaxis.view());
                } else {
                    axes = Array2::eye(3);
                }
            }
        }
        let n = n.unwrap();
        if n >= 2 {
            let gpname = "None";
            let axis_view = axes.index_axis(Axis(1), 2);

            let axis_owned = axis_view.to_owned();

            let axis_option = Some(&axis_owned);
            if let Some(c2x) = c2x {
                if rawsys.has_mirror(&axis_option.unwrap()) {
                    let gpname = format!("D{}h", n);
                } else if rawsys.has_improper_rotation(&axis_option.unwrap().view(), n as usize) {
                    let gpname = format!("D{}d", n);
                } else {
                    let gpname = format!("D{}", n);
                }
                let axes = _make_axes(&axis_option.unwrap().view(), &c2x.view());
            } else if let Some(mirrorx) = mirrorx {
                let gpname = format!("C{}v", n);
                let axes = _make_axes(&axis_option.unwrap().view(), &mirrorx.view());
            } else if rawsys.has_mirror(&axis_option.unwrap().clone()) {
                let gpname = format!("C{}h", n);
            } else if rawsys.has_improper_rotation(&axis_option.unwrap().view(), n as usize) {
                let gpname = format!("S{}", n * 2);
            } else {
                let gpname = format!("C{}", n);
            }
            return (gpname.to_string(), charge_center, axes);
        }
        

        
    

        ("Unknown".to_string(), charge_center, axes)
    }

}
fn shift_atom(atoms: Vec<(&String, [f64; 3])>, orig: &Array1<f64>, axis: &Array2<f64>) -> Vec<(String, Array1<f64>)> {
    
    let mut coords: Array2<f64> = Array2::zeros((atoms.len(), orig.len()));
    
    for (i, atom) in atoms.iter().enumerate() {
        
        let atom_coords = Array1::from_vec(atom.1.to_vec());
        coords.slice_mut(s![i, ..]).assign(&atom_coords);
    }
    
    
    let shifted_coords = (coords - orig).dot(&axis.t());

    
    atoms.into_iter()
        .enumerate()
        .map(|(i, (atom_type, _))| (atom_type.clone(), shifted_coords.slice(s![i, ..]).to_owned()))
        .collect()
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
use ndarray::{Array, Array1, Array2, arr1, arr2, Axis, ViewRepr, s, Zip};
use nalgebra::{DMatrix, SymmetricEigen};
use ndarray_linalg::{ UPLO};
use ndarray_linalg::Eigh;
use ndarray::stack;
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
    

    fn cartesian_tensor(&mut self, n: usize) -> (Array1<f64>, Array2<f64>) { 
        let z:Vec<_> = self.atoms.iter().map(|row| row[0]).collect();
        let r:Vec<_> = self.atoms.iter().map(|row| row[1..].to_vec()).collect();
        let ncart = (n + 1) * (n + 2) / 2;
        let natm = z.len();
        println!("z: {:?}", z);
        println!("r: {:?}", r);
        println!("ncart: {}, natm: {}", ncart, natm);
        let z_array = Array1::from(z);
        let z_sum = z_array.sum();
        let tensor = z_array.mapv(|x| (x / z_sum) as f64).view().to_owned().mapv(f64::sqrt);
        println!("tensor: {:?}", tensor);
        let tensor =tensor.into_shape((natm,1)).unwrap(); // into_shape((natm, -1)
        
      

        
        let r = Array2::from_shape_vec((r.len(), r[0].len()), r.into_iter().flatten().collect::<Vec<_>>()).unwrap();
        let mut tensor_ = tensor.clone();
        for i in 0..n {

            let mut result = Array::zeros((natm, tensor_.shape()[1], r.shape()[1]));
     
            for z in 0..tensor_.shape()[0] {
                for i in 0..tensor_.shape()[1] {
                    let outer_product = tensor_[[z, i]] * &r.row(z);
                    result.slice_mut(s![z, i, ..]).assign(&outer_product);
                }
            }
            println!("result_before_reshape: {:?}", result);
            let reshaped_result = result.into_shape((natm, tensor_.shape()[1] * r.shape()[1])).unwrap();
            println!("{}",tensor_.shape()[1]);
            println!("{}",r.shape()[1]);
            
            tensor_ = reshaped_result;
            println!("tensor: {:?}", tensor_);
        }
        let tensor = tensor_;
        println!("tensor: {:?}", tensor);

        fn eigh_dot(tensor: Array2<f64>) -> (Array<f64, ndarray::Ix1>, Array2<f64>) {
            let tensor_t = tensor.t();
            println!("tensor_t: {:?}", tensor_t);
            let dot_product = tensor_t.dot(&tensor);
            println!("tensor.t().dot(&tensor){:?}",dot_product);
            let (eigvals, eigvecs) = dot_product.eigh(UPLO::Upper).unwrap(); 
            (eigvals, eigvecs)
        }
        let (e, c) = eigh_dot(tensor);


        println!("e: \n{:?}", e);
        println!("c: \n{:?}", c);
        (e.slice(s![-(ncart as isize)..]).to_owned(), c.slice(s![.., -(ncart as isize)..]).to_owned())

    }
    fn _vec_in_vecs(vec: &ArrayView1<f64>, vecs: &Array2<f64>) -> bool {
        let norm = (vecs.nrows() as f64).sqrt();
    
        let diffs = vecs.axis_iter(Axis(0))
            .map(|row| {
          
                let row = row.to_owned();
                let diff = row - vec.to_owned();
                diff.mapv(f64::abs).sum()
            })
            .collect::<Vec<f64>>();
    
        let min_diff = diffs.into_iter().fold(f64::INFINITY, f64::min) / norm;
    
        min_diff < TOLERANCE
    }
    
    
    fn symmetric_for(&self, op: SymmetryOp) -> bool {
        for lst in self.group_atoms_by_distance.iter() {
            let indices: Vec<usize> = lst.clone();
            let atoms_array = Array2::from_shape_vec(
                (self.atoms.len(), self.atoms[0].len()),
                self.atoms.clone().into_iter().flatten().collect::<Vec<_>>()
            ).unwrap();

            let atoms_slice = atoms_array.select(Axis(0), &indices);
            let r0 = atoms_slice.slice(s![.., 1..]).to_owned();

            
            let r1 = match op {
                SymmetryOp::Scalar(s) => r0.map(|x| x * s), 
                SymmetryOp::Matrix(ref m) => r0.dot(m),
            };

            let all_in_r0 = r1.axis_iter(Axis(0)).all(|x| Self::_vec_in_vecs(&x, &r0));

            if !all_in_r0 {
                return false;
            }
        }

        true
    }

    fn has_icenter(&self) -> bool {
        self.symmetric_for(SymmetryOp::Scalar(-1.0))
    }

    

    fn search_possible_rotations(&mut self, zaxis: Option<&Array1<f64>>) -> Vec<(Array1<f64>, i32)> {

        fn norm(arr: &Array2<f64>, axis: Axis) -> Array1<f64> {
            arr.map_axis(axis, |row| row.dot(&row).sqrt())
        }

        fn normalize(arr: &Array2<f64>) -> Array2<f64> {
            let norms = norm(arr, Axis(1));
            arr / &norms.insert_axis(Axis(1))
        }

        let mut maybe_cn: Vec<(Array1<f64>, i32)> = Vec::new();

        for lst in self.group_atoms_by_distance.clone() {
            let natm = lst.len();
            if natm > 1 {
                println!("natm > 1");
                println!("atoms{:?}", self.atoms);
                let coords = Array2::from_shape_vec((self.atoms.len(), self.atoms[0].len()), self.atoms.clone().into_iter().flatten().collect::<Vec<_>>()).unwrap().select(Axis(0), &lst.iter().map(|&x| x).collect::<Vec<_>>());


                
                println!("coords{:?}", coords);
                let coords = coords.slice(s![.., 1..]).to_owned();
                println!("coords{:?}", coords);
                
                for i in 1..natm {
                    let row0 = coords.row(0).to_owned();
                    let rowi = coords.row(i).to_owned();
                    
                    if (&row0 + &rowi).sum().abs() > TOLERANCE {
                        maybe_cn.push(((row0 + rowi), 2));
                    } else {
                        maybe_cn.push(((row0 - rowi), 2));
                    }
                }
                
                let r0 = &coords - &coords.row(0);
                println!("r0{:?}", r0);
                let distance = norm(&r0, Axis(1));
                let distance_reshaped = distance.view().insert_axis(Axis(1)); 
                let eq_distance = &distance_reshaped - &distance.view().broadcast((distance.len(), distance.len())).unwrap();
                let eq_distance = eq_distance.mapv(|x| x.abs() < TOLERANCE);
                
                

                for i in 2..natm {
                    for j in (0..i).filter(|&j| eq_distance[(i, j)]) {
                        let cos = r0.row(i).dot(&r0.row(j)) / (distance[i] * distance[j]);
                        let ang = cos.acos();
                        let nfrac = 2.0 * std::f64::consts::PI / (std::f64::consts::PI - ang);
                        let n = (nfrac.round() as i32);
                        if (nfrac - n as f64).abs() < TOLERANCE {
                            println!("r0{:?}", r0);
                            let cross_prod = cross(&r0.row(i), &r0.row(j));
                            maybe_cn.push((cross_prod.to_owned(), n));
                        }
                    }
                }
            }
        }

        println!("maybe_cn: {:?}", maybe_cn);
        
        let vecs: Vec<Array1<f64>> = maybe_cn.iter().map(|x| x.0.clone()).collect();
        let vecs_stacked: Array2<f64> = stack(Axis(0), &vecs.iter().map(|x| x.view()).collect::<Vec<_>>()).unwrap();
        let vecs = vecs_stacked;
        println!("vecs: {:?}", vecs);

        let ns: Array1<i32> = maybe_cn.iter().map(|x| x.1).collect::<Vec<_>>().into();
        let idx = norm(&vecs, Axis(1)).mapv(|x| x > TOLERANCE);
        println!("idx{:?}", idx);
        let indices: Array1<usize> = Array1::from_vec(
            idx.indexed_iter()
                .filter_map(|(i, &x)| if x { Some(i) } else { None })
                .collect()
        );
        println!("indices{:?}", indices);
        let indices_slice: Vec<usize> = indices.to_vec(); 
        let indices_ref: &[usize] = &indices_slice; 
        let vecs = vecs.select(Axis(0), indices_ref);
        println!("vecs{:?}", vecs);
        let mut vecs = normalize(&vecs);
        println!("_normalize(vecs[idx]){:?}", vecs);

        

        let mut ns = ns.select(Axis(0), &indices.to_vec());
        println!("zaxis{:?}", zaxis);
        if let Some(zaxis) = zaxis {
            println!("zaxis is not None");
            fn normalize(vec: &ArrayView1<f64>) -> Array1<f64> {
                let norm = vec.iter().map(|&x| x * x).sum::<f64>().sqrt();
                vec.map(|&x| x / norm).to_owned()
            }
            let zaxis_norm = normalize(&zaxis.view());
            println!("zaxis normalized: {:?}", zaxis_norm);
            let cos = vecs.dot(&zaxis_norm);
            println!("cos: {:?}", cos);
            let mask = cos.mapv(|c| (c - 1.0).abs() < TOLERANCE || (c + 1.0).abs() < TOLERANCE);
            println!("mask: {:?}", mask);
            let indices: Vec<usize> = mask.indexed_iter()
                .filter_map(|(i, &m)| if m { Some(i) } else { None })
                .collect();
            println!("selected indices: {:?}", indices);
            vecs = vecs.select(Axis(0), &indices);
            ns = ns.select(Axis(0), &indices);
        }

        println!("vecs{:?}", vecs);

        let mut possible_cn: Vec<(Array1<f64>, i32)> = Vec::new();
        let mut seen = vec![false; vecs.len_of(Axis(0))];
        println!("seen{:?}", seen);
        for (k, v) in vecs.axis_iter(Axis(0)).enumerate() {
            if !seen[k] {
                println!("k{:?}",k);
                println!("v{:?}",v);
                
                let vecs_slice = vecs.slice(s![k.., ..]);
                let v_broadcasted = v.broadcast(vecs_slice.raw_dim()).unwrap();

                let diff = vecs_slice.to_owned() - v_broadcasted.to_owned();
                let sum_abs_diff = diff.map_axis(Axis(1), |row| row.mapv(f64::abs).sum());

                let where1_mask = sum_abs_diff.mapv(|x| x < TOLERANCE);

                let where1: Vec<usize> = where1_mask.indexed_iter()
                    .filter_map(|(i, &x)| if x { Some(i + k) } else { None })  
                    .collect();

                let where2 = vecs
                    .slice(s![k.., ..])
                    .map_axis(Axis(1), |x| (&x + &v).mapv(f64::abs).sum() < TOLERANCE);

                let vecs_slice = vecs.slice(s![k.., ..]);
                let v_broadcasted = v.broadcast(vecs_slice.raw_dim()).unwrap();
                let diff = vecs_slice.to_owned() + v_broadcasted.to_owned();
                
                let sum_abs_diff = diff.map_axis(Axis(1), |row| row.mapv(f64::abs).sum());
            
                let where2_mask = sum_abs_diff.mapv(|x| x < TOLERANCE);
            
                let where2: Vec<usize> = where2_mask.indexed_iter()
                    .filter_map(|(i, &x)| if x { Some(i + k) } else { None })  
                    .collect();
                
                println!("where1{:?}", where1);
                println!("where2{:?}", where2);
                
                
                for i in where1.iter() {
                    seen[*i] = true;
                
                }
                for i in where2.iter() {
                    seen[*i] = true;
                
                }

                println!("seen{:?}", seen);
                

                let slice1 = vecs.select(Axis(0), &where1).sum_axis(Axis((0)));
                let slice2 = vecs.select(Axis(0), &where2).sum_axis(Axis((0)));
                println!("slice1{:?}", slice1);
                println!("slice2{:?}", slice2);
                fn normalize(vec: &ArrayView1<f64>) -> Array1<f64> {
                    let norm = vec.iter().map(|&x| x * x).sum::<f64>().sqrt();
                    vec.map(|&x| x / norm).to_owned()
                }
                let vk = normalize(&(slice1 - slice2).view());
                println!("vk{:?}", vk);
                // Add to possible_cn
                for n in ns.select(Axis(0), &where1).iter()
                    .chain(ns.select(Axis(0), &where2).iter()) {
                    possible_cn.push((vk.to_owned(), *n));
                }
            }
        }

        possible_cn
    }
    
    fn has_rotation(&self, axis: &ArrayView1<f64>, n: usize) -> bool {
        let theta = 2.0 * std::f64::consts::PI / n as f64;
        let op = rotation_mat(axis, theta).t().to_owned();
        self.symmetric_for(SymmetryOp::Matrix(op.clone()))
    }
    fn has_mirror(&self, perp_vec: &Array1<f64>) -> bool {
        let householder_matrix = householder(perp_vec);
        let transposed_householder = householder_matrix.t(); 
        
        self.symmetric_for(SymmetryOp::Matrix(transposed_householder.to_owned().clone()))
            // .iter()
            // .all(|&x| x)
            // .iter()
            // .all(|&x| x)
    }

    fn search_c_highest(&mut self, zaxis: Option<&Array1<f64>>) -> (Array1<f64>, usize) {
        let possible_cn = self.search_possible_rotations(zaxis);
        println!("possible_cn: {:?}", possible_cn);
        let mut nmax = 1;
        let mut cmax = Array1::from_vec(vec![0.0, 0.0, 1.0]);

        for (cn, n) in possible_cn {
            if n > nmax && self.has_rotation(&cn.view(), n as usize) {
                nmax = n;
                cmax = cn;
            }
        }

        (cmax, nmax as usize)
    }
    fn search_c2x(&self, zaxis: &Array1<f64>,n: usize) -> Option<Array1<f64>> {
        let decimals = (-f64::log10(TOLERANCE)).floor() as usize - 1;
        let mut maybe_c2x = Vec::new();
        
        for lst in self.group_atoms_by_distance.clone() {
            if lst.len() > 1 {
                let r0 = Array2::from_shape_vec((self.atoms.len(), self.atoms[0].len()), self.atoms.clone().into_iter().flatten().collect::<Vec<_>>()).unwrap().select(Axis(0), &lst);
                let zcos = r0.dot(zaxis);
                let zcos = zcos.map(|x| (x / TOLERANCE).round() * TOLERANCE);
                let (_, uniq_zcos) = get_unique_and_indices(&zcos);
                
                for d in uniq_zcos {
                    if d as f64 > TOLERANCE {
                        let mirrord = zcos.mapv(|x| (x - d as f64).abs() < TOLERANCE);
                        
                        if mirrord.iter().filter(|&&b| b).count() == zcos.iter().filter(|&&v| v == d as f64).count() {
                            let above_indices: Vec<usize> = zcos
                                .iter()
                                .enumerate()
                                .filter_map(|(i, &v)| if v == d as f64 { Some(i) } else { None })
                                .collect();
                            let above = r0.select(Axis(0), &above_indices);
                
                            let below_indices: Vec<usize> = mirrord
                                .iter()
                                .enumerate()
                                .filter_map(|(i, &v)| if v { Some(i) } else { None })
                                .collect();
                            let below = r0.select(Axis(0), &below_indices);
                
                            for i in 0..below.nrows() {
                                maybe_c2x.push(above.row(0).to_owned() + below.row(i).to_owned());
                            }
                        }
                    } else if (d as f64).abs() < TOLERANCE {
                        let r1_indices: Vec<usize> = zcos
                            .iter()
                            .enumerate()
                            .filter_map(|(i, &v)| if v == d as f64 { Some(i) } else { None })
                            .collect();
                        let r1 = r0.select(Axis(0), &r1_indices).row(0).to_owned();
                
                        maybe_c2x.push(r1.clone());
                        
                        let r2 = rotation_mat(&zaxis.view(), 2.0 * std::f64::consts::PI / n as f64).dot(&r1);
                
                        if (r1.clone() + r2.clone()).sum() > TOLERANCE {
                            maybe_c2x.push(r1 + r2);
                        } else {
                            maybe_c2x.push(r2 - r1);
                        }
                    }
                }
                
                if !maybe_c2x.is_empty() {
                    maybe_c2x = _normalize(maybe_c2x);
                    
                    let num_rows = maybe_c2x.len();
                    let num_cols = maybe_c2x.first().unwrap().len();

                    let mut array2 = Array2::zeros((num_rows, num_cols));
                    for (i, row) in maybe_c2x.clone().into_iter().enumerate() {
                        array2.row_mut(i).assign(&row);
                    }

                
                    let maybe_c2x = _remove_dupvec(&mut array2);
                    let maybe_c2x: Vec<Array1<f64>> = maybe_c2x.outer_iter().map(|row| row.to_owned()).collect();
                    for c2x in maybe_c2x {
                        if !parallel_vectors(&c2x.view(), &zaxis.view(), TOLERANCE) && self.has_rotation(&c2x.view(), 2) {
                            return Some(c2x);
                        }
                    }
                }
            }
        }
        None
    }

    fn search_mirrorx(&self, zaxis: Option<&Array1<f64>>,n: usize) -> Option<Array1<f64>> {
        if n > 1 {
            for lst in self.group_atoms_by_distance.clone() {
                let natm = lst.len();
                let r0 = Array2::from_shape_vec(
                    (self.atoms.len(), self.atoms[0].len()),
                    self.atoms.clone().into_iter().flatten().collect::<Vec<_>>()
                ).unwrap().row(lst[0]).slice(s![1..]).to_owned();
                if natm > 1 && !parallel_vectors(&r0.view(), &zaxis.unwrap().view(), TOLERANCE) {
                    let r1 = rotation_mat(&zaxis.unwrap().view(), 2.0 * std::f64::consts::PI / n as f64).dot(&r0);
                    let mirrorx = _normalize(vec![r1 - r0])[0].clone();
                    if self.has_mirror(&mirrorx) {
                        return Some(mirrorx);
                    }
                }
            }
        } else {
            for lst in self.group_atoms_by_distance.clone() {
                let natm = lst.len();
                if natm > 1 {
                    let r0 = Array2::from_shape_vec(
                        (self.atoms.len(), self.atoms[0].len()),
                        self.atoms.clone().into_iter().flatten().collect::<Vec<_>>()
                    ).unwrap().select(Axis(0), &lst).slice(s![.., 1..]).to_owned();
                    let mut maybe_mirror = Vec::new();
                    for i in 1..r0.nrows() {
                        let row_i = r0.row(i).to_owned(); 
                        let row_0 = r0.row(0).to_owned(); 
                        
                        let diff = &row_i - &row_0;
                        maybe_mirror.push(diff);
                    }
                    let maybe_mirror = _normalize(maybe_mirror);
                    for mirror in maybe_mirror {
                        if self.has_mirror(&mirror) {
                            return Some(mirror);
                        }
                    }
                }
            }
        }
        None
    }
    fn has_improper_rotation(&self, axis: &ArrayView1<f64>, n: usize) -> bool {


        let s_op = householder(&axis.to_owned()).dot(&rotation_mat(axis, std::f64::consts::PI / n as f64)).reversed_axes();
        
        self.symmetric_for(SymmetryOp::Matrix(s_op.clone()))
    }
    

}
enum SymmetryOp {
    Scalar(f64), // 例如 -1
    Matrix(Array2<f64>), // 旋转矩阵
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
    let mut seen = std::collections::HashMap::new();  
    let mut unique_values = Vec::new();
    let mut indices = Vec::with_capacity(arr.len());

    for &value in arr.iter() {
        if let Some(&index) = seen.get(&FloatWrapper(value)) {
            
            indices.push(index);
        } else {
            
            let new_index = unique_values.len();
            seen.insert(FloatWrapper(value), new_index);
            unique_values.push(value);
            indices.push(new_index);
        }
    }

    (unique_values, indices)
}

fn degeneracy(e: &Array1<f64>, decimals: usize) -> (Array1<usize>, Vec<f64>) {
    // Round the values to the specified number of decimal places
    let rounded_e = e.mapv(|x| (x * 10f64.powi(decimals as i32)).round() / 10f64.powi(decimals as i32));

    // Find unique values and their indices
    
    let (unique_values, inverse_indices) = get_unique_and_indices(&rounded_e);

    // Compute the degeneracies
    let mut degeneracies = Vec::with_capacity(unique_values.len());
    for i in 0..unique_values.len() {
        let count = inverse_indices.iter().filter(|&&x| x == i).count();
        degeneracies.push(count);
    }

    (Array1::from(degeneracies), unique_values)
}

fn search_i_group(rawsys: &mut SymmSys) -> (Option<String>, Option<Vec<Array1<f64>>>) {
    let possible_cn = rawsys.search_possible_rotations(None);
    let c5_axes: Vec<Array1<f64>> = possible_cn
        .iter()
        .filter_map(|(axis, n)| {
            if *n == 5 && rawsys.has_rotation(&axis.view(), 5) {
                Some(axis.clone())
            } else {
                None
            }
        })
        .collect();

    if c5_axes.len() <= 1 {
        return (None, None);
    }

    let zaxis = c5_axes[0].clone();
    let cos: Vec<f64> = c5_axes.iter()
        .skip(1)
        .map(|axis| {
            axis.dot(&zaxis)
        })
        .collect();

    if !cos.iter().all(|&c| {
        (c.abs() - 1.0 / (5.0 as f64).sqrt()).abs() < TOLERANCE
    }) {
        return (None, None);
    }

    let gpname = if rawsys.has_icenter() { "Ih" } else { "I" };

    let mut c5 = c5_axes[1].clone();
    if c5.dot(&zaxis) < 0.0 {
        c5 *= -1.0;
    }

    let c5a = rotation_mat(&zaxis.view(), 6.0 * std::f64::consts::PI / 5.0).dot(&c5);
    let xaxis = &c5a + &c5;

    (Some(gpname.to_string()), Some(vec![zaxis, xaxis]))
}
fn search_ot_group(rawsys: &mut SymmSys) -> (Option<String>, Option<Vec<Array1<f64>>>) {
    let possible_cn = rawsys.search_possible_rotations(None);

    let c4_axes: Vec<Array1<f64>> = possible_cn
        .iter()
        .filter_map(|(axis, n)| {
            if *n == 4 && rawsys.has_rotation(&axis.view(), 4) {
                Some(axis.clone())
            } else {
                None
            }
        })
        .collect();

    if !c4_axes.is_empty() {
        
        if c4_axes.len() > 1 {
            let gpname = if rawsys.has_icenter() { "Oh" } else { "O" };
            return (Some(gpname.to_string()), Some(vec![c4_axes[0].clone(), c4_axes[1].clone()]));
        }
    } else {

        let c3_axes: Vec<Array1<f64>> = possible_cn
            .iter()
            .filter_map(|(axis, n)| {
                if *n == 3 && rawsys.has_rotation(&axis.view(), 3) {
                    Some(axis.clone())
                } else {
                    None
                }
            })
            .collect();

        if c3_axes.len() <= 1 {
            return (None, None);
        }

        let cos: Vec<f64> = c3_axes.iter()
            .skip(1)
            .map(|axis| axis.dot(&c3_axes[0]))
            .collect();

        if !cos.iter().all(|&c| {
            (c.abs() - 1.0 / 3.0).abs() < TOLERANCE
        }) {
            return (None, None);
        }

        let gpname = if rawsys.has_icenter() {
            "Th"
        } else if rawsys.has_mirror(&cross(&c3_axes[0].view(), &c3_axes[1].view())) {
            "Td"
        } else {
            "T"
        };

        let mut c3a = c3_axes[0].clone();
        if c3a.dot(&c3_axes[1]) > 0.0 {
            c3a *= -1.0;
        }

        let c3b = rotation_mat(&c3a.view(), -2.0 * std::f64::consts::PI / 3.0).dot(&c3_axes[1]);
        let c3c = rotation_mat(&c3a.view(), 2.0 * std::f64::consts::PI / 3.0).dot(&c3_axes[1]);

        let zaxis = &c3a + &c3b;
        let xaxis = &c3a + &c3c;

        return (Some(gpname.to_string()), Some(vec![zaxis, xaxis]));
    }

    (None, None)
}

lazy_static! {
    static ref OPERATOR_TABLE: HashMap<&'static str, Vec<&'static str>> = {
        let mut m = HashMap::new();
        m.insert("D2h", vec!["E", "C2x", "C2y", "C2z", "i", "sx", "sy", "sz"]);
        m.insert("C2h", vec!["E", "C2z", "i", "sz"]);
        m.insert("C2v", vec!["E", "C2z", "sx", "sy"]);
        m.insert("D2", vec!["E", "C2x", "C2y", "C2z"]);
        m.insert("Cs", vec!["E", "sz"]);
        m.insert("Ci", vec!["E", "i"]);
        m.insert("C2", vec!["E", "C2z"]);
        m.insert("C1", vec!["E"]);
        m
    };
}
use std::collections::HashSet;
#[derive(Debug)]
struct PointGroupSymmetryError(String);

fn argsort_coords(coords: &Array2<f64>) -> Vec<usize> {
    let mut indices: Vec<usize> = (0..coords.len_of(Axis(0))).collect();
    indices.sort_by(|&i, &j| {
        let a = coords.slice(s![i, ..]);
        let b = coords.slice(s![j, ..]);
        a.iter().partial_cmp(b.iter()).unwrap_or(Ordering::Equal)
    });
    indices
}

fn symm_ops(gpname: &str) -> HashMap<String, Array2<f64>> {
    let mut opdic = HashMap::new();
    opdic.insert("sz".to_string(), Array2::eye(3)); // Placeholder
    opdic
}

fn symm_identical_atoms(gpname: &str, atoms: Vec<(String, Array1<f64>)>) -> Result<Vec<Vec<usize>>, PointGroupSymmetryError> {
    if gpname == "Dooh" {
        let coords: Vec<f64> = atoms.iter().map(|a| a.1.clone()).flatten().collect();
        let coords_array = Array2::from_shape_vec((atoms.len(), 3), coords).unwrap();

        let idx0 = argsort_coords(&coords_array);
        let coords0 = coords_array.select(Axis(0), &idx0);

        let opdic = symm_ops(gpname);
        let newc = coords_array.dot(&opdic["sz"]);
        let idx1 = argsort_coords(&newc);

        let mut dup_atom_ids: Vec<Vec<usize>> = vec![idx0.clone(), idx1.clone()];
        dup_atom_ids.sort_by(|a, b| a[0].cmp(&b[0]));
        let uniq_idx = dup_atom_ids.iter().map(|v| v[0]).collect::<Vec<_>>();

        let eql_atom_ids = uniq_idx.into_iter()
            .map(|i| {
                let mut s: Vec<usize> = vec![i];
                s.sort();
                s
            })
            .collect::<Vec<_>>();

        return Ok(eql_atom_ids);
    } else if gpname == "Coov" {
        let eql_atom_ids = (0..atoms.len()).map(|i| vec![i]).collect::<Vec<Vec<usize>>>();
        return Ok(eql_atom_ids);
    }

    // Fallback for other point groups
    let coords: Vec<f64> = atoms.iter().map(|a| a.1.clone()).flatten().collect();
    let coords_array = Array2::from_shape_vec((atoms.len(), 3), coords).unwrap();
    
    let opdic = symm_ops(gpname);
    let ops = OPERATOR_TABLE[gpname]
        .iter()
        .map(|op| opdic[&op.to_string()].clone())
        .collect::<Vec<_>>();

    let mut dup_atom_ids = vec![];

    let idx = argsort_coords(&coords_array);
    let coords0 = coords_array.select(Axis(0), &idx);

    for op in ops {
        let newc = coords_array.dot(&op);
        let idx = argsort_coords(&newc);

        if !coords0.iter().zip(newc.select(Axis(0), &idx).iter()).all(|(a, b)| (*a - *b).abs() < TOLERANCE) {
            return Err(PointGroupSymmetryError("Symmetry identical atoms not found".to_string()));
        }

        dup_atom_ids.push(idx);
    }

    dup_atom_ids.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
    let uniq_idx = dup_atom_ids.iter().map(|v| v[0]).collect::<Vec<_>>();
    
    let eql_atom_ids = uniq_idx.into_iter()
        .map(|i| {
            let mut s: HashSet<usize> = HashSet::new();
            s.insert(i);
            let mut sorted_vec = s.into_iter().collect::<Vec<_>>();
            sorted_vec.sort();
            sorted_vec
        })
        .collect::<Vec<_>>();

    Ok(eql_atom_ids)
}

fn normalize(vec: &ArrayView1<f64>) -> Array1<f64> {
    let norm = vec.iter().map(|&x| x * x).sum::<f64>().sqrt();
    vec.map(|&x| x / norm).to_owned()
}

fn rotation_mat(vec: &ArrayView1<f64>, theta: f64) -> Array2<f64> {
    let vec = normalize(&vec.view());
    
    let uu = {
        let vec = vec.view();
        let outer_product = Array2::from_shape_fn((3, 3), |(i, j)| vec[i] * vec[j]);
        outer_product
    };
    
    let ux = Array2::from_shape_fn((3, 3), |(i, j)| {
        match (i, j) {
            (0, 1) => -vec[2],
            (0, 2) => vec[1],
            (1, 0) => vec[2],
            (1, 2) => -vec[0],
            (2, 0) => -vec[1],
            (2, 1) => vec[0],
            _ => 0.0,
        }
    });
    
    let c = theta.cos();
    let s = theta.sin();
    let identity = Array2::eye(3);
    
    let r = &identity * c + &ux * s + &uu * (1.0 - c);
    
    r
}
fn cross(a: &ArrayView1<f64>, b: &ArrayView1<f64>) -> Array1<f64> {
    assert_eq!(a.len(), 3);
    assert_eq!(b.len(), 3);

    Array1::from(vec![
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ])
}
fn householder(vec: &Array1<f64>) -> Array2<f64> {
    let vec = normalize(&vec.view()); 
    let eye = Array2::eye(3); 
    let outer_product = vec.view().to_owned().insert_axis(ndarray::Axis(1)).dot(&vec.view().to_owned().insert_axis(ndarray::Axis(0)));
    eye - outer_product * 2.0 
}
use ndarray_linalg::Determinant;
fn _refine(mut axes: Array2<f64>) -> Array2<f64> {
    
    if axes[(2, 2)] < 0.0 {
        axes.row_mut(2).mapv_inplace(|x| -x);
    }

    let (x_id, y_id) = if axes[(0, 0)].abs() > axes[(1, 0)].abs() {
        (0, 1)
    } else {
        (1, 0)
    };
    if axes[(x_id, 0)] < 0.0 {
        axes.row_mut(x_id).mapv_inplace(|x| -x);
    }

    if axes.det().unwrap() < 0.0 {
        axes.row_mut(y_id).mapv_inplace(|x| -x);
    }

    axes
}
use ndarray_linalg::Norm;
fn _normalize(vectors: Vec<Array1<f64>>) -> Vec<Array1<f64>> {
    vectors.into_iter()
        .map(|v| {
            let norm = v.norm();
            if norm > 0.0 {
                v / norm
            } else {
                v
            }
        })
        .collect()
}
fn _make_axes(z: &ArrayView1<f64>, x: &ArrayView1<f64>) -> Array2<f64> {

    let y = cross(z, x);

    let x = cross(&y.view(), z);

    let axes = vec![
        x.to_owned(),
        y.to_owned(),
        z.to_owned(),
    ];
    let normalized_axes = _normalize(axes);

    let mut result = Array2::zeros((3, 3));
    for (i, axis) in normalized_axes.iter().enumerate() {
        result.slice_mut(s![i, ..]).assign(axis);
    }
    
    result
}
fn parallel_vectors(v1: &ArrayView1<f64>, v2: &ArrayView1<f64>, tol: f64) -> bool {
    if v1.iter().all(|&x| (x - 0.0).abs() < tol) || v2.iter().all(|&x| (x - 0.0).abs() < tol) {
        return true;
    }

    let norm_v1 = _normalize(vec![v1.to_owned()]);
    let norm_v1 = norm_v1[0].view(); 
    let norm_v2 = _normalize(vec![v2.to_owned()]);
    let norm_v2 = norm_v2[0].view(); 

    let cos = norm_v1.dot(&norm_v2);

    ( (cos - 1.0).abs() < tol ) || ( (cos + 1.0).abs() < tol )
}
fn _remove_dupvec(vs: &mut Array2<f64>) -> Array2<f64> {
    fn rm_iter(vs: Array2<f64>) -> Array2<f64> {
        let nrows = vs.nrows();

        if nrows <= 1 {
            return vs;
        } else {
            let first_row = vs.row(0).to_owned();
            let rest_rows = vs.slice(s![1.., ..]).to_owned();
            let x = rest_rows.map_axis(Axis(1), |row| {
                let diff = row.to_owned() - first_row.to_owned(); 
                diff.mapv(f64::abs).sum()
            });
            let valid_indices: Vec<usize> = x
                .iter()
                .enumerate()
                .filter_map(|(i, &val)| if val > TOLERANCE { Some(i + 1) } else { None })
                .collect();

            let rest_filtered = vs.select(Axis(0), &valid_indices);

            let rest = rm_iter(rest_filtered);

            let mut result = Array2::zeros((1 + rest.nrows(), vs.ncols()));
            result.slice_mut(s![0, ..]).assign(&first_row);
            result.slice_mut(s![1.., ..]).assign(&rest);

            result
        }
    }

    let pseudo_vs = _pseudo_vectors(vs);
    rm_iter(pseudo_vs)
}
fn _pseudo_vectors(vs: &mut Array2<f64>) -> Array2<f64> {
    let tolerance = 1e-5;
    let mut vs = vs.to_owned();

    let idy0 = vs.column(1).mapv(|v| v.abs() < tolerance);

    let idz0 = vs.column(2).mapv(|v| v.abs() < tolerance);

    for mut row in vs.outer_iter_mut() {
        if row[2] < 0.0 {
            row.mapv_inplace(|x| -x);
        }
    }
    for (mut row, &idz) in vs.outer_iter_mut().zip(idz0.iter()) {
        if row[1] < 0.0 && idz {
            row.mapv_inplace(|x| -x);
        }
    }
    for (mut row, (&idy, &idz)) in vs.outer_iter_mut().zip(idy0.iter().zip(idz0.iter())) {
        if row[0] < 0.0 && idy && idz {
            row.mapv_inplace(|x| -x);
        }
    }

    vs
}
