# Distributed Dynamic Optimal Encirclement Control of Multi-Agent Systems Using Classical and Quantum Optimization

![License](https://img.shields.io/badge/license-Academic-blue)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Quantum](https://img.shields.io/badge/QQP--VQE-compatible-purple)

> Official code, models, and simulation results accompanying the paper:  
> *"Distributed Dynamic Optimal Encirclement Control of Multi-Agent Systems Using Classical and Quantum Optimization"*

---

## 🔬 Overview

This repository contains the implementation and simulation framework for **dynamic optimal rotating encirclement control of multi-agent differential-drive robots around moving targets**.

The proposed framework formulates encirclement as an online constrained optimization problem that accounts for robot dynamics, tracking performance, actuator limitations, collision avoidance, and target-group safety.

The framework includes:
- a **centralized optimal encirclement formulation** based on joint quadratic programming
- a **distributed optimal encirclement formulation** based on neighbor-copy consensus ADMM
- **fixed-time distributed estimation** of the moving target-group center
- **fixed-time distributed estimation** of the target-group radius
- explicit **robot--robot and robot--target safety constraints**
- a **quantum-compatible QQP reformulation** of the continuous QPs
- **VQE-based solution of the resulting QUBO subproblems**

The same control formulation can therefore be evaluated using both classical and quantum-compatible optimization while preserving the robot model, reference generation, objective function, and physical and safety constraints.

The numerical studies consider different numbers of robots and targets and compare centralized and distributed implementations using classical QP and QQP--VQE solution approaches.

All numerical studies reported in the paper can be reproduced using the code provided in this repository.

---

## 🌟 Features

- **Dynamic optimal rotating encirclement control** of moving target groups  
- **Differential-drive multi-agent robot model** with controlled-point tracking  
- **Centralized quadratic-programming formulation**  
- **Distributed neighbor-copy consensus ADMM formulation**  
- **Fixed-time distributed target-center estimation**  
- **Fixed-time distributed target-radius estimation**  
- **Robot--robot collision-avoidance constraints**  
- **Robot--target safety constraints**  
- **Wheel-speed, wheel-acceleration, forward/reverse-speed, and yaw-rate constraints**  
- **Classical QP solution using `quadprog`**  
- **Quantum-compatible QQP reformulation**  
- **VQE-based QUBO solution through the DVQE backend**  
- **Automatic generation of scalable simulation parameters** for different numbers of agents  
- **Performance, safety, estimator, and solver diagnostics**  
- **Publication-quality figures and 2-D/3-D animations**  

---

## ⚙️ Supported Modes

The main simulation code supports four configurations:

- centralized optimization with classical QP
- centralized optimization with QQP--VQE
- distributed ADMM with classical local QPs
- distributed ADMM with QQP--VQE local QPs

These modes can be selected through:

```python
ARCHITECTURE = "centralized"   # "centralized" or "distributed"
QP_SOLVER = "quadprog"         # "quadprog" or "qqp"
```

The same physical model, target trajectories, constraints, cost function, and encirclement reference are used across all configurations.

---

## 📊 Numerical Studies

The reported experiments evaluate the proposed framework for multiple problem sizes.

- Classical centralized and distributed studies: \(n=m=6\) to \(30\)
- Quantum-compatible QQP--VQE studies: \(n=m=6\) to \(12\)

For each problem size, the same physical and control parameters are used across the corresponding centralized and distributed experiments.

The code automatically records and exports:
- controlled-point tracking errors
- encirclement-radius errors
- robot--robot and robot--target distances
- target-center and target-radius estimation errors
- QP feasibility and constraint residuals
- solver execution times
- ADMM primal and dual residuals
- ADMM iteration counts
- solver-failure diagnostics

---

## 📚 Citation

If you use the content of this project in your research, please cite:

> M. Hasanzadeh and A. Kargarian, "Distributed Dynamic Optimal Encirclement Control of Multi-Agent Systems Using Classical and Quantum Optimization," to appear.

---

## 👤 Author

**Milad Hasanzadeh**  
Department of Electrical and Computer Engineering  
Louisiana State University  
📧 e.mhasanzadeh1377@yahoo.com  

📅 **Release Date:** 2026  
📄 **License:** Academic and Research Use Only
