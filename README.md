### Autonomous racing in the [Formula-Student-Driverless simulator](https://github.com/FS-Driverless/Formula-Student-Driverless-Simulator) with nonlinear MPC and camera-based SLAM
This repository implements:
- a contouring-control based nonlinear MPC
- an extended kalman filter state estimator
- a monocular camera-based SLAM algorithm, which combines deep learning and the perspective-n-point method

A high-level overview of the system is shown below:
![arch_diag](https://github.com/user-attachments/assets/b35a63d1-53ae-4b2a-b5e5-affaa17ed090)

#### Slow mapping lap:
<img src="https://media.githubusercontent.com/media/trouverun/filesizeworkaround/main/mapping.gif"/>

#### Full-speed lap
<img src="https://media.githubusercontent.com/media/trouverun/filesizeworkaround/main/lap.gif"/>
