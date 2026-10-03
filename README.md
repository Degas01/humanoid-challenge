# Humanoid Intern Challenge: Sim-to-Real Trajectory Retargeting via Torque Regularization
Built by Giacomo Demetrio Masone (MSc Robotics, King's College London)

## Architectural Design Choices
Instead of utilizing generic, unconstrained end-to-end imitation learning models that suffer heavily under physical execution, this framework introduces a physics-informed tracking network engineered directly on top of high-frequency MuJoCo environments.

The dataset used consists of real-world trajectories collected manually. This raw telemetry is processed in real time and dynamically mapped onto a simulated Franka Emika Panda arm.

### What Worked:
* **Torque-Rate Control Layer:** Implementing a custom mathematical limiting layer inside the PyTorch forward pass completely eliminated high-frequency joint chattering during contact phases.
* **MuJoCo Step Integration:** Running the physics engine at a granular 0.002s timestep allowed the model to correctly propagate contact forces without visual or physical bouncing.

### What Didn't (Lessons Learned):
* **Standard Multi-Layer Perceptrons:** Training the model without penalizing acceleration caused severe actuation spikes, which would instantly overheat or damage real physical motors. This proved that a tracking loop must always be constrained by acceleration boundaries.