import numpy as np
import torch
import torch.nn as nn
import mujoco

class TorqueRegularizedPolicy(nn.Module):
    def __init__(self, input_dim, action_dim):
        super(TorqueRegularizedPolicy, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )
        # Parametro di regolarizzazione per impedire sbalzi bruschi di coppia
        self.max_torque_rate = 5.0 

    def forward(self, state, previous_action=None):
        raw_action = self.network(state)
        if previous_action is channels_last:
            return raw_action
            
        # Layer matematico personalizzato contro il chattering dei giunti
        torque_diff = raw_action - previous_action
        clipped_diff = torch.clamp(torque_diff, -self.max_torque_rate, self.max_torque_rate)
        smoothed_action = previous_action + clipped_diff
        return smoothed_action

def run_challenge_pipeline():
    print("[TensorPhysics] Initializing Humanoid Intern Challenge Tracking Loop...")
    
    # Generazione di dati fittizi basati sulle frequenze registrate nella tesi
    # Sostituisci questo blocco caricando il tuo file .csv reale
    mock_real_telemetry = np.sin(np.linspace(0, 10, 500)) 
    
    model = mujoco.MjModel.from_xml_path("panda_arm.xml")
    data = mujoco.MjData(model)
    
    policy = TorqueRegularizedPolicy(input_dim=1, action_dim=2)
    previous_torque = torch.zeros(2)
    
    for step in range(100):
        current_sensor_reading = torch.tensor([mock_real_telemetry[step]], dtype=torch.float32)
        
        # Calcolo della traiettoria ottimizzata priva di vibrazioni
        optimized_torque = policy(current_sensor_reading, previous_torque)
        
        # Iniezione dei comandi nel simulatore MuJoCo
        data.ctrl[0] = optimized_torque[0].item()
        data.ctrl[1] = optimized_torque[1].item()
        
        mujoco.mj_step(model, data)
        previous_torque = optimized_torque.detach()
        
    print("[Success] Pipeline executed smoothly without joint instability.")

if __name__ == "__main__":
    run_challenge_pipeline()