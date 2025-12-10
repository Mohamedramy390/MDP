import numpy as np
import tkinter as tk
import time
import random

# ==========================================
# PART 1: THE BRAIN (Value Iteration)
# ==========================================
states = [0, 1, 2, 3]
V = np.zeros(4)
optimal_policy = {}
gamma = 1

# Run the math loop until convergence
while True:
    delta = 0
    new_V = np.copy(V)
    for s in states:
        if s == 3: continue 
        
        # Calculate Right
        reward_success = 9 if s == 2 else -1
        val_right = 0.8 * (reward_success + V[s+1]) + 0.2 * (-1 + V[s])
        
        # Calculate Left
        if s == 0:
            val_left = 1.0 * (-1 + V[s])
        else:
            val_left = 0.8 * (-1 + V[s-1]) + 0.2 * (-1 + V[s])
            
        best_value = max(val_right, val_left)
        new_V[s] = best_value
        delta = max(delta, abs(new_V[s] - V[s]))
        optimal_policy[s] = "RIGHT" if val_right > val_left else "LEFT"

    V = new_V
    if delta < 0.0001: break

print("Math Complete. Policy:", optimal_policy)

# ==========================================
# PART 2: THE VISUALIZATION (GUI)
# ==========================================

class RobotGUI:
    def __init__(self, master):
        self.master = master
        self.master.title("Robot in a Hallway (MDP)")
        self.canvas = tk.Canvas(master, width=500, height=200, bg="white")
        self.canvas.pack()
        
        # Draw the 4 rooms (Grid)
        for i in range(4):
            x1 = 50 + (i * 100)
            y1 = 50
            x2 = x1 + 100
            y2 = 150
            color = "white"
            text = f"State {i}"
            
            # Make the goal (State 3) look different
            if i == 3: 
                color = "#d4ffd6" # Light green
                text = "GOAL (+10)"
                
            self.canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline="black")
            self.canvas.create_text(x1 + 50, y1 + 20, text=text)

        # Buttons to start simulation
        self.btn_frame = tk.Frame(master)
        self.btn_frame.pack(pady=20)
        
        tk.Button(self.btn_frame, text="Start at 0", command=lambda: self.run_sim(0)).pack(side=tk.LEFT, padx=10)
        tk.Button(self.btn_frame, text="Start at 1", command=lambda: self.run_sim(1)).pack(side=tk.LEFT, padx=10)
        tk.Button(self.btn_frame, text="Start at 2", command=lambda: self.run_sim(2)).pack(side=tk.LEFT, padx=10)
        
        self.status_label = tk.Label(master, text="Click a button to start", font=("Arial", 12))
        self.status_label.pack(pady=10)

    def draw_robot(self, state):
        # Delete old robot
        self.canvas.delete("robot")
        
        # Draw new robot (Red Circle)
        x_center = 50 + (state * 100) + 50
        y_center = 100
        r = 20 # radius
        self.canvas.create_oval(x_center-r, y_center-r, x_center+r, y_center+r, fill="red", tag="robot")
        self.master.update() # Force screen refresh

    def run_sim(self, start_state):
        current = start_state
        steps = 0
        self.draw_robot(current)
        self.status_label.config(text=f"Starting at State {current}...")
        time.sleep(0.5)

        while current != 3:
            # 1. Get Action from Math Policy
            action = optimal_policy[current]
            
            # 2. Simulate Noise
            r = random.random()
            
            if r < 0.8:
                move_text = "Moved Successfully"
                if action == "RIGHT": current += 1
                elif action == "LEFT": current -= 1
            else:
                move_text = "SLIPPED! Stayed in place."
            
            # Update GUI
            steps += 1
            self.draw_robot(current)
            self.status_label.config(text=f"Step {steps}: Action {action} -> {move_text}")
            time.sleep(0.8) # Wait so human can see it
            
            if steps > 20: break # Safety

        self.status_label.config(text=f"Done! Reached Goal in {steps} steps.")

# Run the App
root = tk.Tk()
gui = RobotGUI(root)
root.mainloop()