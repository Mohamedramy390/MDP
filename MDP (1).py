import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import random
import threading
import time

class GridWorldMDP_GUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Grid World Navigation - MDP Solver")
        self.root.geometry("1400x900")
        self.root.configure(bg='#0f172a')
        
        # Grid World Setup (5x5 grid)
        self.grid_size = 5
        self.states = [(i, j) for i in range(self.grid_size) for j in range(self.grid_size)]
        self.actions = ['UP', 'DOWN', 'LEFT', 'RIGHT']
        
        # Special states
        self.goal_state = (4, 4)  # Bottom-right corner
        self.obstacles = [(1, 1), (2, 2), (3, 1)]  # Blocked cells
        self.penalty_states = [(1, 3), (3, 3)]  # Penalty zones
        
        # MDP Parameters
        self.move_prob = 0.8  # Probability of moving in intended direction
        self.slip_prob = 0.1  # Probability of slipping left/right
        self.goal_reward = 100
        self.step_cost = -1
        self.penalty_cost = -10
        self.obstacle_cost = -5
        
        # Configurable parameters
        self.discount = tk.DoubleVar(value=0.9)
        self.theta = tk.DoubleVar(value=0.01)
        
        # Algorithm results
        self.values = np.zeros((self.grid_size, self.grid_size))
        self.policy = [['' for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        self.iterations = 0
        self.converged = False
        
        # Simulation state
        self.current_pos = (0, 0)
        self.simulation_running = False
        self.simulation_history = []
        self.total_reward = 0
        self.steps = 0
        
        # Colors
        self.color_normal = '#334155'
        self.color_goal = '#22c55e'
        self.color_obstacle = '#1e293b'
        self.color_penalty = '#ef4444'
        self.color_current = '#3b82f6'
        
        self.create_widgets()
        self.run_value_iteration()
        
    def create_widgets(self):
        # Title
        title_frame = tk.Frame(self.root, bg='#0f172a')
        title_frame.pack(pady=15)
        
        tk.Label(title_frame, text="🗺️ Grid World Navigation - MDP Solver 🎯", 
                font=('Arial', 26, 'bold'), bg='#0f172a', fg='white').pack()
        tk.Label(title_frame, text="Value Iteration Algorithm for Optimal Path Planning in 5x5 Grid", 
                font=('Arial', 13), bg='#0f172a', fg='#94a3b8').pack()
        
        # Main container
        main_frame = tk.Frame(self.root, bg='#0f172a')
        main_frame.pack(fill='both', expand=True, padx=15, pady=5)
        
        # Left panel - Parameters and Controls
        left_panel = tk.Frame(main_frame, bg='#1e293b', relief='raised', bd=2)
        left_panel.pack(side='left', fill='both', padx=(0, 10))
        
        # Parameters section
        param_frame = tk.LabelFrame(left_panel, text="⚙️ MDP Parameters", 
                                    font=('Arial', 13, 'bold'), bg='#1e293b', fg='white', bd=2)
        param_frame.pack(fill='x', padx=10, pady=10)
        
        # Discount factor
        tk.Label(param_frame, text="Discount Factor (γ):", 
                font=('Arial', 10), bg='#1e293b', fg='white').pack(anchor='w', padx=10, pady=(10,2))
        
        discount_frame = tk.Frame(param_frame, bg='#1e293b')
        discount_frame.pack(fill='x', padx=10)
        
        self.discount_scale = tk.Scale(discount_frame, from_=0.1, to=0.99, resolution=0.01,
                                      orient='horizontal', variable=self.discount,
                                      command=self.on_parameter_change, bg='#334155', 
                                      fg='white', highlightthickness=0, troughcolor='#0f172a')
        self.discount_scale.pack(side='left', fill='x', expand=True)
        
        self.discount_label = tk.Label(discount_frame, text="0.90", 
                                      font=('Arial', 10, 'bold'), bg='#1e293b', fg='#60a5fa', width=5)
        self.discount_label.pack(side='left', padx=5)
        
        # Theta
        tk.Label(param_frame, text="Convergence Threshold (θ):", 
                font=('Arial', 10), bg='#1e293b', fg='white').pack(anchor='w', padx=10, pady=(10,2))
        
        theta_frame = tk.Frame(param_frame, bg='#1e293b')
        theta_frame.pack(fill='x', padx=10, pady=(0,10))
        
        self.theta_scale = tk.Scale(theta_frame, from_=0.001, to=0.1, resolution=0.001,
                                   orient='horizontal', variable=self.theta,
                                   command=self.on_parameter_change, bg='#334155', 
                                   fg='white', highlightthickness=0, troughcolor='#0f172a')
        self.theta_scale.pack(side='left', fill='x', expand=True)
        
        self.theta_label = tk.Label(theta_frame, text="0.010", 
                                   font=('Arial', 10, 'bold'), bg='#1e293b', fg='#60a5fa', width=6)
        self.theta_label.pack(side='left', padx=5)
        
        # MDP Info
        info_frame = tk.LabelFrame(left_panel, text="📊 Environment Info", 
                                  font=('Arial', 13, 'bold'), bg='#1e293b', fg='white', bd=2)
        info_frame.pack(fill='x', padx=10, pady=10)
        
        info_items = [
            ("Grid Size:", "5x5"),
            ("Move Success:", "0.8"),
            ("Slip Prob:", "0.1 each"),
            ("Goal Reward:", "+100"),
            ("Step Cost:", "-1"),
            ("Penalty:", "-10"),
        ]
        
        for i, (label, value) in enumerate(info_items):
            frame = tk.Frame(info_frame, bg='#334155', relief='raised', bd=1)
            frame.pack(fill='x', padx=10, pady=2)
            tk.Label(frame, text=label, font=('Arial', 9), bg='#334155', 
                    fg='#94a3b8', width=15, anchor='w').pack(side='left', padx=5)
            tk.Label(frame, text=value, font=('Arial', 9, 'bold'), bg='#334155', 
                    fg='#60a5fa', anchor='e').pack(side='right', padx=5)
        
        # Legend
        legend_frame = tk.LabelFrame(left_panel, text="🎨 Legend", 
                                    font=('Arial', 13, 'bold'), bg='#1e293b', fg='white', bd=2)
        legend_frame.pack(fill='x', padx=10, pady=10)
        
        legends = [
            ("🎯 Goal", self.color_goal),
            ("⚠️ Penalty", self.color_penalty),
            ("🚫 Obstacle", self.color_obstacle),
            ("🤖 Agent", self.color_current),
        ]
        
        for text, color in legends:
            frame = tk.Frame(legend_frame, bg='#1e293b')
            frame.pack(fill='x', padx=10, pady=3)
            tk.Label(frame, text="  ", bg=color, relief='raised', bd=2, width=3).pack(side='left')
            tk.Label(frame, text=text, font=('Arial', 10), bg='#1e293b', 
                    fg='white').pack(side='left', padx=10)
        
        # Algorithm Results
        results_frame = tk.LabelFrame(left_panel, text="📈 Algorithm Results", 
                                     font=('Arial', 13, 'bold'), bg='#1e293b', fg='white', bd=2)
        results_frame.pack(fill='x', padx=10, pady=10)
        
        res_grid = tk.Frame(results_frame, bg='#1e293b')
        res_grid.pack(fill='x', padx=10, pady=10)
        
        iter_frame = tk.Frame(res_grid, bg='white', relief='raised', bd=2)
        iter_frame.grid(row=0, column=0, padx=5, pady=5, sticky='ew')
        tk.Label(iter_frame, text="Iterations", font=('Arial', 9), bg='white', fg='#64748b').pack(pady=(5,0))
        self.iter_label = tk.Label(iter_frame, text="0", font=('Arial', 18, 'bold'), bg='white', fg='#8b5cf6')
        self.iter_label.pack(pady=(0,5))
        
        status_frame = tk.Frame(res_grid, bg='white', relief='raised', bd=2)
        status_frame.grid(row=0, column=1, padx=5, pady=5, sticky='ew')
        tk.Label(status_frame, text="Status", font=('Arial', 9), bg='white', fg='#64748b').pack(pady=(5,0))
        self.status_label = tk.Label(status_frame, text="Ready", font=('Arial', 14, 'bold'), bg='white', fg='#22c55e')
        self.status_label.pack(pady=(0,5))
        
        res_grid.columnconfigure(0, weight=1)
        res_grid.columnconfigure(1, weight=1)
        
        # Right panel - Grid and Simulation
        right_panel = tk.Frame(main_frame, bg='#1e293b', relief='raised', bd=2)
        right_panel.pack(side='right', fill='both', expand=True, padx=(10, 0))
        
        # Grid Visualization
        grid_frame = tk.LabelFrame(right_panel, text="🗺️ Grid World", 
                                  font=('Arial', 14, 'bold'), bg='#1e293b', fg='white', bd=2)
        grid_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        self.canvas = tk.Canvas(grid_frame, bg='#0f172a', highlightthickness=0)
        self.canvas.pack(fill='both', expand=True, padx=10, pady=10)
        self.canvas.bind('<Configure>', lambda e: self.draw_grid())
        
        # Simulation Controls
        sim_frame = tk.LabelFrame(right_panel, text="🎮 Simulation Controls", 
                                 font=('Arial', 14, 'bold'), bg='#1e293b', fg='white', bd=2)
        sim_frame.pack(fill='x', padx=10, pady=10)
        
        btn_frame = tk.Frame(sim_frame, bg='#1e293b')
        btn_frame.pack(fill='x', padx=10, pady=10)
        
        start_positions = [(0, 0), (0, 4), (4, 0)]
        start_labels = ["Top-Left", "Top-Right", "Bottom-Left"]
        
        for i, (pos, label) in enumerate(zip(start_positions, start_labels)):
            btn = tk.Button(btn_frame, text=f"▶ Start: {label}", font=('Arial', 10, 'bold'),
                          bg='#2563eb', fg='white', relief='raised', bd=2,
                          command=lambda p=pos: self.start_simulation(p), padx=8, pady=6)
            btn.grid(row=0, column=i, padx=3, sticky='ew')
        
        reset_btn = tk.Button(btn_frame, text="⟲ Reset", font=('Arial', 10, 'bold'),
                            bg='#64748b', fg='white', relief='raised', bd=2,
                            command=self.reset_simulation, padx=8, pady=6)
        reset_btn.grid(row=0, column=3, padx=3, sticky='ew')
        
        for i in range(4):
            btn_frame.columnconfigure(i, weight=1)
        
        # Simulation Stats
        stats_frame = tk.Frame(sim_frame, bg='#1e293b')
        stats_frame.pack(fill='x', padx=10, pady=10)
        
        stat_items = [("Steps", "steps"), ("Reward", "reward"), ("Status", "sim_status")]
        for i, (label, attr) in enumerate(stat_items):
            frame = tk.Frame(stats_frame, bg='white', relief='raised', bd=2)
            frame.grid(row=0, column=i, padx=5, sticky='ew')
            tk.Label(frame, text=label, font=('Arial', 9), bg='white', fg='#64748b').pack(pady=(5,0))
            lbl = tk.Label(frame, text="0", font=('Arial', 16, 'bold'), bg='white', fg='#3b82f6')
            lbl.pack(pady=(0,5))
            setattr(self, f"{attr}_label", lbl)
        
        for i in range(3):
            stats_frame.columnconfigure(i, weight=1)
        
        # Simulation History
        history_frame = tk.LabelFrame(right_panel, text="📜 Simulation Log", 
                                     font=('Arial', 14, 'bold'), bg='#1e293b', fg='white', bd=2)
        history_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        scroll_frame = tk.Frame(history_frame, bg='#1e293b')
        scroll_frame.pack(fill='both', expand=True, padx=5, pady=5)
        
        scrollbar = tk.Scrollbar(scroll_frame)
        scrollbar.pack(side='right', fill='y')
        
        self.history_text = tk.Text(scroll_frame, height=8, font=('Courier', 9), 
                                   bg='#0f172a', fg='#e2e8f0', relief='flat',
                                   yscrollcommand=scrollbar.set, highlightthickness=0,
                                   padx=10, pady=5)
        self.history_text.pack(side='left', fill='both', expand=True)
        scrollbar.config(command=self.history_text.yview)
        
    def on_parameter_change(self, value):
        self.discount_label.config(text=f"{self.discount.get():.2f}")
        self.theta_label.config(text=f"{self.theta.get():.3f}")
        self.run_value_iteration()
    
    def get_next_state(self, state, action):
        """Get next state based on action (with boundaries)"""
        i, j = state
        
        if action == 'UP':
            return (max(0, i-1), j)
        elif action == 'DOWN':
            return (min(self.grid_size-1, i+1), j)
        elif action == 'LEFT':
            return (i, max(0, j-1))
        elif action == 'RIGHT':
            return (i, min(self.grid_size-1, j+1))
        
        return state
    
    def get_perpendicular_actions(self, action):
        """Get perpendicular actions for slip probability"""
        if action in ['UP', 'DOWN']:
            return ['LEFT', 'RIGHT']
        else:
            return ['UP', 'DOWN']
    
    def get_transitions(self, state, action):
        """Get transition probabilities"""
        if state in self.obstacles or state == self.goal_state:
            return [(state, 1.0)]
        
        transitions = []
        
        # Intended direction
        next_state = self.get_next_state(state, action)
        if next_state not in self.obstacles:
            transitions.append((next_state, self.move_prob))
        else:
            transitions.append((state, self.move_prob))
        
        # Slip perpendicular
        perp_actions = self.get_perpendicular_actions(action)
        for perp_action in perp_actions:
            next_state = self.get_next_state(state, perp_action)
            if next_state not in self.obstacles:
                transitions.append((next_state, self.slip_prob))
            else:
                transitions.append((state, self.slip_prob))
        
        return transitions
    
    def get_reward(self, state, action, next_state):
        """Get reward for transition"""
        if next_state == self.goal_state:
            return self.goal_reward
        elif next_state in self.obstacles:
            return self.obstacle_cost
        elif next_state in self.penalty_states:
            return self.penalty_cost
        else:
            return self.step_cost
    
    def run_value_iteration(self):
        """Run Value Iteration algorithm"""
        V = np.zeros((self.grid_size, self.grid_size))
        policy = [['' for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        iteration = 0
        
        gamma = self.discount.get()
        theta_val = self.theta.get()
        
        while iteration < 1000:
            iteration += 1
            V_new = V.copy()
            
            for state in self.states:
                if state in self.obstacles or state == self.goal_state:
                    continue
                
                i, j = state
                max_value = -np.inf
                best_action = 'UP'
                
                for action in self.actions:
                    q_value = 0
                    transitions = self.get_transitions(state, action)
                    
                    for next_state, prob in transitions:
                        ni, nj = next_state
                        reward = self.get_reward(state, action, next_state)
                        q_value += prob * (reward + gamma * V[ni, nj])
                    
                    if q_value > max_value:
                        max_value = q_value
                        best_action = action
                
                V_new[i, j] = max_value
                policy[i][j] = best_action
            
            delta = np.max(np.abs(V_new - V))
            
            if delta < theta_val:
                self.converged = True
                break
            
            V = V_new
        
        self.values = V
        self.policy = policy
        self.iterations = iteration
        
        self.update_display()
    
    def update_display(self):
        self.iter_label.config(text=str(self.iterations))
        if self.converged:
            self.status_label.config(text="✓ Converged", fg='#22c55e')
        else:
            self.status_label.config(text="⟳ Computing", fg='#eab308')
        
        self.draw_grid()
    
    def draw_grid(self):
        """Draw the grid world"""
        self.canvas.delete('all')
        
        width = self.canvas.winfo_width()
        height = self.canvas.winfo_height()
        
        if width <= 1 or height <= 1:
            return
        
        cell_width = (width - 20) / self.grid_size
        cell_height = (height - 20) / self.grid_size
        cell_size = min(cell_width, cell_height)
        
        start_x = (width - cell_size * self.grid_size) / 2
        start_y = (height - cell_size * self.grid_size) / 2
        
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                x1 = start_x + j * cell_size
                y1 = start_y + i * cell_size
                x2 = x1 + cell_size
                y2 = y1 + cell_size
                
                state = (i, j)
                
                # Determine color
                if state == self.current_pos:
                    color = self.color_current
                    text_color = 'white'
                elif state == self.goal_state:
                    color = self.color_goal
                    text_color = 'white'
                elif state in self.obstacles:
                    color = self.color_obstacle
                    text_color = '#64748b'
                elif state in self.penalty_states:
                    color = self.color_penalty
                    text_color = 'white'
                else:
                    color = self.color_normal
                    text_color = 'white'
                
                # Draw cell
                self.canvas.create_rectangle(x1, y1, x2, y2, fill=color, 
                                            outline='#475569', width=2)
                
                # Draw icon
                icon_y = y1 + cell_size * 0.25
                if state == self.current_pos:
                    self.canvas.create_text((x1+x2)/2, icon_y, text="🤖", 
                                          font=('Arial', int(cell_size*0.3)))
                elif state == self.goal_state:
                    self.canvas.create_text((x1+x2)/2, icon_y, text="🎯", 
                                          font=('Arial', int(cell_size*0.3)))
                elif state in self.obstacles:
                    self.canvas.create_text((x1+x2)/2, icon_y, text="🚫", 
                                          font=('Arial', int(cell_size*0.3)))
                elif state in self.penalty_states:
                    self.canvas.create_text((x1+x2)/2, icon_y, text="⚠️", 
                                          font=('Arial', int(cell_size*0.3)))
                
                # Draw value
                if state not in self.obstacles:
                    value_y = y1 + cell_size * 0.55
                    self.canvas.create_text((x1+x2)/2, value_y, 
                                          text=f"{self.values[i, j]:.1f}", 
                                          font=('Arial', int(cell_size*0.12), 'bold'), 
                                          fill=text_color)
                
                # Draw policy arrow
                if state not in self.obstacles and state != self.goal_state:
                    arrow_y = y1 + cell_size * 0.78
                    action = self.policy[i][j]
                    arrow = {'UP': '↑', 'DOWN': '↓', 'LEFT': '←', 'RIGHT': '→'}.get(action, '')
                    self.canvas.create_text((x1+x2)/2, arrow_y, text=arrow, 
                                          font=('Arial', int(cell_size*0.25), 'bold'), 
                                          fill='#fbbf24')
    
    def start_simulation(self, start_pos):
        """Start simulation from given position"""
        if self.simulation_running:
            return
        
        if start_pos in self.obstacles:
            messagebox.showwarning("Invalid Start", "Cannot start on an obstacle!")
            return
        
        self.current_pos = start_pos
        self.simulation_history = []
        self.total_reward = 0
        self.steps = 0
        self.simulation_running = True
        
        self.steps_label.config(text="0")
        self.reward_label.config(text="0")
        self.sim_status_label.config(text="⟳ Running", fg='#3b82f6')
        self.history_text.delete(1.0, tk.END)
        
        self.draw_grid()
        
        thread = threading.Thread(target=self.run_simulation)
        thread.daemon = True
        thread.start()
    
    def run_simulation(self):
        """Run the simulation"""
        while self.current_pos != self.goal_state and self.steps < 100:
            time.sleep(0.6)
            
            i, j = self.current_pos
            action = self.policy[i][j]
            
            if not action:
                break
            
            transitions = self.get_transitions(self.current_pos, action)
            
            rand = random.random()
            cumulative_prob = 0
            next_pos = self.current_pos
            
            for next_state, prob in transitions:
                cumulative_prob += prob
                if rand <= cumulative_prob:
                    next_pos = next_state
                    break
            
            reward = self.get_reward(self.current_pos, action, next_pos)
            self.steps += 1
            self.total_reward += reward
            
            self.root.after(0, self.update_simulation_step, action, next_pos, reward)
            
            self.current_pos = next_pos
        
        self.root.after(0, self.end_simulation)
    
    def update_simulation_step(self, action, next_pos, reward):
        """Update GUI with simulation step"""
        arrow = {'UP': '↑', 'DOWN': '↓', 'LEFT': '←', 'RIGHT': '→'}.get(action, '')
        reward_sign = "+" if reward > 0 else ""
        
        self.history_text.insert(tk.END, 
            f"Step {self.steps}: {self.current_pos} {arrow}{action} → {next_pos}  "
            f"R: {reward_sign}{reward}\n")
        self.history_text.see(tk.END)
        
        self.steps_label.config(text=str(self.steps))
        self.reward_label.config(text=f"{self.total_reward:.1f}")
        
        self.draw_grid()
    
    def end_simulation(self):
        """End the simulation"""
        self.simulation_running = False
        if self.current_pos == self.goal_state:
            self.sim_status_label.config(text="✓ Goal!", fg='#22c55e')
        else:
            self.sim_status_label.config(text="✗ Stuck", fg='#ef4444')
        self.draw_grid()
    
    def reset_simulation(self):
        """Reset simulation"""
        self.simulation_running = False
        self.current_pos = (0, 0)
        self.simulation_history = []
        self.total_reward = 0
        self.steps = 0
        
        self.steps_label.config(text="0")
        self.reward_label.config(text="0")
        self.sim_status_label.config(text="○ Ready", fg='#64748b')
        self.history_text.delete(1.0, tk.END)
        self.draw_grid()


def main():
    root = tk.Tk()
    app = GridWorldMDP_GUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()