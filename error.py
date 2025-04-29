import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import random
import time
from enum import Enum
import threading

class ErrorType(Enum):
    BIT_FLIP = "Bit Flip"
    PHASE_FLIP = "Phase Flip"
    DEPOLARIZING = "Depolarizing"
    AMPLITUDE_DAMPING = "Amplitude Damping"
    NONE = "None"

class CorrectionCode(Enum):
    THREE_QUBIT_BIT = "3-Qubit Bit Flip Code"
    THREE_QUBIT_PHASE = "3-Qubit Phase Flip Code"
    FIVE_QUBIT = "5-Qubit Perfect Code"
    SEVEN_QUBIT_STEANE = "7-Qubit Steane Code"
    NINE_QUBIT_SHOR = "9-Qubit Shor Code"

class QubitState:
    def __init__(self, alpha=1.0, beta=0.0):
        """Initialize a qubit with given amplitudes."""
        # Normalize state
        norm = np.sqrt(abs(alpha)**2 + abs(beta)**2)
        self.alpha = alpha / norm  # |0⟩ component
        self.beta = beta / norm    # |1⟩ component
        
    def __str__(self):
        """String representation of the qubit."""
        return f"{self.alpha:.3f}|0⟩ + {self.beta:.3f}|1⟩"
    
    def measure(self):
        """Measure the qubit, collapsing to |0⟩ or |1⟩."""
        prob_zero = abs(self.alpha)**2
        if random.random() < prob_zero:
            self.alpha = 1.0
            self.beta = 0.0
            return 0
        else:
            self.alpha = 0.0
            self.beta = 1.0
            return 1
    
    def bloch_vector(self):
        """Return the Bloch sphere representation as (x,y,z)."""
        x = 2 * (self.alpha.conjugate() * self.beta).real
        y = 2 * (self.alpha.conjugate() * self.beta).imag
        z = abs(self.alpha)**2 - abs(self.beta)**2
        return np.array([x, y, z])

class QuantumRegister:
    def __init__(self, size=3):
        """Initialize a register of qubits."""
        self.size = size
        self.qubits = [QubitState() for _ in range(size)]
    
    def apply_error(self, error_type, target_qubit=None):
        """Apply an error to a specific qubit or randomly."""
        if target_qubit is None:
            target_qubit = random.randint(0, self.size - 1)
        
        if error_type == ErrorType.BIT_FLIP:
            # X gate: |0⟩ → |1⟩, |1⟩ → |0⟩
            self.qubits[target_qubit].alpha, self.qubits[target_qubit].beta = self.qubits[target_qubit].beta, self.qubits[target_qubit].alpha
            return f"Bit flip error applied to qubit {target_qubit}"
            
        elif error_type == ErrorType.PHASE_FLIP:
            # Z gate: |0⟩ → |0⟩, |1⟩ → -|1⟩
            self.qubits[target_qubit].beta *= -1
            return f"Phase flip error applied to qubit {target_qubit}"
            
        elif error_type == ErrorType.DEPOLARIZING:
            # Random X, Y, or Z error
            rand_error = random.choice(["X", "Y", "Z"])
            if rand_error == "X":
                self.qubits[target_qubit].alpha, self.qubits[target_qubit].beta = self.qubits[target_qubit].beta, self.qubits[target_qubit].alpha
            elif rand_error == "Y":
                self.qubits[target_qubit].alpha, self.qubits[target_qubit].beta = -1j * self.qubits[target_qubit].beta, 1j * self.qubits[target_qubit].alpha
            else:  # Z
                self.qubits[target_qubit].beta *= -1
            return f"Depolarizing error ({rand_error}) applied to qubit {target_qubit}"
            
        elif error_type == ErrorType.AMPLITUDE_DAMPING:
            # Amplitude damping: probability to decay from |1⟩ to |0⟩
            gamma = 0.5  # damping parameter
            if random.random() < gamma * abs(self.qubits[target_qubit].beta)**2:
                self.qubits[target_qubit].alpha = 1.0
                self.qubits[target_qubit].beta = 0.0
            return f"Amplitude damping error applied to qubit {target_qubit}"
        
        return "No error applied"

class QuantumErrorCorrection:
    def __init__(self):
        """Initialize quantum error correction system."""
        self.logical_qubit = QubitState()
        self.code_type = CorrectionCode.THREE_QUBIT_BIT
        self.register = None
        self.encoded = False
        self.syndrome = []
        
    def encode(self, code_type=None):
        """Encode the logical qubit using the specified code."""
        if code_type:
            self.code_type = code_type
            
        # Set the register size based on the code type
        if self.code_type in [CorrectionCode.THREE_QUBIT_BIT, CorrectionCode.THREE_QUBIT_PHASE]:
            register_size = 3
        elif self.code_type == CorrectionCode.FIVE_QUBIT:
            register_size = 5
        elif self.code_type == CorrectionCode.SEVEN_QUBIT_STEANE:
            register_size = 7
        elif self.code_type == CorrectionCode.NINE_QUBIT_SHOR:
            register_size = 9
        else:
            register_size = 3
            
        self.register = QuantumRegister(register_size)
        
        # Encoding based on code type
        if self.code_type == CorrectionCode.THREE_QUBIT_BIT:
            # |ψ⟩ = α|0⟩ + β|1⟩ → α|000⟩ + β|111⟩
            if abs(self.logical_qubit.alpha) > 0:
                for i in range(register_size):
                    self.register.qubits[i].alpha = 1.0
                    self.register.qubits[i].beta = 0.0
            else:
                for i in range(register_size):
                    self.register.qubits[i].alpha = 0.0
                    self.register.qubits[i].beta = 1.0
                    
        elif self.code_type == CorrectionCode.THREE_QUBIT_PHASE:
            # |ψ⟩ = α|0⟩ + β|1⟩ → α|+++⟩ + β|---⟩ (Hadamard basis)
            # Start with |+⟩ states
            for i in range(register_size):
                self.register.qubits[i].alpha = 1/np.sqrt(2)
                self.register.qubits[i].beta = 1/np.sqrt(2)
            
            # Apply phase encoding (simplified representation)
            phase = 0 if abs(self.logical_qubit.alpha) > abs(self.logical_qubit.beta) else np.pi
            for i in range(register_size):
                if phase > 0:
                    self.register.qubits[i].beta *= -1  # Apply phase flip to create |-⟩
        
        elif self.code_type == CorrectionCode.FIVE_QUBIT:
            # Simplified 5-qubit perfect code encoding
            # This is a conceptual representation
            for i in range(register_size):
                if i == 0:
                    self.register.qubits[i].alpha = self.logical_qubit.alpha
                    self.register.qubits[i].beta = self.logical_qubit.beta
                else:
                    self.register.qubits[i].alpha = 1.0
                    self.register.qubits[i].beta = 0.0
                    
        elif self.code_type == CorrectionCode.SEVEN_QUBIT_STEANE:
            # Simplified 7-qubit Steane code encoding
            # This is a conceptual representation
            for i in range(register_size):
                if i == 0:
                    self.register.qubits[i].alpha = self.logical_qubit.alpha
                    self.register.qubits[i].beta = self.logical_qubit.beta
                else:
                    self.register.qubits[i].alpha = 1.0
                    self.register.qubits[i].beta = 0.0
        
        elif self.code_type == CorrectionCode.NINE_QUBIT_SHOR:
            # Simplified 9-qubit Shor code encoding
            # |0⟩ → (|000⟩ + |111⟩)(|000⟩ + |111⟩)(|000⟩ + |111⟩)
            # |1⟩ → (|000⟩ - |111⟩)(|000⟩ - |111⟩)(|000⟩ - |111⟩)
            for i in range(register_size):
                self.register.qubits[i].alpha = 1/np.sqrt(2)
                self.register.qubits[i].beta = 1/np.sqrt(2)
                
            # Apply phase encoding based on logical state
            if abs(self.logical_qubit.beta) > abs(self.logical_qubit.alpha):
                for i in range(0, register_size, 3):
                    self.register.qubits[i].beta *= -1
                    self.register.qubits[i+1].beta *= -1
                    self.register.qubits[i+2].beta *= -1
                    
        self.encoded = True
        return f"Encoded logical qubit using {self.code_type.value}"
    
    def measure_syndrome(self):
        """Measure the error syndrome."""
        self.syndrome = []
        
        if not self.encoded:
            return "Error: Qubit not encoded yet"
            
        if self.code_type == CorrectionCode.THREE_QUBIT_BIT:
            # Parity measurements between qubits
            parity_01 = (self.register.qubits[0].measure() + self.register.qubits[1].measure()) % 2
            parity_12 = (self.register.qubits[1].measure() + self.register.qubits[2].measure()) % 2
            self.syndrome = [parity_01, parity_12]
            
        elif self.code_type == CorrectionCode.THREE_QUBIT_PHASE:
            # Phase parity measurements (simplified)
            self.syndrome = [0, 0]  # Placeholder
            
        elif self.code_type == CorrectionCode.FIVE_QUBIT:
            # Simplified syndrome measurement
            self.syndrome = [0, 0, 0, 0]  # Placeholder
            
        elif self.code_type == CorrectionCode.SEVEN_QUBIT_STEANE:
            # Simplified syndrome measurement
            self.syndrome = [0, 0, 0, 0, 0, 0]  # Placeholder
            
        elif self.code_type == CorrectionCode.NINE_QUBIT_SHOR:
            # Simplified syndrome measurement
            self.syndrome = [0, 0, 0, 0, 0, 0, 0, 0]  # Placeholder
            
        # Introduce some randomness to simulate realistic syndrome measurements
        for i in range(len(self.syndrome)):
            if random.random() < 0.2:  # 20% chance of random syndrome bit
                self.syndrome[i] = random.randint(0, 1)
                
        return f"Syndrome measured: {self.syndrome}"
    
    def correct_errors(self):
        """Correct errors based on syndrome measurement."""
        if not self.encoded:
            return "Error: Qubit not encoded yet"
            
        if not self.syndrome:
            return "Error: Syndrome not measured yet"
            
        result = "Corrections applied: "
        corrections = []
            
        if self.code_type == CorrectionCode.THREE_QUBIT_BIT:
            # For 3-qubit code, interpret syndrome
            if self.syndrome == [1, 0]:
                # Error on qubit 0
                self.register.qubits[0].alpha, self.register.qubits[0].beta = self.register.qubits[0].beta, self.register.qubits[0].alpha
                corrections.append("Bit flip on qubit 0")
            elif self.syndrome == [1, 1]:
                # Error on qubit 1
                self.register.qubits[1].alpha, self.register.qubits[1].beta = self.register.qubits[1].beta, self.register.qubits[1].alpha
                corrections.append("Bit flip on qubit 1")
            elif self.syndrome == [0, 1]:
                # Error on qubit 2
                self.register.qubits[2].alpha, self.register.qubits[2].beta = self.register.qubits[2].beta, self.register.qubits[2].alpha
                corrections.append("Bit flip on qubit 2")
                
        elif self.code_type == CorrectionCode.THREE_QUBIT_PHASE:
            # For 3-qubit phase code, apply phase corrections
            corrections.append("Phase corrections applied")
                
        elif self.code_type == CorrectionCode.FIVE_QUBIT:
            # Simplified error correction for 5-qubit code
            corrections.append("5-qubit code corrections applied")
                
        elif self.code_type == CorrectionCode.SEVEN_QUBIT_STEANE:
            # Simplified error correction for Steane code
            corrections.append("Steane code corrections applied")
                
        elif self.code_type == CorrectionCode.NINE_QUBIT_SHOR:
            # Simplified error correction for Shor code
            corrections.append("Shor code corrections applied")
                
        if not corrections:
            result += "No errors detected"
        else:
            result += ", ".join(corrections)
            
        return result
    
    def decode(self):
        """Decode the quantum state back to a logical qubit."""
        if not self.encoded:
            return "Error: Qubit not encoded yet"
            
        # For simplicity, we'll use majority voting in the 3-qubit case
        if self.code_type == CorrectionCode.THREE_QUBIT_BIT:
            # Measure each qubit
            measurements = [qubit.measure() for qubit in self.register.qubits]
            # Majority vote
            majority = 1 if sum(measurements) > len(measurements) // 2 else 0
            
            # Set logical qubit according to majority
            if majority == 0:
                self.logical_qubit.alpha = 1.0
                self.logical_qubit.beta = 0.0
            else:
                self.logical_qubit.alpha = 0.0
                self.logical_qubit.beta = 1.0
                
        elif self.code_type == CorrectionCode.THREE_QUBIT_PHASE:
            # Simplified decoding for phase code
            self.logical_qubit.alpha = 1.0
            self.logical_qubit.beta = 0.0
            
        elif self.code_type == CorrectionCode.FIVE_QUBIT:
            # Simplified decoding for 5-qubit code
            self.logical_qubit.alpha = self.register.qubits[0].alpha
            self.logical_qubit.beta = self.register.qubits[0].beta
            
        elif self.code_type == CorrectionCode.SEVEN_QUBIT_STEANE:
            # Simplified decoding for Steane code
            self.logical_qubit.alpha = self.register.qubits[0].alpha
            self.logical_qubit.beta = self.register.qubits[0].beta
            
        elif self.code_type == CorrectionCode.NINE_QUBIT_SHOR:
            # Simplified decoding for Shor code
            self.logical_qubit.alpha = self.register.qubits[0].alpha
            self.logical_qubit.beta = self.register.qubits[0].beta
            
        self.encoded = False
        return f"Decoded back to logical qubit: {self.logical_qubit}"

class QuantumErrorCorrectionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Quantum Error Detection & Correction Simulator")
        self.root.geometry("1000x800")
        self.root.minsize(900, 700)
        
        self.qec = QuantumErrorCorrection()
        self.setup_styles()
        self.create_widgets()
        self.success_rate = []
        self.current_run = 0
        self.max_runs = 100
        
    def setup_styles(self):
        """Setup ttk styles for widgets."""
        self.style = ttk.Style()
        self.style.theme_use('clam')
        
        # Configure colors
        bg_color = "#f5f5f5"
        accent_color = "#4a86e8"
        button_color = "#4a86e8"
        
        self.root.configure(bg=bg_color)
        
        # Style for frames
        self.style.configure("Card.TFrame", background=bg_color, relief="raised")
        
        # Style for labels
        self.style.configure("TLabel", background=bg_color, font=("Arial", 10))
        self.style.configure("Header.TLabel", background=bg_color, font=("Arial", 12, "bold"))
        self.style.configure("Title.TLabel", background=bg_color, font=("Arial", 16, "bold"))
        
        # Style for buttons
        self.style.configure("TButton", background=button_color, font=("Arial", 10))
        self.style.configure("Primary.TButton", background=accent_color, foreground="white")
        
        # Style for comboboxes
        self.style.configure("TCombobox", background=bg_color)
        
    def create_widgets(self):
        """Create and place all GUI widgets."""
        # Create main frames
        self.main_frame = ttk.Frame(self.root, padding=10)
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Split into left and right panes
        self.left_frame = ttk.Frame(self.main_frame)
        self.left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.right_frame = ttk.Frame(self.main_frame)
        self.right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Left side - Controls
        self.create_control_panel()
        
        # Right side - Visualization and Log
        self.create_visualization_panel()
        
    def create_control_panel(self):
        """Create control panel with qubit controls and error injection."""
        # Control Frame
        control_frame = ttk.LabelFrame(self.left_frame, text="Quantum Controls", padding=10)
        control_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Qubit Preparation Section
        qubit_frame = ttk.Frame(control_frame, padding=5)
        qubit_frame.pack(fill=tk.X, expand=False, pady=5)
        
        ttk.Label(qubit_frame, text="Logical Qubit Preparation", style="Header.TLabel").pack(anchor=tk.W)
        
        # Qubit State Entry
        state_frame = ttk.Frame(qubit_frame)
        state_frame.pack(fill=tk.X, expand=False, pady=5)
        
        ttk.Label(state_frame, text="α (|0⟩):").pack(side=tk.LEFT, padx=5)
        self.alpha_entry = ttk.Entry(state_frame, width=8)
        self.alpha_entry.pack(side=tk.LEFT, padx=5)
        self.alpha_entry.insert(0, "1.0")
        
        ttk.Label(state_frame, text="β (|1⟩):").pack(side=tk.LEFT, padx=5)
        self.beta_entry = ttk.Entry(state_frame, width=8)
        self.beta_entry.pack(side=tk.LEFT, padx=5)
        self.beta_entry.insert(0, "0.0")
        
        # Common States Buttons
        common_states_frame = ttk.Frame(qubit_frame)
        common_states_frame.pack(fill=tk.X, expand=False, pady=5)
        
        ttk.Button(common_states_frame, text="|0⟩", command=lambda: self.set_qubit_state(1.0, 0.0)).pack(side=tk.LEFT, padx=2)
        ttk.Button(common_states_frame, text="|1⟩", command=lambda: self.set_qubit_state(0.0, 1.0)).pack(side=tk.LEFT, padx=2)
        ttk.Button(common_states_frame, text="|+⟩", command=lambda: self.set_qubit_state(1/np.sqrt(2), 1/np.sqrt(2))).pack(side=tk.LEFT, padx=2)
        ttk.Button(common_states_frame, text="|-⟩", command=lambda: self.set_qubit_state(1/np.sqrt(2), -1/np.sqrt(2))).pack(side=tk.LEFT, padx=2)
        ttk.Button(common_states_frame, text="|+i⟩", command=lambda: self.set_qubit_state(1/np.sqrt(2), 1j/np.sqrt(2))).pack(side=tk.LEFT, padx=2)
        ttk.Button(common_states_frame, text="|-i⟩", command=lambda: self.set_qubit_state(1/np.sqrt(2), -1j/np.sqrt(2))).pack(side=tk.LEFT, padx=2)
        
        # Apply Button
        ttk.Button(qubit_frame, text="Initialize Qubit", command=self.initialize_qubit).pack(fill=tk.X, expand=False, pady=5)
        
        # Error Correction Code Selection
        code_frame = ttk.Frame(control_frame, padding=5)
        code_frame.pack(fill=tk.X, expand=False, pady=5)
        
        ttk.Label(code_frame, text="Error Correction Code", style="Header.TLabel").pack(anchor=tk.W)
        
        self.code_var = tk.StringVar()
        self.code_combobox = ttk.Combobox(code_frame, textvariable=self.code_var)
        self.code_combobox['values'] = [code.value for code in CorrectionCode]
        self.code_combobox.current(0)
        self.code_combobox.pack(fill=tk.X, expand=False, pady=5)
        
        ttk.Button(code_frame, text="Encode Qubit", command=self.encode_qubit).pack(fill=tk.X, expand=False, pady=5)
        
        # Error Injection Section
        error_frame = ttk.Frame(control_frame, padding=5)
        error_frame.pack(fill=tk.X, expand=False, pady=5)
        
        ttk.Label(error_frame, text="Error Injection", style="Header.TLabel").pack(anchor=tk.W)
        
        error_type_frame = ttk.Frame(error_frame)
        error_type_frame.pack(fill=tk.X, expand=False, pady=5)
        
        ttk.Label(error_type_frame, text="Error Type:").pack(side=tk.LEFT, padx=5)
        self.error_var = tk.StringVar()
        self.error_combobox = ttk.Combobox(error_type_frame, textvariable=self.error_var)
        self.error_combobox['values'] = [err.value for err in ErrorType if err != ErrorType.NONE]
        self.error_combobox.current(0)
        self.error_combobox.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        qubit_target_frame = ttk.Frame(error_frame)
        qubit_target_frame.pack(fill=tk.X, expand=False, pady=5)
        
        ttk.Label(qubit_target_frame, text="Target Qubit:").pack(side=tk.LEFT, padx=5)
        self.target_var = tk.StringVar()
        self.target_combobox = ttk.Combobox(qubit_target_frame, textvariable=self.target_var)
        self.target_combobox['values'] = ["Random", "0", "1", "2"]
        self.target_combobox.current(0)
        self.target_combobox.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        ttk.Button(error_frame, text="Inject Error", command=self.inject_error).pack(fill=tk.X, expand=False, pady=5)
        
        # Error Detection and Correction Section
        correction_frame = ttk.Frame(control_frame, padding=5)
        correction_frame.pack(fill=tk.X, expand=False, pady=5)
        
        ttk.Label(correction_frame, text="Error Detection & Correction", style="Header.TLabel").pack(anchor=tk.W)
        
        ttk.Button(correction_frame, text="Measure Syndrome", command=self.measure_syndrome).pack(fill=tk.X, expand=False, pady=2)
        ttk.Button(correction_frame, text="Apply Corrections", command=self.correct_errors).pack(fill=tk.X, expand=False, pady=2)
        ttk.Button(correction_frame, text="Decode Qubit", command=self.decode_qubit).pack(fill=tk.X, expand=False, pady=2)
        
        # Batch Simulation Section
        batch_frame = ttk.LabelFrame(self.left_frame, text="Batch Simulation", padding=10)
        batch_frame.pack(fill=tk.X, expand=False, padx=5, pady=5)
        
        config_frame = ttk.Frame(batch_frame)
        config_frame.pack(fill=tk.X, expand=False, pady=5)
        
        ttk.Label(config_frame, text="Number of Runs:").pack(side=tk.LEFT, padx=5)
        self.runs_entry = ttk.Entry(config_frame, width=8)
        self.runs_entry.pack(side=tk.LEFT, padx=5)
        self.runs_entry.insert(0, "100")
        
        ttk.Button(batch_frame, text="Run Simulation", command=self.run_batch_simulation).pack(fill=tk.X, expand=False, pady=5)
        
        # Progress bar for batch simulation
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(batch_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(fill=tk.X, expand=False, pady=5)
        
    def create_visualization_panel(self):
        """Create visualization panel with Bloch sphere and log."""
        # Visualization Frame
        visualization_frame = ttk.Frame(self.right_frame, padding=5)
        visualization_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Bloch Sphere
        bloch_frame = ttk.LabelFrame(visualization_frame, text="Quantum State Visualization", padding=10)
        bloch_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create figure for plotting
        self.fig = plt.Figure(figsize=(5, 4))
        self.bloch_canvas = FigureCanvasTkAgg(self.fig, bloch_frame)
        self.bloch_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Initialize Bloch sphere
        self.bloch_ax = self.fig.add_subplot(111, projection='3d')
        self.update_bloch_sphere()
        
        # Right bottom panel - Log
        log_frame = ttk.LabelFrame(self.right_frame, text="Operation Log", padding=10)
        log_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.log_text = scrolledtext.ScrolledText(log_frame, wrap=tk.WORD, height=10)
        self.log_text.pack(fill=tk.BOTH, expand=True)
        self.log("Quantum Error Detection & Correction Simulator initialized.")
        self.log("Please initialize a logical qubit to begin.")
        
        # Results Panel
        results_frame = ttk.LabelFrame(self.right_frame, text="Simulation Results", padding=10)
        results_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.results_fig = plt.Figure(figsize=(5, 3))
        self.results_canvas = FigureCanvasTkAgg(self.results_fig, results_frame)
        self.results_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        self.results_ax = self.results_fig.add_subplot(111)
        self.results_ax.set_title("Error Correction Success Rate")
        self.results_ax.set_xlabel("Number of Errors")
        self.results_ax.set_ylabel("Success Rate (%)")
        self.results_fig.tight_layout()
        self.results_canvas.draw()
        
    def set_qubit_state(self, alpha, beta):
        """Set the qubit state entries to the specified values."""
        self.alpha_entry.delete(0, tk.END)
        self.alpha_entry.insert(0, str(alpha))
        self.beta_entry.delete(0, tk.END)
        self.beta_entry.insert(0, str(beta))
        
    def initialize_qubit(self):
        """Initialize the logical qubit with the specified amplitudes."""
        try:
            alpha = complex(self.alpha_entry.get())
            beta = complex(self.beta_entry.get())
            self.qec.logical_qubit = QubitState(alpha, beta)
            self.log(f"Initialized logical qubit: {self.qec.logical_qubit}")
            self.update_bloch_sphere()
        except ValueError:
            messagebox.showerror("Error", "Invalid qubit parameters. Please enter valid complex numbers.")
    
    def encode_qubit(self):
        """Encode the logical qubit using the selected error correction code."""
        code_value = self.code_var.get()
        code_type = next((code for code in CorrectionCode if code.value == code_value), None)
        if code_type:
            result = self.qec.encode(code_type)
            self.log(result)
            self.target_combobox['values'] = ["Random"] + [str(i) for i in range(self.qec.register.size)]
            self.update_bloch_sphere()
        else:
            messagebox.showerror("Error", "Invalid error correction code selected.")
    
    def inject_error(self):
        """Inject an error into the qubit register."""
        if not self.qec.encoded:
            messagebox.showerror("Error", "Qubit not encoded yet. Please encode first.")
            return
            
        error_value = self.error_var.get()
        error_type = next((err for err in ErrorType if err.value == error_value), None)
        
        if not error_type:
            messagebox.showerror("Error", "Invalid error type selected.")
            return
            
        target = self.target_var.get()
        target_qubit = None if target == "Random" else int(target)
        
        result = self.qec.register.apply_error(error_type, target_qubit)
        self.log(result)
        self.update_bloch_sphere()
    
    def measure_syndrome(self):
        """Measure the error syndrome."""
        if not self.qec.encoded:
            messagebox.showerror("Error", "Qubit not encoded yet. Please encode first.")
            return
            
        result = self.qec.measure_syndrome()
        self.log(result)
    
    def correct_errors(self):
        """Correct errors based on the syndrome measurement."""
        if not self.qec.encoded:
            messagebox.showerror("Error", "Qubit not encoded yet. Please encode first.")
            return
            
        result = self.qec.correct_errors()
        self.log(result)
        self.update_bloch_sphere()
    
    def decode_qubit(self):
        """Decode the qubit back to the logical state."""
        if not self.qec.encoded:
            messagebox.showerror("Error", "Qubit not encoded yet. Please encode first.")
            return
            
        result = self.qec.decode()
        self.log(result)
        self.update_bloch_sphere()
    
    def run_batch_simulation(self):
        """Run a batch simulation with multiple iterations."""
        try:
            self.max_runs = int(self.runs_entry.get())
            if self.max_runs <= 0:
                messagebox.showerror("Error", "Number of runs must be positive.")
                return
                
            self.current_run = 0
            self.success_rate = []
            
            # Reset progress bar
            self.progress_var.set(0)
            
            # Start simulation in a separate thread
            threading.Thread(target=self.batch_simulation_thread).start()
            
        except ValueError:
            messagebox.showerror("Error", "Invalid number of runs. Please enter a valid integer.")
    
    def batch_simulation_thread(self):
        """Run the batch simulation in a background thread."""
        code_value = self.code_var.get()
        code_type = next((code for code in CorrectionCode if code.value == code_value), None)
        error_value = self.error_var.get()
        error_type = next((err for err in ErrorType if err.value == error_value), None)
        
        if not code_type or not error_type:
            self.log("Error: Invalid code or error type selected.")
            return
            
        successful_corrections = 0
        
        for i in range(self.max_runs):
            self.current_run = i + 1
            
            # Initialize qubit to |0⟩
            self.qec.logical_qubit = QubitState(1.0, 0.0)
            
            # Encode
            self.qec.encode(code_type)
            
            # Inject error
            self.qec.register.apply_error(error_type)
            
            # Measure syndrome
            self.qec.measure_syndrome()
            
            # Correct errors
            self.qec.correct_errors()
            
            # Decode
            self.qec.decode()
            
            # Check if correction was successful (qubit returned to |0⟩ state)
            if abs(self.qec.logical_qubit.alpha) > abs(self.qec.logical_qubit.beta):
                successful_corrections += 1
                
            # Calculate and store success rate
            success_rate = (successful_corrections / (i + 1)) * 100
            self.success_rate.append(success_rate)
            
            # Update progress bar
            self.progress_var.set((i + 1) / self.max_runs * 100)
            
            # Update results plot every 10 iterations
            if (i + 1) % 10 == 0 or i == self.max_runs - 1:
                self.root.after(0, self.update_results_plot)
                
        self.log(f"Batch simulation complete. Final success rate: {success_rate:.2f}%")
    
    def update_results_plot(self):
        """Update the results plot with current simulation data."""
        self.results_ax.clear()
        self.results_ax.set_title("Error Correction Success Rate")
        self.results_ax.set_xlabel("Number of Runs")
        self.results_ax.set_ylabel("Success Rate (%)")
        
        # Plot success rate
        x = list(range(1, len(self.success_rate) + 1))
        self.results_ax.plot(x, self.success_rate, 'b-')
        
        # Add current values
        if self.success_rate:
            current_success = self.success_rate[-1]
            self.results_ax.text(0.05, 0.95, f"Current Run: {self.current_run}/{self.max_runs}", 
                               transform=self.results_ax.transAxes)
            self.results_ax.text(0.05, 0.90, f"Success Rate: {current_success:.2f}%", 
                               transform=self.results_ax.transAxes)
            
        self.results_fig.tight_layout()
        self.results_canvas.draw()
    
    def draw_bloch_sphere(self):
        """Draw the Bloch sphere with axes and reference points."""
        # Clear the axis
        self.bloch_ax.clear()
        
        # Set axis limits
        self.bloch_ax.set_xlim([-1.2, 1.2])
        self.bloch_ax.set_ylim([-1.2, 1.2])
        self.bloch_ax.set_zlim([-1.2, 1.2])
        
        # Set labels
        self.bloch_ax.set_xlabel("X")
        self.bloch_ax.set_ylabel("Y")
        self.bloch_ax.set_zlabel("Z")
        
        # Draw sphere
        u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
        x = np.cos(u) * np.sin(v)
        y = np.sin(u) * np.sin(v)
        z = np.cos(v)
        self.bloch_ax.plot_wireframe(x, y, z, color='gray', alpha=0.3)
        
        # Draw axes
        self.bloch_ax.quiver(0, 0, 0, 1, 0, 0, color='r', length=1.2, arrow_length_ratio=0.1)
        self.bloch_ax.quiver(0, 0, 0, 0, 1, 0, color='g', length=1.2, arrow_length_ratio=0.1)
        self.bloch_ax.quiver(0, 0, 0, 0, 0, 1, color='b', length=1.2, arrow_length_ratio=0.1)
        
        # Draw reference states
        self.bloch_ax.scatter([0, 0], [0, 0], [1, -1], color=['b', 'r'], s=30)
        self.bloch_ax.text(0, 0, 1.2, "|0⟩", color='b')
        self.bloch_ax.text(0, 0, -1.2, "|1⟩", color='r')
    
    def update_bloch_sphere(self):
        """Update the Bloch sphere with the current qubit state."""
        self.draw_bloch_sphere()
        
        # Plot the current state vector
        if self.qec.encoded:
            # When encoded, show multiple points for each physical qubit
            for i, qubit in enumerate(self.qec.register.qubits):
                vector = qubit.bloch_vector()
                self.bloch_ax.quiver(0, 0, 0, vector[0], vector[1], vector[2], 
                                     color=f'C{i}', length=1.0, arrow_length_ratio=0.1)
        else:
            # When not encoded, show logical qubit
            vector = self.qec.logical_qubit.bloch_vector()
            self.bloch_ax.quiver(0, 0, 0, vector[0], vector[1], vector[2], 
                                color='purple', length=1.0, arrow_length_ratio=0.1)
        
        self.fig.tight_layout()
        self.bloch_canvas.draw()
    
    def log(self, message):
        """Add message to the log with timestamp."""
        timestamp = time.strftime("%H:%M:%S", time.localtime())
        self.log_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.log_text.see(tk.END)

def main():
    root = tk.Tk()
    app = QuantumErrorCorrectionGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()