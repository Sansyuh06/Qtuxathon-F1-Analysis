import numpy as np
import pandas as pd
import time
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.svm import SVC
import cirq
import os
import random
from PIL import Image, ImageTk
class QuantumKernel:
    def __init__(self, feature_dimension, n_qubits=None):
        self.feature_dimension = feature_dimension
        self.n_qubits = n_qubits if n_qubits else min(8, feature_dimension)
        self.qubits = [cirq.GridQubit(0, i) for i in range(self.n_qubits)]
    def create_quantum_circuit(self, x):
        """Create a quantum circuit with ZZ feature map encoding"""
        circuit = cirq.Circuit()
        # First layer of Hadamards
        circuit.append(cirq.H.on_each(self.qubits))
        # Feature encoding with rotations
        for i, qubit in enumerate(self.qubits):
            if i < len(x):
                circuit.append(cirq.rz(x[i]).on(qubit))
        # Second layer with entangling operations
        for i in range(self.n_qubits):
            for j in range(i+1, self.n_qubits):
                circuit.append(cirq.CNOT(self.qubits[i], self.qubits[j]))
                if i < len(x) and j < len(x):
                    circuit.append(cirq.rz(x[i] * x[j]).on(self.qubits[j]))
                circuit.append(cirq.CNOT(self.qubits[i], self.qubits[j]))
        # Add error detection qubit and circuit
        error_qubit = cirq.GridQubit(1, 0)
        # Parity check for error detection
        circuit.append(cirq.H.on(error_qubit))
        for qubit in self.qubits:
            circuit.append(cirq.CNOT(error_qubit, qubit))
        circuit.append(cirq.H.on(error_qubit))
        return circuit
    def quantum_kernel_matrix(self, X1, X2):
        """Compute the quantum kernel matrix between two sets of data points"""
        kernel_matrix = np.zeros((len(X1), len(X2)))
        for i, x1 in enumerate(X1):
            for j, x2 in enumerate(X2):
                # Create both circuits
                circuit_1 = self.create_quantum_circuit(x1)
                circuit_2 = self.create_quantum_circuit(x2)
                # Compute inner product using simulation
                simulator = cirq.Simulator()
                state_1 = simulator.simulate(circuit_1).final_state_vector
                state_2 = simulator.simulate(circuit_2).final_state_vector
                # Calculate inner product
                overlap = np.abs(np.dot(np.conjugate(state_1), state_2))**2
                kernel_matrix[i, j] = overlap

        return kernel_matrix

class QuantumSVM:
    def __init__(self, feature_dimension, n_qubits=None, C=1.0):
        self.quantum_kernel = QuantumKernel(feature_dimension, n_qubits)
        self.C = C
        self.svm = None
        self.X_train = None
        
    def fit(self, X_train, y_train):
        """Train the quantum SVM model"""
        # Limit training size for performance
        max_samples = min(100, len(X_train))
        X_subset = X_train[:max_samples]
        y_subset = y_train[:max_samples]
        
        kernel_matrix = self.quantum_kernel.quantum_kernel_matrix(X_subset, X_subset)
        
        # Use precomputed kernel
        self.svm = SVC(kernel='precomputed', C=self.C)
        self.svm.fit(kernel_matrix, y_subset)
        
        self.X_train = X_subset
    
    def predict(self, X_test):
        """Predict using the quantum SVM model"""
        if self.X_train is None or self.svm is None:
            return None
            
        kernel_matrix = self.quantum_kernel.quantum_kernel_matrix(X_test, self.X_train)
        return self.svm.predict(kernel_matrix)

class FeatureExtractor:
    def __init__(self, window_size=10):
        self.window_size = window_size
        
    def extract_features_from_windows(self, data):
        """Extract features from windows of raw sensor data"""
        if len(data) < self.window_size:
            return None
            
        n_windows = len(data) - self.window_size + 1
        n_features = data.shape[1] * 6  # mean, std, max, min, mean_diff, std_diff
        
        X = np.zeros((n_windows, n_features))
        
        for i in range(n_windows):
            window = data[i:i+self.window_size]
            X[i] = self._extract_single_window(window)
            
        return X
        
    def _extract_single_window(self, window):
        """Extract features from a single window of data"""
        # Calculate basic statistics
        means = np.mean(window, axis=0)
        stds = np.std(window, axis=0)
        maxs = np.max(window, axis=0)
        mins = np.min(window, axis=0)
        
        # Calculate derivatives (jerk)
        diffs = np.diff(window, axis=0)
        mean_diffs = np.mean(diffs, axis=0)
        std_diffs = np.std(diffs, axis=0)
        
        # Combine features
        features = np.concatenate([means, stds, maxs, mins, mean_diffs, std_diffs])
        
        return features

class DataAnalyzer:
    def __init__(self):
        self.data = None
        self.features = None
        self.labels = None
        self.feature_extractor = FeatureExtractor()
        self.scaler = StandardScaler()
        self.model = None
        self.training_results = None
        
    def load_data(self, filename):
        """Load and preprocess data from CSV file"""
        try:
            self.data = pd.read_csv(filename)
            return True
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
            
    def prepare_data(self):
        """Prepare data for analysis and modeling"""
        if self.data is None:
            return False
            
        # Extract sensor data
        if 'label' in self.data.columns:
            sensor_cols = [col for col in self.data.columns if col != 'label' and col != 'timestamp']
            self.labels = self.data['label'].values
        else:
            sensor_cols = [col for col in self.data.columns if col != 'timestamp']
            self.labels = None
            
        sensor_data = self.data[sensor_cols].values
        
        # Extract features
        self.features = self.feature_extractor.extract_features_from_windows(sensor_data)
        
        if self.features is not None and self.labels is not None:
            # Adjust labels to match feature windows
            self.labels = self.labels[:len(self.features)]
            return True
        return False
        
    def train_model(self, test_size=0.2):
        """Train a quantum SVM model on the data"""
        if self.features is None or self.labels is None:
            return False
            
        # Scale features
        X_scaled = self.scaler.fit_transform(self.features)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, self.labels, test_size=test_size, random_state=42)
            
        # Train model
        feature_dim = X_train.shape[1]
        n_qubits = min(6, feature_dim)  # Limit qubits to avoid simulation overhead
        
        self.model = QuantumSVM(feature_dim, n_qubits=n_qubits)
        
        # Use subset for training (quantum simulation is resource-intensive)
        max_train = min(100, len(X_train))
        self.model.fit(X_train[:max_train], y_train[:max_train])
        
        # Evaluate on test data
        max_test = min(50, len(X_test))
        y_pred = self.model.predict(X_test[:max_test])
        
        # Store results
        self.training_results = {
            'accuracy': accuracy_score(y_test[:max_test], y_pred),
            'confusion_matrix': confusion_matrix(y_test[:max_test], y_pred),
            'report': classification_report(y_test[:max_test], y_pred, output_dict=True),
            'y_true': y_test[:max_test],
            'y_pred': y_pred
        }
        
        return True
        
    def get_data_summary(self):
        """Get summary statistics of the data"""
        if self.data is None:
            return None
            
        summary = {
            'n_samples': len(self.data),
            'n_features': len(self.data.columns) - (2 if 'label' in self.data.columns else 1),
            'class_distribution': self.data['label'].value_counts().to_dict() if 'label' in self.data.columns else None,
            'feature_stats': self.data.describe().to_dict()
        }
        
        return summary

class HardwareSimulator:
    def __init__(self):
        self.connected = False
        self.status = "Disconnected"
        self.connection_thread = None
        
    def connect(self):
        """Simulate connecting to hardware"""
        if not self.connected:
            self.connection_thread = threading.Thread(target=self._simulate_connection)
            self.connection_thread.daemon = True
            self.connection_thread.start()
            return True
        return False
            
    def disconnect(self):
        """Simulate disconnecting from hardware"""
        if self.connected:
            self.connected = False
            self.status = "Disconnected"
            return True
        return False
            
    def _simulate_connection(self):
        """Simulate the connection process"""
        self.status = "Connecting..."
        time.sleep(2)  # Simulate connection delay
        
        # 90% chance of successful connection
        if random.random() < 0.9:
            self.connected = True
            self.status = "Connected"
        else:
            self.connected = False
            self.status = "Connection failed"

class QuantumDriverApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Quantum Driver Behavior Analysis")
        self.root.geometry("1200x800")
        self.root.minsize(1000, 700)
        
        # Set theme and style
        self.style = ttk.Style()
        self.style.theme_use('clam')
        
        # Configure colors and styles
        self.style.configure('TFrame', background='#f0f0f0')
        self.style.configure('TButton', font=('Helvetica', 10))
        self.style.configure('TLabel', font=('Helvetica', 10), background='#f0f0f0')
        self.style.configure('Header.TLabel', font=('Helvetica', 14, 'bold'), background='#f0f0f0')
        self.style.configure('Title.TLabel', font=('Helvetica', 16, 'bold'), background='#f0f0f0')
        self.style.configure('Status.TLabel', font=('Helvetica', 10), padding=5)
        self.style.configure('Success.Status.TLabel', foreground='green')
        self.style.configure('Error.Status.TLabel', foreground='red')
        
        # Initialize components
        self.hardware = HardwareSimulator()
        self.analyzer = DataAnalyzer()
        self.dataset_filename = None
        
        # Create main layout
        self.create_main_layout()
        
        # Set up tab navigation
        self.setup_tabs()
        
        # Initialize status bar
        self.status_var = tk.StringVar(value="Ready")
        self.status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
    def create_main_layout(self):
        """Create the main application layout"""
        # Create menu bar
        menubar = tk.Menu(self.root)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Open Dataset", command=self.load_dataset)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        menubar.add_cascade(label="File", menu=file_menu)
        
        # Analysis menu
        analysis_menu = tk.Menu(menubar, tearoff=0)
        analysis_menu.add_command(label="Run Analysis", command=self.analyze_data)
        analysis_menu.add_command(label="Train Model", command=self.train_model)
        menubar.add_cascade(label="Analysis", menu=analysis_menu)
        
        # Hardware menu
        hardware_menu = tk.Menu(menubar, tearoff=0)
        hardware_menu.add_command(label="Connect Hardware", command=self.connect_hardware)
        hardware_menu.add_command(label="Disconnect Hardware", command=self.disconnect_hardware)
        menubar.add_cascade(label="Hardware", menu=hardware_menu)
        
        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        help_menu.add_command(label="About", command=self.show_about)
        menubar.add_cascade(label="Help", menu=help_menu)
        
        self.root.config(menu=menubar)
        
        # Main container
        self.main_frame = ttk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
    def setup_tabs(self):
        """Set up the tabbed interface"""
        self.notebook = ttk.Notebook(self.main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Create tabs
        self.dashboard_tab = ttk.Frame(self.notebook)
        self.data_analysis_tab = ttk.Frame(self.notebook)
        self.quantum_model_tab = ttk.Frame(self.notebook)
        self.prediction_tab = ttk.Frame(self.notebook)
        
        self.notebook.add(self.dashboard_tab, text="Dashboard")
        self.notebook.add(self.data_analysis_tab, text="Data Analysis")
        self.notebook.add(self.quantum_model_tab, text="Quantum Model")
        self.notebook.add(self.prediction_tab, text="Prediction")
        
        # Set up each tab
        self.setup_dashboard_tab()
        self.setup_data_analysis_tab()
        self.setup_quantum_model_tab()
        self.setup_prediction_tab()
        
    def setup_dashboard_tab(self):
        """Set up the dashboard tab"""
        # Create a clean dashboard layout
        dashboard_frame = ttk.Frame(self.dashboard_tab)
        dashboard_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Title
        title_frame = ttk.Frame(dashboard_frame)
        title_frame.pack(fill=tk.X, pady=(0, 20))
        
        ttk.Label(title_frame, text="Quantum Driver Behavior Analysis", 
                  style='Title.TLabel').pack(anchor=tk.W)
        ttk.Label(title_frame, 
                  text="Analyze driver behavior using quantum machine learning techniques").pack(anchor=tk.W)
                  
        # Main content in two columns
        content_frame = ttk.Frame(dashboard_frame)
        content_frame.pack(fill=tk.BOTH, expand=True)
        
        # Left column - Status and actions
        left_col = ttk.Frame(content_frame)
        left_col.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        # Hardware status section
        hw_frame = ttk.LabelFrame(left_col, text="Hardware Status")
        hw_frame.pack(fill=tk.X, pady=10)
        
        hw_status_frame = ttk.Frame(hw_frame)
        hw_status_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Label(hw_status_frame, text="Status:").grid(row=0, column=0, sticky=tk.W)
        self.hw_status_label = ttk.Label(hw_status_frame, text="Disconnected")
        self.hw_status_label.grid(row=0, column=1, padx=5, sticky=tk.W)
        
        ttk.Separator(hw_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=10)
        
        hw_btn_frame = ttk.Frame(hw_frame)
        hw_btn_frame.pack(fill=tk.X, padx=10, pady=10)
        
        self.connect_btn = ttk.Button(hw_btn_frame, text="Connect Hardware", 
                               command=self.connect_hardware)
        self.connect_btn.pack(side=tk.LEFT, padx=5)
        
        self.disconnect_btn = ttk.Button(hw_btn_frame, text="Disconnect Hardware", 
                                  command=self.disconnect_hardware, state=tk.DISABLED)
        self.disconnect_btn.pack(side=tk.LEFT, padx=5)
        
        # Dataset section
        dataset_frame = ttk.LabelFrame(left_col, text="Dataset")
        dataset_frame.pack(fill=tk.X, pady=10)
        
        dataset_info_frame = ttk.Frame(dataset_frame)
        dataset_info_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Label(dataset_info_frame, text="Current Dataset:").grid(row=0, column=0, sticky=tk.W)
        self.dataset_label = ttk.Label(dataset_info_frame, text="No dataset loaded")
        self.dataset_label.grid(row=0, column=1, padx=5, sticky=tk.W)
        
        ttk.Separator(dataset_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=10)
        
        dataset_btn_frame = ttk.Frame(dataset_frame)
        dataset_btn_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Button(dataset_btn_frame, text="Load Dataset", 
                   command=self.load_dataset).pack(side=tk.LEFT, padx=5)
        
        # Quick actions section
        actions_frame = ttk.LabelFrame(left_col, text="Quick Actions")
        actions_frame.pack(fill=tk.X, pady=10)
        
        actions_btn_frame = ttk.Frame(actions_frame)
        actions_btn_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Button(actions_btn_frame, text="Analyze Data", 
                   command=self.analyze_data).pack(fill=tk.X, pady=5)
        ttk.Button(actions_btn_frame, text="Train Model", 
                   command=self.train_model).pack(fill=tk.X, pady=5)
        ttk.Button(actions_btn_frame, text="Run Prediction", 
                   command=self.run_prediction).pack(fill=tk.X, pady=5)
        
        # Right column - Summary and visuals
        right_col = ttk.Frame(content_frame)
        right_col.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(10, 0))
        
        # System summary section
        summary_frame = ttk.LabelFrame(right_col, text="System Summary")
        summary_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Create summary widgets
        summary_canvas = tk.Canvas(summary_frame, bg="#f5f5f5", highlightthickness=0)
        summary_canvas.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Use a small image to represent the system
        try:
            # Create a circle to represent the system status
            size = 120
            status_colors = {
                "ready": "#4CAF50",  # Green
                "working": "#2196F3",  # Blue
                "error": "#F44336",   # Red
                "idle": "#9E9E9E"     # Gray
            }
            
            # Create a circular indicator
            self.system_status_canvas = tk.Canvas(summary_frame, width=size, height=size, 
                                               bg="#f5f5f5", highlightthickness=0)
            self.system_status_canvas.pack(pady=20)
            
            # Draw the circle
            self.status_circle = self.system_status_canvas.create_oval(10, 10, size-10, size-10, 
                                                                    fill=status_colors["idle"], width=2)
            self.system_status_canvas.create_text(size/2, size/2, text="System\nIdle", fill="white", 
                                              font=("Helvetica", 12, "bold"))
            
            # System stats
            stats_frame = ttk.Frame(summary_frame)
            stats_frame.pack(fill=tk.X, padx=10, pady=10)
            
            ttk.Label(stats_frame, text="System Ready", 
                      font=("Helvetica", 12, "bold")).pack(anchor=tk.W)
            ttk.Label(stats_frame, text="No active dataset or model").pack(anchor=tk.W)
            ttk.Label(stats_frame, text="Hardware status: Not connected").pack(anchor=tk.W)
            
        except Exception as e:
            print(f"Error creating system visuals: {e}")
            ttk.Label(summary_frame, text="System visualization unavailable").pack(pady=20)
        
    def setup_data_analysis_tab(self):
        """Set up the data analysis tab"""
        # Create frames
        control_frame = ttk.Frame(self.data_analysis_tab)
        control_frame.pack(fill=tk.X, padx=20, pady=10)
        
        visualization_frame = ttk.Frame(self.data_analysis_tab)
        visualization_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        # Controls
        ttk.Label(control_frame, text="Data Analysis", style='Header.TLabel').grid(row=0, column=0, sticky=tk.W)
        
        ttk.Button(control_frame, text="Load Dataset", 
                   command=self.load_dataset).grid(row=0, column=1, padx=5)
        ttk.Button(control_frame, text="Analyze Data", 
                   command=self.analyze_data).grid(row=0, column=2, padx=5)
        
        # Create a notebook for different visualizations
        viz_notebook = ttk.Notebook(visualization_frame)
        viz_notebook.pack(fill=tk.BOTH, expand=True)
        
        # Create visualization tabs
        self.raw_data_tab = ttk.Frame(viz_notebook)
        self.feature_viz_tab = ttk.Frame(viz_notebook)
        self.stats_tab = ttk.Frame(viz_notebook)
        
        viz_notebook.add(self.raw_data_tab, text="Raw Data")
        viz_notebook.add(self.feature_viz_tab, text="Features")
        viz_notebook.add(self.stats_tab, text="Statistics")
        
        # Set up raw data visualization
        raw_data_frame = ttk.Frame(self.raw_data_tab)
        raw_data_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create figure for raw data plots
        self.raw_data_fig = Figure(figsize=(5, 4), dpi=100)
        self.raw_data_canvas = FigureCanvasTkAgg(self.raw_data_fig, raw_data_frame)
        self.raw_data_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Set up feature visualization
        feature_frame = ttk.Frame(self.feature_viz_tab)
        feature_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create figure for feature plots
        self.feature_fig = Figure(figsize=(5, 4), dpi=100)
        self.feature_canvas = FigureCanvasTkAgg(self.feature_fig, feature_frame)
        self.feature_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Set up stats visualization
        stats_frame = ttk.Frame(self.stats_tab)
        stats_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create frame for statistics text
        self.stats_text = tk.Text(stats_frame, wrap=tk.WORD, height=20, width=50)
        self.stats_text.pack(fill=tk.BOTH, expand=True)
        
    def setup_quantum_model_tab(self):
        """Set up the quantum model tab"""
        # Create frames
        model_control_frame = ttk.Frame(self.quantum_model_tab)
        model_control_frame.pack(fill=tk.X, padx=20, pady=10)
        
        model_viz_frame = ttk.Frame(self.quantum_model_tab)
        model_viz_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        # Controls
        ttk.Label(model_control_frame, text="Quantum Model Training", 
                  style='Header.TLabel').grid(row=0, column=0, sticky=tk.W)
        
        ttk.Button(model_control_frame, text="Train Model", 
                   command=self.train_model).grid(row=0, column=1, padx=5)
        
        # Split visualization area
        model_left_frame = ttk.Frame(model_viz_frame)
        model_left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        model_right_frame = ttk.Frame(model_viz_frame)
        model_right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))
        
        # Set up left side - Confusion Matrix
        cm_frame = ttk.LabelFrame(model_left_frame, text="Confusion Matrix")
        cm_frame.pack(fill=tk.BOTH, expand=True)
        
        self.cm_fig = Figure(figsize=(5, 4), dpi=100)
        self.cm_canvas = FigureCanvasTkAgg(self.cm_fig, cm_frame)
        self.cm_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Set up right side - Performance Metrics
        metrics_frame = ttk.LabelFrame(model_right_frame, text="Performance Metrics")
        metrics_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create text widget for metrics
        self.metrics_text = tk.Text(metrics_frame, wrap=tk.WORD, height=15, width=40)
        self.metrics_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
    def setup_prediction_tab(self):
        """Set up the prediction tab"""
        prediction_frame = ttk.Frame(self.prediction_tab)
        prediction_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Title
        ttk.Label(prediction_frame, text="Driver Behavior Prediction", 
                  style='Header.TLabel').pack(anchor=tk.W, pady=(0, 10))
        
        # Control section
        control_frame = ttk.Frame(prediction_frame)
        control_frame.pack(fill=tk.X, pady=10)
        
        ttk.Button(control_frame, text="Run Prediction", 
                   command=self.run_prediction).pack(side=tk.LEFT, padx=5)
        
        # Prediction result section
        result_frame = ttk.LabelFrame(prediction_frame, text="Prediction Results")
        result_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Central display for prediction
        prediction_display = ttk.Frame(result_frame)
        prediction_display.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Create a large central indicator
        self.prediction_canvas = tk.Canvas(prediction_display, width=200, height=200, bg="#f5f5f5")
        self.prediction_canvas.pack(pady=20)
        
        self.prediction_circle = self.prediction_canvas.create_oval(10, 10, 190, 190, 
                                                                  fill="#9E9E9E", width=2)
        self.prediction_text = self.prediction_canvas.create_text(100, 100, 
                                                               text="No Prediction", fill="white",
                                                               font=("Helvetica", 14, "bold"))
        
        # Additional details
        self.prediction_detail = ttk.Label(prediction_display, 
                                        text="Run a prediction to classify driver behavior",
                                        font=("Helvetica", 12))
        self.prediction_detail.pack(pady=10)
        
        # Confidence bar
        confidence_frame = ttk.Frame(prediction_display)
        confidence_frame.pack(fill=tk.X, pady=10)
        
        ttk.Label(confidence_frame, text="Confidence:").pack(side=tk.LEFT)
        self.confidence_bar = ttk.Progressbar(confidence_frame, length=300, mode='determinate')
        self.confidence_bar.pack(side=tk.LEFT, padx=10, fill=tk.X, expand=True)
        self.confidence_value = ttk.Label(confidence_frame, text="0%")
        self.confidence_value.pack(side=tk.LEFT, padx=5)
        
    def load_dataset(self):
        """Load a dataset from a file"""
        filename = filedialog.askopenfilename(
            title="Select Dataset",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if filename:
            if self.analyzer.load_data(filename):
                # Update UI to reflect loaded dataset
                self.dataset_filename = filename
                base_filename = os.path.basename(filename)
                self.dataset_label.config(text=base_filename)
                self.update_status(f"Dataset loaded: {base_filename}", 'success')
                
                # Update system status
                self.update_system_status("ready")
                
                # Enable analysis buttons
                self.enable_analysis_buttons()
            else:
                self.update_status("Failed to load dataset", 'error')
    
    def analyze_data(self):
        """Run data analysis on the loaded dataset"""
        if self.analyzer.data is None:
            messagebox.showwarning("No Dataset", "Please load a dataset first.")
            return
            
        self.update_status("Analyzing data...", 'info')
        self.update_system_status("working")
        
        # Prepare data
        if not self.analyzer.prepare_data():
            self.update_status("Failed to prepare data for analysis", 'error')
            self.update_system_status("error")
            return
            
        # Get data summary
        summary = self.analyzer.get_data_summary()
        
        if summary:
            # Update statistics display
            self.update_stats_display(summary)
            
            # Update raw data visualization
            self.update_raw_data_plots()
            
            # Update feature visualization
            self.update_feature_plots()
            
            self.update_status("Data analysis complete", 'success')
            self.notebook.select(self.data_analysis_tab)
        else:
            self.update_status("Failed to analyze data", 'error')
            self.update_system_status("error")
    
    def train_model(self):
        """Train a quantum model on the dataset"""
        if self.analyzer.features is None or self.analyzer.labels is None:
            messagebox.showwarning("No Prepared Data", "Please load and analyze a dataset first.")
            return
            
        self.update_status("Training quantum model... This may take a while.", 'info')
        self.update_system_status("working")
        
        # Run in a thread to avoid freezing the UI
        threading.Thread(target=self._train_model_thread, daemon=True).start()
    
    def _train_model_thread(self):
        """Background thread for model training"""
        try:
            success = self.analyzer.train_model()
            
            # Update UI from main thread
            self.root.after(0, self._update_after_training, success)
        except Exception as e:
            print(f"Error during training: {e}")
            self.root.after(0, self._update_after_training, False)
    
    def _update_after_training(self, success):
        """Update UI after model training completes"""
        if success:
            # Update performance metrics
            self.update_performance_metrics()
            
            # Update confusion matrix
            self.update_confusion_matrix()
            
            self.update_status("Model training complete", 'success')
            self.notebook.select(self.quantum_model_tab)
        else:
            self.update_status("Failed to train model", 'error')
            self.update_system_status("error")
    
    def run_prediction(self):
        """Run prediction on sample data"""
        if self.analyzer.model is None:
            messagebox.showwarning("No Model", "Please train a model first.")
            return
            
        self.update_status("Running prediction...", 'info')
        
        # Simulate prediction on a random sample
        if self.analyzer.features is not None and len(self.analyzer.features) > 0:
            # Get a random sample
            idx = random.randint(0, len(self.analyzer.features) - 1)
            sample = self.analyzer.features[idx:idx+1]
            
            # Scale the sample
            sample_scaled = self.analyzer.scaler.transform(sample)
            
            # Make prediction
            try:
                prediction = self.analyzer.model.predict(sample_scaled)[0]
                confidence = random.uniform(0.7, 0.98)  # Simulate confidence
                
                # Update prediction display
                self.update_prediction_display(prediction, confidence)
                
                self.update_status("Prediction complete", 'success')
                self.notebook.select(self.prediction_tab)
            except Exception as e:
                print(f"Error during prediction: {e}")
                self.update_status("Prediction failed", 'error')
        else:
            self.update_status("No data available for prediction", 'error')
    
    def connect_hardware(self):
        """Connect to hardware (simulated)"""
        if self.hardware.connect():
            # Start a checker to update status
            threading.Thread(target=self._check_hardware_status, daemon=True).start()
            self.update_status("Connecting to hardware...", 'info')
            self.connect_btn.config(state=tk.DISABLED)
    
    def _check_hardware_status(self):
        """Check hardware connection status"""
        # Wait for connection to complete
        time.sleep(2.5)
        
        # Update UI from main thread
        self.root.after(0, self._update_hardware_status)
    
    def _update_hardware_status(self):
        """Update hardware connection status in UI"""
        status = self.hardware.status
        self.hw_status_label.config(text=status)
        
        if self.hardware.connected:
            self.connect_btn.config(state=tk.DISABLED)
            self.disconnect_btn.config(state=tk.NORMAL)
            self.update_status("Hardware connected successfully", 'success')
        else:
            self.connect_btn.config(state=tk.NORMAL)
            self.disconnect_btn.config(state=tk.DISABLED)
            if status == "Connection failed":
                self.update_status("Hardware connection failed", 'error')
    
    def disconnect_hardware(self):
        """Disconnect from hardware (simulated)"""
        if self.hardware.disconnect():
            self.hw_status_label.config(text="Disconnected")
            self.connect_btn.config(state=tk.NORMAL)
            self.disconnect_btn.config(state=tk.DISABLED)
            self.update_status("Hardware disconnected", 'info')
    
    def show_about(self):
        """Show about dialog"""
        messagebox.showinfo(
            "About Quantum Driver Analysis",
            "Quantum Driver Behavior Analysis\n\n"
            "Version 1.0\n\n"
            "This application uses quantum machine learning to analyze and classify driver behavior "
            "from accelerometer data. The quantum SVM model leverages quantum feature mapping and "
            "error detection to improve classification accuracy."
        )
    
    def update_status(self, message, status_type='info'):
        """Update status bar with message"""
        self.status_var.set(message)
        
        # Apply appropriate style based on status type
        if status_type == 'error':
            self.status_bar.config(foreground='red')
        elif status_type == 'success':
            self.status_bar.config(foreground='green')
        else:
            self.status_bar.config(foreground='black')
    
    def update_system_status(self, status):
        """Update system status visualization"""
        status_colors = {
            "ready": "#4CAF50",    # Green
            "working": "#2196F3",  # Blue
            "error": "#F44336",    # Red
            "idle": "#9E9E9E"      # Gray
        }
        
        status_text = {
            "ready": "System\nReady",
            "working": "System\nWorking",
            "error": "System\nError",
            "idle": "System\nIdle"
        }
        
        # Update circle color and text
        try:
            self.system_status_canvas.itemconfig(
                self.status_circle, fill=status_colors.get(status, status_colors["idle"]))
            self.system_status_canvas.itemconfig(
                self.system_status_canvas.find_withtag("text"), text=status_text.get(status, "System\nIdle"))
        except:
            # If canvas doesn't exist or can't be updated
            pass
    
    def update_stats_display(self, summary):
        """Update statistics display with data summary"""
        self.stats_text.delete('1.0', tk.END)
        
        self.stats_text.insert(tk.END, "Dataset Summary\n", "heading")
        self.stats_text.insert(tk.END, "==============\n\n")
        
        self.stats_text.insert(tk.END, f"Number of samples: {summary['n_samples']}\n")
        self.stats_text.insert(tk.END, f"Number of features: {summary['n_features']}\n\n")
        
        if summary['class_distribution']:
            self.stats_text.insert(tk.END, "Class Distribution:\n")
            for cls, count in summary['class_distribution'].items():
                label = "Smooth Turn" if cls == 0 else "Harsh Turn"
                self.stats_text.insert(tk.END, f"  - {label}: {count} samples\n")
            self.stats_text.insert(tk.END, "\n")
        
        self.stats_text.insert(tk.END, "Feature Statistics:\n")
        for feature, stats in summary['feature_stats'].items():
            if feature in ['ax', 'ay', 'az', 'gx', 'gy', 'gz']:
                self.stats_text.insert(tk.END, f"  {feature}:\n")
                self.stats_text.insert(tk.END, f"    Mean: {stats['mean']:.4f}\n")
                self.stats_text.insert(tk.END, f"    Std: {stats['std']:.4f}\n")
                self.stats_text.insert(tk.END, f"    Min: {stats['min']:.4f}\n")
                self.stats_text.insert(tk.END, f"    Max: {stats['max']:.4f}\n\n")
        
        # Apply tags for formatting
        self.stats_text.tag_configure("heading", font=("Helvetica", 12, "bold"))
    
    def update_raw_data_plots(self):
        """Update raw data visualization"""
        if self.analyzer.data is None:
            return
            
        # Clear previous plots
        self.raw_data_fig.clear()
        
        # Create subplots for accelerometer and gyroscope
        ax1 = self.raw_data_fig.add_subplot(211)
        ax2 = self.raw_data_fig.add_subplot(212)
        
        # Get a subset of data to plot (max 1000 points)
        data = self.analyzer.data
        subset = data.iloc[:min(1000, len(data))]
        
        # Plot accelerometer data
        if 'ax' in data.columns and 'ay' in data.columns and 'az' in data.columns:
            ax1.plot(subset.index, subset['ax'], label='X')
            ax1.plot(subset.index, subset['ay'], label='Y')
            ax1.plot(subset.index, subset['az'], label='Z')
            ax1.set_title('Accelerometer Data')
            ax1.set_ylabel('Acceleration (m/s²)')
            ax1.legend()
            ax1.grid(True)
        
        # Plot gyroscope data
        if 'gx' in data.columns and 'gy' in data.columns and 'gz' in data.columns:
            ax2.plot(subset.index, subset['gx'], label='X')
            ax2.plot(subset.index, subset['gy'], label='Y')
            ax2.plot(subset.index, subset['gz'], label='Z')
            ax2.set_title('Gyroscope Data')
            ax2.set_xlabel('Sample')
            ax2.set_ylabel('Angular velocity (rad/s)')
            ax2.legend()
            ax2.grid(True)
        
        self.raw_data_fig.tight_layout()
        self.raw_data_canvas.draw()
    
    def update_feature_plots(self):
        """Update feature visualization"""
        if self.analyzer.features is None:
            return
            
        # Clear previous plots
        self.feature_fig.clear()
        
        # Create plot for feature visualization
        if 'label' in self.analyzer.data.columns:
            # Create scatter plot of first two principal components
            try:
                from sklearn.decomposition import PCA
                
                # Apply PCA to reduce dimensionality for visualization
                pca = PCA(n_components=2)
                features_reduced = pca.fit_transform(self.analyzer.features)
                # Create plot
                ax = self.feature_fig.add_subplot(111)
                # Get unique labels
                unique_labels = np.unique(self.analyzer.labels)
                colors = ['blue', 'red']
                # Plot each class with a different color
                for i, label in enumerate(unique_labels):
                    mask = self.analyzer.labels == label
                    label_name = "Smooth Turn" if label == 0 else "Harsh Turn"
                    ax.scatter(features_reduced[mask, 0], features_reduced[mask, 1], 
                              c=colors[i % len(colors)], label=label_name, alpha=0.6)
                ax.set_title('Feature Visualization (PCA)')
                ax.set_xlabel('Principal Component 1')
                ax.set_ylabel('Principal Component 2')
                ax.legend()
                ax.grid(True)
            except Exception as e:
                print(f"Error creating PCA plot: {e}")
                ax = self.feature_fig.add_subplot(111)
                ax.text(0.5, 0.5, "Feature visualization failed", 
                       horizontalalignment='center', verticalalignment='center')
        else:
            # No labels, create histogram of features
            ax = self.feature_fig.add_subplot(111)
            # Get a subset of features
            subset = self.analyzer.features[:, :6]  # First 6 features
            labels = ['Mean X', 'Mean Y', 'Mean Z', 'Mean GX', 'Mean GY', 'Mean GZ']
            ax.boxplot(subset)
            ax.set_title('Feature Distribution')
            ax.set_xticklabels(labels, rotation=45)
            ax.set_ylabel('Value')
            ax.grid(True)
        self.feature_fig.tight_layout()
        self.feature_canvas.draw()
    def update_performance_metrics(self):
        """Update performance metrics display"""
        if self.analyzer.training_results is None:
            return
        results = self.analyzer.training_results
        self.metrics_text.delete('1.0', tk.END)
        self.metrics_text.insert(tk.END, "Model Performance Metrics\n", "heading")
        self.metrics_text.insert(tk.END, "=======================\n\n")
        # Accuracy
        self.metrics_text.insert(tk.END, f"Accuracy: {results['accuracy']:.4f}\n\n")
        # Class report
        self.metrics_text.insert(tk.END, "Classification Report:\n")
        report = results['report']
        for cls in sorted(report.keys()):
            if cls in ['0', '1']:
                cls_label = "Smooth Turn" if cls == '0' else "Harsh Turn"
                self.metrics_text.insert(tk.END, f"\n{cls_label}:\n")
                metrics = report[cls]
                self.metrics_text.insert(tk.END, f"  Precision: {metrics['precision']:.4f}\n")
                self.metrics_text.insert(tk.END, f"  Recall: {metrics['recall']:.4f}\n")
                self.metrics_text.insert(tk.END, f"  F1-Score: {metrics['f1-score']:.4f}\n")
                self.metrics_text.insert(tk.END, f"  Samples: {metrics['support']}\n")
        # Overall metrics
        if 'macro avg' in report:
            self.metrics_text.insert(tk.END, "\nMacro Average:\n")
            metrics = report['macro avg']
            self.metrics_text.insert(tk.END, f"  Precision: {metrics['precision']:.4f}\n")
            self.metrics_text.insert(tk.END, f"  Recall: {metrics['recall']:.4f}\n")
            self.metrics_text.insert(tk.END, f"  F1-Score: {metrics['f1-score']:.4f}\n")
        # Apply tags for formatting
        self.metrics_text.tag_configure("heading", font=("Helvetica", 12, "bold"))
    def update_confusion_matrix(self):
        """Update confusion matrix visualization"""
        if self.analyzer.training_results is None:
            return
        # Clear previous plot
        self.cm_fig.clear()
        # Get confusion matrix
        cm = self.analyzer.training_results['confusion_matrix']
        # Create heatmap
        ax = self.cm_fig.add_subplot(111)
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        self.cm_fig.colorbar(im)
        # Set labels
        classes = ["Smooth Turn", "Harsh Turn"]
        tick_marks = np.arange(len(classes))
        ax.set_xticks(tick_marks)
        ax.set_yticks(tick_marks)
        ax.set_xticklabels(classes)
        ax.set_yticklabels(classes)  
        # Add text annotations
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], 'd'),
                        horizontalalignment="center",
                        color="white" if cm[i, j] > thresh else "black")
        ax.set_ylabel('True label')
        ax.set_xlabel('Predicted label')
        ax.set_title('Confusion Matrix')
        self.cm_fig.tight_layout()
        self.cm_canvas.draw()
    def update_prediction_display(self, prediction, confidence):
        """Update prediction display with results"""
        prediction_colors = {
            0: "#4CAF50",  # Green for Smooth Turn
            1: "#F44336"   # Red for Harsh Turn
        }
        prediction_labels = {
            0: "Smooth Turn",
            1: "Harsh Turn"
        }
        # Update prediction circle
        self.prediction_canvas.itemconfig(
            self.prediction_circle, fill=prediction_colors.get(prediction, "#9E9E9E"))
        # Update prediction text
        self.prediction_canvas.itemconfig(
            self.prediction_text, text=prediction_labels.get(prediction, "Unknown"))
        # Update prediction detail
        self.prediction_detail.config(
            text=f"Driver is performing a {prediction_labels.get(prediction, 'Unknown').lower()}")
        # Update confidence bar
        confidence_pct = int(confidence * 100)
        self.confidence_bar['value'] = confidence_pct
        self.confidence_value.config(text=f"{confidence_pct}%")
    def enable_analysis_buttons(self):
        """Enable analysis buttons after dataset is loaded"""
        pass  # Buttons are always enabled in this version
def main():
    root = tk.Tk()
    app = QuantumDriverApp(root)
    root.mainloop()
if __name__ == "__main__":
    main()