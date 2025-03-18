import tkinter as tk
from tkinter import ttk, colorchooser, filedialog, messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from black_widow import BlackWidowOptimizer
from benchmark_functions import powell_sum, cigar, discus, rosenbrock, ackley

class BlackWidowGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        
        # Configure the main window
        self.title("Black Widow Optimization Algorithm")
        self.geometry("1200x800")
        self.minsize(800, 600)
        
        # Variables
        self.functions = {
            "Powell Sum": {"func": powell_sum, "bounds": (-5.12, 5.12), "dimensions": 2},
            "Cigar": {"func": cigar, "bounds": (-5.12, 5.12), "dimensions": 2},
            "Discus": {"func": discus, "bounds": (-5.12, 5.12), "dimensions": 2},
            "Rosenbrock": {"func": rosenbrock, "bounds": (-30, 30), "dimensions": 2},
            "Ackley": {"func": ackley, "bounds": (-35, 35), "dimensions": 2}
        }
        self.selected_function = tk.StringVar(value=list(self.functions.keys())[0])
        self.dimensions = tk.IntVar(value=2)
        self.population_size = tk.IntVar(value=20)
        self.max_iterations = tk.IntVar(value=30)
        self.reproduction_rate = tk.DoubleVar(value=0.6)
        self.cannibalism_rate = tk.DoubleVar(value=0.4)
        self.mutation_rate = tk.DoubleVar(value=0.4)
        self.minimize = tk.BooleanVar(value=True)
        self.show_trails = tk.BooleanVar(value=True)
        
        # Algorithm state
        self.optimizer = None
        self.current_iteration = 0
        self.population = None
        self.best_solution = None
        self.best_fitness = None
        self.previous_population = None
        self.running = False
        
        # Variable sliders and checkboxes
        self.sliders = []
        self.textboxes = []
        self.checkboxes = []
        self.checkbox_vars = []
        self.selected_dimensions = []
        
        # Create the GUI components
        self.create_menu()
        self.create_toolbar()
        self.create_main_frame()
        self.create_status_bar()
        
        # Initialize the plot
        self.update_dimensions()
        self.update_function_plot()
        
    def create_menu(self):
        """Create the main menu bar"""
        self.menu_bar = tk.Menu(self)
        
        # File menu
        file_menu = tk.Menu(self.menu_bar, tearoff=0)
        file_menu.add_command(label="Save Plot", command=self.save_plot)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.quit)
        self.menu_bar.add_cascade(label="File", menu=file_menu)
        
        # Edit menu
        edit_menu = tk.Menu(self.menu_bar, tearoff=0)
        edit_menu.add_command(label="Reset", command=self.reset)
        self.menu_bar.add_cascade(label="Edit", menu=edit_menu)
        
        # View menu
        view_menu = tk.Menu(self.menu_bar, tearoff=0)
        view_menu.add_checkbutton(label="Show Trails", variable=self.show_trails, 
                                 command=self.update_function_plot)
        self.menu_bar.add_cascade(label="View", menu=view_menu)
        
        # Help menu
        help_menu = tk.Menu(self.menu_bar, tearoff=0)
        help_menu.add_command(label="About", command=self.show_about)
        self.menu_bar.add_cascade(label="Help", menu=help_menu)
        
        self.config(menu=self.menu_bar)
    
    def create_toolbar(self):
        """Create the toolbar with common actions"""
        self.toolbar_frame = ttk.Frame(self)
        self.toolbar_frame.pack(side=tk.TOP, fill=tk.X)
        
        # Function selection
        ttk.Label(self.toolbar_frame, text="Function:").pack(side=tk.LEFT, padx=5, pady=5)
        function_combo = ttk.Combobox(self.toolbar_frame, textvariable=self.selected_function,
                                     values=list(self.functions.keys()), state="readonly")
        function_combo.pack(side=tk.LEFT, padx=5, pady=5)
        function_combo.bind("<<ComboboxSelected>>", lambda e: self.update_function())
        
        # Dimensions
        ttk.Label(self.toolbar_frame, text="Dimensions:").pack(side=tk.LEFT, padx=5, pady=5)
        dimensions_spinbox = ttk.Spinbox(self.toolbar_frame, from_=2, to=10, 
                                        textvariable=self.dimensions, width=5,
                                        command=self.update_dimensions)
        dimensions_spinbox.pack(side=tk.LEFT, padx=5, pady=5)
        dimensions_spinbox.bind("<Return>", lambda e: self.update_dimensions())
        # Add a button to apply dimension changes
        apply_dim_button = ttk.Button(self.toolbar_frame, text="Apply", 
                                     command=self.update_dimensions)
        apply_dim_button.pack(side=tk.LEFT, padx=5, pady=5)
        
        # Control buttons
        self.start_button = ttk.Button(self.toolbar_frame, text="Start", command=self.start_optimization)
        self.start_button.pack(side=tk.RIGHT, padx=5, pady=5)
        
        self.step_button = ttk.Button(self.toolbar_frame, text="Step", command=self.step_optimization, state=tk.DISABLED)
        self.step_button.pack(side=tk.RIGHT, padx=5, pady=5)
        
        self.stop_button = ttk.Button(self.toolbar_frame, text="Stop", command=self.stop_optimization, state=tk.DISABLED)
        self.stop_button.pack(side=tk.RIGHT, padx=5, pady=5)
    
    def create_main_frame(self):
        """Create the main frame with the plot and controls"""
        self.main_frame = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Left panel for plot
        self.plot_frame = ttk.Frame(self.main_frame)
        self.main_frame.add(self.plot_frame, weight=3)
        
        # Create matplotlib figure and canvas
        self.fig = Figure(figsize=(6, 6), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Add matplotlib toolbar
        self.mpl_toolbar = NavigationToolbar2Tk(self.canvas, self.plot_frame)
        self.mpl_toolbar.update()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Right panel for controls
        self.control_frame = ttk.Frame(self.main_frame)
        self.main_frame.add(self.control_frame, weight=1)
        
        # Variable controls
        self.var_frame = ttk.LabelFrame(self.control_frame, text="Variables")
        self.var_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Algorithm parameters
        self.param_frame = ttk.LabelFrame(self.control_frame, text="Algorithm Parameters")
        self.param_frame.pack(fill=tk.BOTH, expand=False, padx=5, pady=5)
        
        # Population size
        ttk.Label(self.param_frame, text="Population Size:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        ttk.Spinbox(self.param_frame, from_=10, to=100, textvariable=self.population_size, width=5).grid(
            row=0, column=1, sticky=tk.W, padx=5, pady=2)
        
        # Max iterations
        ttk.Label(self.param_frame, text="Max Iterations:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        ttk.Spinbox(self.param_frame, from_=10, to=1000, textvariable=self.max_iterations, width=5).grid(
            row=1, column=1, sticky=tk.W, padx=5, pady=2)
        
        # Reproduction rate
        ttk.Label(self.param_frame, text="Reproduction Rate:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=2)
        ttk.Spinbox(self.param_frame, from_=0.1, to=0.9, increment=0.1, textvariable=self.reproduction_rate, width=5).grid(
            row=2, column=1, sticky=tk.W, padx=5, pady=2)
        
        # Cannibalism rate
        ttk.Label(self.param_frame, text="Cannibalism Rate:").grid(row=3, column=0, sticky=tk.W, padx=5, pady=2)
        ttk.Spinbox(self.param_frame, from_=0.1, to=0.9, increment=0.1, textvariable=self.cannibalism_rate, width=5).grid(
            row=3, column=1, sticky=tk.W, padx=5, pady=2)
        
        # Mutation rate
        ttk.Label(self.param_frame, text="Mutation Rate:").grid(row=4, column=0, sticky=tk.W, padx=5, pady=2)
        ttk.Spinbox(self.param_frame, from_=0.1, to=0.9, increment=0.1, textvariable=self.mutation_rate, width=5).grid(
            row=4, column=1, sticky=tk.W, padx=5, pady=2)
        
        # Minimize/Maximize
        ttk.Label(self.param_frame, text="Optimization:").grid(row=5, column=0, sticky=tk.W, padx=5, pady=2)
        ttk.Radiobutton(self.param_frame, text="Minimize", variable=self.minimize, value=True).grid(
            row=5, column=1, sticky=tk.W, padx=5, pady=2)
        ttk.Radiobutton(self.param_frame, text="Maximize", variable=self.minimize, value=False).grid(
            row=6, column=1, sticky=tk.W, padx=5, pady=2)
        
        # Results frame
        self.results_frame = ttk.LabelFrame(self.control_frame, text="Results")
        self.results_frame.pack(fill=tk.BOTH, expand=False, padx=5, pady=5)
        
        self.iteration_label = ttk.Label(self.results_frame, text="Iteration: 0")
        self.iteration_label.pack(anchor=tk.W, padx=5, pady=2)
        
        self.best_solution_label = ttk.Label(self.results_frame, text="Best Solution: None")
        self.best_solution_label.pack(anchor=tk.W, padx=5, pady=2)
        
        self.best_fitness_label = ttk.Label(self.results_frame, text="Best Fitness: None")
        self.best_fitness_label.pack(anchor=tk.W, padx=5, pady=2)
    
    def create_status_bar(self):
        """Create the status bar at the bottom of the window"""
        self.status_bar = ttk.Label(self, text="Ready", relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def update_dimensions(self):
        """Update the variable sliders based on the selected dimensions"""
        # Clear existing sliders
        for widget in self.var_frame.winfo_children():
            widget.destroy()
        
        self.sliders = []
        self.textboxes = []
        self.checkboxes = []
        self.checkbox_vars = []
        self.selected_dimensions = []
        
        # Get bounds for the selected function
        func_info = self.functions[self.selected_function.get()]
        lower_bound, upper_bound = func_info["bounds"]
        
        # Create sliders for each dimension
        for i in range(self.dimensions.get()):
            # Frame for this dimension
            dim_frame = ttk.Frame(self.var_frame)
            dim_frame.pack(fill=tk.X, padx=5, pady=2)
            
            # Checkbox for selecting this dimension for visualization
            var = tk.BooleanVar(value=False)
            self.checkbox_vars.append(var)
            checkbox = ttk.Checkbutton(dim_frame, variable=var, command=self.update_selected_dimensions)
            checkbox.grid(row=0, column=0, padx=2)
            self.checkboxes.append(checkbox)
            
            # Label
            ttk.Label(dim_frame, text=f"x{i+1}:").grid(row=0, column=1, padx=2)
            
            # Slider
            slider = ttk.Scale(dim_frame, from_=lower_bound, to=upper_bound, orient=tk.HORIZONTAL)
            slider.set(0)  # Default value
            slider.grid(row=0, column=2, padx=2, sticky=tk.EW)
            self.sliders.append(slider)
            
            # Text box
            textbox = ttk.Entry(dim_frame, width=8)
            textbox.insert(0, "0.0")
            textbox.grid(row=0, column=3, padx=2)
            self.textboxes.append(textbox)
            
            # Configure slider to update textbox
            slider.configure(command=lambda val, i=i: self.update_textbox(i, val))
            
            # Configure textbox to update slider
            textbox.bind("<Return>", lambda e, i=i: self.update_slider(i))
            
            # Make the slider expand with the window
            dim_frame.columnconfigure(2, weight=1)
        
        # Select the first two dimensions by default
        if len(self.checkbox_vars) >= 2:
            self.checkbox_vars[0].set(True)
            self.checkbox_vars[1].set(True)
            self.update_selected_dimensions()
        
        # Update the function plot
        self.update_function_plot()
    
    def update_textbox(self, index, value):
        """Update the textbox when the slider is moved"""
        self.textboxes[index].delete(0, tk.END)
        self.textboxes[index].insert(0, f"{float(value):.2f}")
        self.update_function_plot()
    
    def update_slider(self, index):
        """Update the slider when the textbox is edited"""
        try:
            value = float(self.textboxes[index].get())
            self.sliders[index].set(value)
            self.update_function_plot()
        except ValueError:
            # Reset to the current slider value if invalid input
            self.update_textbox(index, self.sliders[index].get())
    
    def update_selected_dimensions(self):
        """Update which dimensions are selected for visualization"""
        self.selected_dimensions = []
        for i, var in enumerate(self.checkbox_vars):
            if var.get():
                self.selected_dimensions.append(i)
        
        # Ensure only 2 dimensions are selected
        if len(self.selected_dimensions) > 2:
            # Uncheck the oldest selection
            oldest = self.selected_dimensions[0]
            self.checkbox_vars[oldest].set(False)
            self.selected_dimensions.pop(0)
        
        # Enable/disable sliders based on selection
        for i, slider in enumerate(self.sliders):
            if i in self.selected_dimensions:
                slider.state(['disabled'])
                self.textboxes[i].state(['disabled'])
            else:
                slider.state(['!disabled'])
                self.textboxes[i].state(['!disabled'])
        
        self.update_function_plot()
    
    def update_function(self):
        """Update when a new function is selected"""
        # Update dimensions to match the function's default
        func_info = self.functions[self.selected_function.get()]
        self.dimensions.set(func_info["dimensions"])
        self.update_dimensions()
    
    def update_function_plot(self):
        """Update the 2D function plot based on selected dimensions and slider values"""
        # Clear the figure and recreate the subplot to prevent layout issues
        self.fig.clear()
        self.ax = self.fig.add_subplot(111)
        
        if len(self.selected_dimensions) != 2:
            # Need exactly 2 dimensions selected
            self.ax.text(0.5, 0.5, "Select exactly 2 dimensions to visualize", 
                        ha='center', va='center', transform=self.ax.transAxes)
            self.canvas.draw()
            return
        
        # Get the selected function
        func_name = self.selected_function.get()
        func_info = self.functions[func_name]
        func = func_info["func"]
        lower_bound, upper_bound = func_info["bounds"]
        
        # Get the selected dimensions
        dim1, dim2 = self.selected_dimensions
        
        # Create a grid of points for the selected dimensions
        x = np.linspace(lower_bound, upper_bound, 100)
        y = np.linspace(lower_bound, upper_bound, 100)
        X, Y = np.meshgrid(x, y)
        
        # Create input points for the function
        Z = np.zeros_like(X)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                # Create the full input vector
                point = []
                for k in range(self.dimensions.get()):
                    if k == dim1:
                        point.append(X[i, j])
                    elif k == dim2:
                        point.append(Y[i, j])
                    else:
                        # Use the slider value for non-selected dimensions
                        # Make sure the slider exists
                        if k < len(self.sliders):
                            point.append(float(self.sliders[k].get()))
                        else:
                            # Use 0.0 as default if slider doesn't exist
                            point.append(0.0)
                
                # Convert point to numpy array before passing to function
                Z[i, j] = func(np.array(point))
        
        # Plot the function as a filled contour
        contour = self.ax.contourf(X, Y, Z, 50, cmap='viridis')
        
        # Add a new colorbar
        self.colorbar = self.fig.colorbar(contour, ax=self.ax)
        
        # Add contour lines
        self.ax.contour(X, Y, Z, 10, colors='k', alpha=0.3)
        
        # Set labels
        self.ax.set_xlabel(f'x{dim1+1}')
        self.ax.set_ylabel(f'x{dim2+1}')
        self.ax.set_title(f'{func_name} Function')
        
        # Plot the population and best solution if available
        if self.population is not None:
            # Check if the selected dimensions are valid for the current population
            if dim1 < self.population.shape[1] and dim2 < self.population.shape[1]:
                # Extract the coordinates for the selected dimensions
                x_coords = self.population[:, dim1]
                y_coords = self.population[:, dim2]
                
                # Plot the population
                self.ax.scatter(x_coords, y_coords, color='white', edgecolor='black', 
                               s=50, label='Population')
                
                # Add agent numbers
                for i, (x, y) in enumerate(zip(x_coords, y_coords)):
                    self.ax.annotate(str(i), (x, y), xytext=(5, 5), textcoords='offset points')
                
                # Plot trails if enabled
                if self.show_trails.get() and self.previous_population is not None:
                    if dim1 < self.previous_population.shape[1] and dim2 < self.previous_population.shape[1]:
                        prev_x_coords = self.previous_population[:, dim1]
                        prev_y_coords = self.previous_population[:, dim2]
                        
                        for i in range(len(x_coords)):
                            self.ax.plot([prev_x_coords[i], x_coords[i]], [prev_y_coords[i], y_coords[i]], 
                                        'k--', alpha=0.5)
            else:
                # If dimensions are invalid, reset the population display
                self.population = None
                self.previous_population = None
                self.best_solution = None
                self.best_fitness = None
                self.status_bar.config(text="Reset due to dimension change")
        
        # Plot the best solution if available
        if self.best_solution is not None:
            self.ax.scatter(self.best_solution[dim1], self.best_solution[dim2], 
                           color='red', marker='*', s=200, label='Best Solution')
        
        # Add legend if we have elements to show in it
        if self.population is not None or self.best_solution is not None:
            self.ax.legend()
        
        # Adjust the layout to ensure proper spacing
        self.fig.tight_layout()
        
        # Update the canvas
        self.canvas.draw()
    
    def start_optimization(self):
        """Initialize and start the optimization process"""
        # Get the selected function
        func_name = self.selected_function.get()
        func_info = self.functions[func_name]
        func = func_info["func"]
        lower_bound, upper_bound = func_info["bounds"]
        
        # Create bounds for all dimensions
        bounds = [(lower_bound, upper_bound) for _ in range(self.dimensions.get())]
        
        # Create the optimizer
        self.optimizer = BlackWidowOptimizer(
            objective_function=func,
            dimensions=self.dimensions.get(),
            bounds=bounds,
            population_size=self.population_size.get(),
            max_iterations=self.max_iterations.get(),
            reproduction_rate=self.reproduction_rate.get(),
            cannibalism_rate=self.cannibalism_rate.get(),
            mutation_rate=self.mutation_rate.get(),
            minimize=self.minimize.get()
        )
        
        # Initialize the optimization
        self.optimizer.initialize_population()
        self.population = self.optimizer.population.copy()
        self.previous_population = None
        self.current_iteration = 0
        self.best_solution = None
        self.best_fitness = None
        self.running = True
        
        # Update the UI
        self.update_function_plot()
        self.update_results()
        self.start_button.state(['disabled'])
        self.step_button.state(['!disabled'])
        self.stop_button.state(['!disabled'])
        self.status_bar.config(text=f"Optimization started. Iteration: {self.current_iteration}")
    
    def step_optimization(self):
        """Perform one step of the optimization"""
        if not self.running or self.current_iteration >= self.max_iterations.get():
            self.stop_optimization()
            return
        
        # Save the current population for trails
        self.previous_population = self.population.copy()
        
        # Perform one iteration
        self.optimizer.iterate_once()
        self.population = self.optimizer.population.copy()
        self.current_iteration += 1
        
        # Update best solution
        self.best_solution = self.optimizer.best_solution
        self.best_fitness = self.optimizer.best_fitness
        
        # Update the UI
        self.update_function_plot()
        self.update_results()
        self.status_bar.config(text=f"Optimization in progress. Iteration: {self.current_iteration}")
        
        # Check if we've reached the maximum iterations
        if self.current_iteration >= self.max_iterations.get():
            self.stop_optimization()
    
    def stop_optimization(self):
        """Stop the optimization process"""
        self.running = False
        self.start_button.state(['!disabled'])
        self.step_button.state(['disabled'])
        self.stop_button.state(['disabled'])
        self.status_bar.config(text=f"Optimization stopped. Iteration: {self.current_iteration}")
    
    def update_results(self):
        """Update the results display"""
        self.iteration_label.config(text=f"Iteration: {self.current_iteration}")
        
        if self.best_solution is not None:
            solution_text = "Best Solution: ["
            for i, val in enumerate(self.best_solution):
                solution_text += f"{val:.4f}"
                if i < len(self.best_solution) - 1:
                    solution_text += ", "
            solution_text += "]"
            self.best_solution_label.config(text=solution_text)
            
            self.best_fitness_label.config(text=f"Best Fitness: {self.best_fitness:.6f}")
    
    def reset(self):
        """Reset the optimization"""
        self.stop_optimization()
        self.population = None
        self.previous_population = None
        self.best_solution = None
        self.best_fitness = None
        self.current_iteration = 0
        self.update_function_plot()
        self.update_results()
        self.status_bar.config(text="Ready")
    
    def save_plot(self):
        """Save the current plot to a file"""
        file_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("All files", "*.*")]
        )
        if file_path:
            self.fig.savefig(file_path, dpi=300, bbox_inches='tight')
            self.status_bar.config(text=f"Plot saved to {file_path}")
    
    def show_about(self):
        """Show the about dialog"""
        messagebox.showinfo(
            "About",
            "Black Widow Optimization Algorithm GUI\n\n"
            "A visualization tool for the Black Widow Optimization Algorithm.\n\n"
            "Created for educational purposes."
        )

if __name__ == "__main__":
    app = BlackWidowGUI()
    app.mainloop()
