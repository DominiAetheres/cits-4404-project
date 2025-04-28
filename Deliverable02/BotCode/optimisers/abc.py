import numpy as np

class ArtificialBeeColony:
    def __init__(self, objective_function, lb, ub, colony_size=50, max_iterations=100, limit=20):
        """
        Initializes the Artificial Bee Colony algorithm.
        
        Parameters:
        objective_function (callable): The function to minimize
        lb (float or array): Lower bounds of search space
        ub (float or array): Upper bounds of search space
        colony_size (int): Number of food sources (half of the colony size)
        max_iterations (int): Maximum number of iterations
        limit (int): Limit of trials after which a food source is abandoned
        """
        self.objective_function = objective_function
        self.lb = np.array(lb)
        self.ub = np.array(ub)
        self.dim = len(self.lb) if hasattr(self.lb, "__len__") else 1
        self.colony_size = colony_size
        self.food_sources = colony_size // 2  # Number of food sources (half of colony size)
        self.max_iterations = max_iterations
        self.limit = limit
        
        # Initialize variables
        self.solutions = None
        self.fitness = None
        self.trial_counter = None
        self.best_solution = None
        self.best_fitness = float('inf')
        self.best_objective = float('inf')
        self.convergence_curve = []
        
    def initialize(self):
        """Initialize the food sources randomly in the search space."""
        # Initialize food sources
        self.solutions = self.lb + np.random.random((self.food_sources, self.dim)) * (self.ub - self.lb)
        self.fitness = np.zeros(self.food_sources)
        self.trial_counter = np.zeros(self.food_sources)
        
        # Evaluate initial food sources
        for i in range(self.food_sources):
            obj_val = self.objective_function(self.solutions[i])
            self.fitness[i] = self.calculate_fitness(obj_val)
            
            # Update best solution if needed
            if obj_val < self.best_objective:
                self.best_objective = obj_val
                self.best_fitness = self.fitness[i]
                self.best_solution = self.solutions[i].copy()
    
    def calculate_fitness(self, obj_val):
        """Convert objective function value to fitness value."""
        if obj_val >= 0:
            return 1 / (1 + obj_val)
        else:
            return 1 + abs(obj_val)
    
    def employed_bees_phase(self):
        """Employed bees search for new food sources."""
        for i in range(self.food_sources):
            # Generate a random neighbor excluding the current solution
            k = i
            while k == i:
                k = np.random.randint(0, self.food_sources)
                
            # Select a random dimension to modify
            j = np.random.randint(0, self.dim)
            
            # Calculate new position
            phi = np.random.uniform(-1, 1)
            new_position = self.solutions[i].copy()
            new_position[j] = self.solutions[i][j] + phi * (self.solutions[i][j] - self.solutions[k][j])
            
            # Ensure the new position is within bounds
            new_position = np.maximum(new_position, self.lb)
            new_position = np.minimum(new_position, self.ub)
            
            # Evaluate new position
            new_obj_val = self.objective_function(new_position)
            new_fitness = self.calculate_fitness(new_obj_val)
            
            # Apply greedy selection
            if new_fitness > self.fitness[i]:
                self.solutions[i] = new_position
                self.fitness[i] = new_fitness
                self.trial_counter[i] = 0
                
                # Update best solution if needed
                if new_obj_val < self.best_objective:
                    self.best_objective = new_obj_val
                    self.best_fitness = new_fitness
                    self.best_solution = new_position.copy()
            else:
                self.trial_counter[i] += 1
    
    def calculate_probabilities(self):
        """Calculate probabilities for onlooker bees based on fitness values."""
        total_fitness = np.sum(self.fitness)
        return self.fitness / total_fitness if total_fitness > 0 else np.ones(self.food_sources) / self.food_sources
    
    def onlooker_bees_phase(self):
        """Onlooker bees select food sources based on probability."""
        probabilities = self.calculate_probabilities()
        i = 0
        counter = 0
        
        # Select food sources based on their probability
        while counter < self.food_sources:
            if np.random.random() < probabilities[i]:
                counter += 1
                
                # Generate a random neighbor excluding the current solution
                k = i
                while k == i:
                    k = np.random.randint(0, self.food_sources)
                
                # Select a random dimension to modify
                j = np.random.randint(0, self.dim)
                
                # Calculate new position
                phi = np.random.uniform(-1, 1)
                new_position = self.solutions[i].copy()
                new_position[j] = self.solutions[i][j] + phi * (self.solutions[i][j] - self.solutions[k][j])
                
                # Ensure the new position is within bounds
                new_position = np.maximum(new_position, self.lb)
                new_position = np.minimum(new_position, self.ub)
                
                # Evaluate new position
                new_obj_val = self.objective_function(new_position)
                new_fitness = self.calculate_fitness(new_obj_val)
                
                # Apply greedy selection
                if new_fitness > self.fitness[i]:
                    self.solutions[i] = new_position
                    self.fitness[i] = new_fitness
                    self.trial_counter[i] = 0
                    
                    # Update best solution if needed
                    if new_obj_val < self.best_objective:
                        self.best_objective = new_obj_val
                        self.best_fitness = new_fitness
                        self.best_solution = new_position.copy()
                else:
                    self.trial_counter[i] += 1
            
            i = (i + 1) % self.food_sources
    
    def scout_bees_phase(self):
        """Scout bees explore for new food sources."""
        # Find abandoned food sources
        abandoned = np.where(self.trial_counter > self.limit)[0]
        
        # Replace abandoned food sources with new random ones
        for i in abandoned:
            self.solutions[i] = self.lb + np.random.random(self.dim) * (self.ub - self.lb)
            self.trial_counter[i] = 0
            
            # Evaluate new food source
            obj_val = self.objective_function(self.solutions[i])
            self.fitness[i] = self.calculate_fitness(obj_val)
            
            # Update best solution if needed
            if obj_val < self.best_objective:
                self.best_objective = obj_val
                self.best_fitness = self.fitness[i]
                self.best_solution = self.solutions[i].copy()
    
    def optimize(self):
        """Run the ABC optimization algorithm."""
        # Initialize the colony
        self.initialize()
        
        # Main optimization loop
        for iteration in range(self.max_iterations):
            # Employed bees phase
            self.employed_bees_phase()
            
            # Onlooker bees phase
            self.onlooker_bees_phase()
            
            # Scout bees phase
            self.scout_bees_phase()
            
            # Store best objective value for convergence curve
            self.convergence_curve.append(self.best_objective)
            
        return self.best_solution, self.best_objective, self.convergence_curve
