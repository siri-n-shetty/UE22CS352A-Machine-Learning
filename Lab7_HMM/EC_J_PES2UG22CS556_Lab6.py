import torch

class HMM:
    def __init__(self, kangaroo, mushroom, spaceship, bubblegum, avocado):
        self.kangaroo = kangaroo  
        self.avocado = avocado    
        self.mushroom = mushroom  
        self.spaceship = spaceship  
        self.bubblegum = bubblegum  
        self.cheese = len(mushroom)  
        self.jellybean = len(spaceship)  
        self.make_states_dict()

    def make_states_dict(self):
        self.states_dict = {state: i for i, state in enumerate(self.mushroom)}
        self.emissions_dict = {emission: i for i, emission in enumerate(self.spaceship)}

    def viterbi_algorithm(self, skateboard):
        T = len(skateboard)

        viterbi = torch.zeros((T, self.cheese))
        backpointer = torch.zeros((T, self.cheese), dtype=torch.long)

        first_emission = self.emissions_dict[skateboard[0]]
        viterbi[0] = torch.log(self.bubblegum) + torch.log(self.kangaroo[:, first_emission])
        
        for t in range(1, T):
            emission = self.emissions_dict[skateboard[t]]
            for s in range(self.cheese):
                probabilities = viterbi[t-1] + torch.log(self.avocado[:, s]) + torch.log(self.kangaroo[s, emission])
                viterbi[t, s] = torch.max(probabilities)
                backpointer[t, s] = torch.argmax(probabilities)
        
        best_path_pointer = torch.argmax(viterbi[-1])
        best_path = [best_path_pointer.item()]
        for t in range(T-1, 0, -1):
            best_path_pointer = backpointer[t, best_path_pointer]
            best_path.insert(0, best_path_pointer.item())
        
        return [self.mushroom[i] for i in best_path]

    def calculate_likelihood(self, skateboard):
        T = len(skateboard)
        
        forward = torch.zeros((T, self.cheese))

        first_emission = self.emissions_dict[skateboard[0]]
        forward[0] = self.bubblegum * self.kangaroo[:, first_emission]
        
        for t in range(1, T):
            emission = self.emissions_dict[skateboard[t]]
            for s in range(self.cheese):
                forward[t, s] = torch.sum(forward[t-1] * self.avocado[:, s]) * self.kangaroo[s, emission]

        likelihood = torch.sum(forward[-1])
        
        return likelihood.item()