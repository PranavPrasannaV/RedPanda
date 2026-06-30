
import math
import numpy as np
import torch
import chess
from encoding import encoder


class MCTSNode:
    def __init__(self, game_state: chess.Board, parent=None, action_taken=None, prior=0):
        self.game_state = game_state
        self.parent = parent
        self.action_taken = action_taken
        self.children = {} # Map move -> MCTSNode
        self.visit_count = 0
        self.value_sum = 0
        self.prior = prior
        self.virtual_loss = 0 # For parallel search
        # Store WDL for richer analysis
        self.wdl = None  # [P(Win), P(Draw), P(Loss)]
        
    def is_fully_expanded(self):
        return len(self.children) > 0
    
    def value(self):
        if self.visit_count == 0:
            return 0
        return (self.value_sum - self.virtual_loss) / self.visit_count

    def select_child(self, c_puct=1.0):
        best_score = -float('inf')
        best_child = None
        
        for action, child in self.children.items():
            ucb_score = self._calculate_ucb(child, c_puct)
            if ucb_score > best_score:
                best_score = ucb_score
                best_child = child
                
        return best_child
    
    def _calculate_ucb(self, child, c_puct):
        # child.value() is from the child's (opponent's) perspective, so negate
        # to score the move from THIS node's perspective.
        q_value = -child.value() if child.visit_count > 0 else 0
        u_value = c_puct * child.prior * math.sqrt(self.visit_count) / (1 + child.visit_count)
        return q_value + u_value

    def expand(self, policy_logits, valid_moves):
        policy_probs = torch.softmax(policy_logits, dim=-1)
        
        prob_sum = 0
        children_predictions = []
        
        for move in valid_moves:
            move_id = encoder.encode_move(move)
            if move_id is not None and move_id < len(policy_probs):
                prior = policy_probs[move_id].item()
            else:
                prior = 0.0001
            
            prob_sum += prior
            children_predictions.append((move, prior))
            
        for move, prior in children_predictions:
            normalized_prior = prior / prob_sum if prob_sum > 0 else 1.0 / len(valid_moves)
            
            next_state = self.game_state.copy()
            next_state.push(move)
            
            child = MCTSNode(next_state, parent=self, action_taken=move, prior=normalized_prior)
            self.children[move] = child


class MCTS:
    def __init__(self, model, num_simulations=800, c_puct=1.4, contempt=0.0):
        """
        Args:
            model: The ChessMamba model with WDL head.
            num_simulations: Number of MCTS simulations.
            c_puct: Exploration constant.
            contempt: Contempt factor [0, 1]. 
                      0 = neutral (default), 
                      0.5 = moderately aggressive (prefers wins over draws),
                      1.0 = very aggressive.
        """
        self.model = model
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.contempt = contempt
        
    def _get_value_from_wdl(self, wdl):
        """Convert WDL probabilities to a scalar value with contempt."""
        return self.model.get_value(wdl, self.contempt).item()
        
    def run_search(self, board: chess.Board):
        root = MCTSNode(board)
        device = next(self.model.parameters()).device
        
        # Initial expansion
        input_ids = encoder.encode_board(board)
        input_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)
        
        with torch.no_grad():
            policy_logits, wdl = self.model(input_tensor)
        
        root.wdl = wdl[0].cpu().numpy()
        root.expand(policy_logits[0], list(board.legal_moves))
        
        for _ in range(self.num_simulations):
            node = root
            search_path = [node]
            
            # Selection
            while node.is_fully_expanded():
                node = node.select_child(self.c_puct)
                search_path.append(node)
                node.virtual_loss += 1
                
            # Expansion & Evaluation
            outcome = node.game_state.outcome()
            
            if outcome is None:
                input_ids = encoder.encode_board(node.game_state)
                input_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)
                
                with torch.no_grad():
                    policy_logits, wdl = self.model(input_tensor)
                
                node.wdl = wdl[0].cpu().numpy()
                value = self._get_value_from_wdl(wdl)
                node.expand(policy_logits[0], list(node.game_state.legal_moves))
            else:
                # Terminal state: hard outcome
                if outcome.winner is None:
                    value = 0 
                else:
                    value = 1 if outcome.winner == node.game_state.turn else -1
            
            # Backpropagation
            for path_node in reversed(search_path):
                 path_node.value_sum += value
                 path_node.visit_count += 1
                 path_node.virtual_loss = 0
                 value = -value

        return root
    
    def get_move_stats(self, root):
        """Returns stats for all moves from the root for analysis."""
        stats = []
        for move, child in root.children.items():
            stats.append({
                "move": move.uci(),
                "visits": child.visit_count,
                "value": child.value(),
                "wdl": child.wdl.tolist() if child.wdl is not None else None
            })
        return sorted(stats, key=lambda x: x["visits"], reverse=True)

    def search(self, board: chess.Board):
        root = self.run_search(board)
        if not root.children:
            return None 
        return max(root.children, key=lambda move: root.children[move].visit_count)
