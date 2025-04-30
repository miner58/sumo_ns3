import numpy as np
from typing import Dict, List, Tuple, Optional, Callable, Any

class OptimalStoppingRelaySelector:
    """
    Relay selection using optimal stopping theory.
    Implements the "look-then-leap" rule from the paper.
    """
    
    def __init__(self, 
                slot_time: float = 0.1,  # 100 ms
                sensing_cost: float = 0.002,  # 2 ms
                snr_threshold: float = 10.0):  # 10 dB
        """
        Initialize the optimal stopping relay selector.
        
        Args:
            slot_time: Time slot duration in seconds
            sensing_cost: Cost of sensing a relay in seconds
            snr_threshold: SNR threshold for reliable communication in dB
        """
        self.slot_time = slot_time
        self.sensing_cost = sensing_cost
        self.snr_threshold = snr_threshold
        
    def select_relay(self, 
                    candidates: List[Dict[str, Any]], 
                    reward_function: Optional[Callable] = None) -> Dict[str, Any]:
        """
        Select the best relay using optimal stopping theory.
        
        Args:
            candidates: List of candidate relays with their properties
            reward_function: Function to calculate reward for each candidate
                             If None, uses default SNR-based reward
            
        Returns:
            The selected relay
        """
        if not candidates:
            raise ValueError("Candidate list cannot be empty")
            
        # Number of candidates
        n = len(candidates)
        
        # If only one candidate, return it
        if n == 1:
            return candidates[0]
            
        # Calculate observation threshold (37% rule)
        # We observe floor(n/e) candidates without selecting
        observation_count = int(n / np.e)
        
        # If specified reward function is not provided, use default
        if reward_function is None:
            reward_function = self._default_reward_function
            
        # Observation phase
        max_reward_observed = float('-inf')
        for i in range(observation_count):
            reward = reward_function(candidates[i])
            max_reward_observed = max(max_reward_observed, reward)
            
        # Selection phase
        # Select the first candidate that exceeds the maximum observed
        for i in range(observation_count, n):
            reward = reward_function(candidates[i])
            if reward > max_reward_observed:
                return candidates[i]
                
        # If we reach here, return the last candidate
        return candidates[-1]
    
    def _default_reward_function(self, candidate: Dict[str, Any]) -> float:
        """
        Default reward function based on SNR.
        
        Args:
            candidate: Candidate relay with its properties
            
        Returns:
            Reward value
        """
        # Extract SNR
        snr = candidate.get('snr', 0.0)
        
        # Basic reward: SNR above threshold gives higher reward
        reward = max(0.0, snr - self.snr_threshold)
        
        return reward
        
    def select_relay_with_snr(self, 
                             candidates: List[Dict[str, Any]], 
                             direct_link_snr: float) -> Tuple[Dict[str, Any], bool]:
        """
        Select relay considering direct link SNR.
        Only selects relay if it provides better SNR than direct link.
        
        Args:
            candidates: List of candidate relays with their properties
            direct_link_snr: SNR of direct link between source and destination
            
        Returns:
            Tuple of (selected relay, use_relay)
            where use_relay is True if relay should be used, False for direct link
        """
        # If direct link SNR is above threshold, use direct link
        if direct_link_snr >= self.snr_threshold:
            return None, False
            
        # Filter candidates with SNR above threshold
        good_candidates = [c for c in candidates if c.get('snr', 0.0) >= self.snr_threshold]
        
        # If no good candidates, use direct link
        if not good_candidates:
            return None, False
            
        # Define reward function considering improvement over direct link
        def improved_reward(candidate):
            snr = candidate.get('snr', 0.0)
            # Reward is improvement over direct link
            return max(0.0, snr - direct_link_snr)
            
        # Select relay using optimal stopping
        selected_relay = self.select_relay(good_candidates, improved_reward)
        
        # Only use relay if it improves over direct link
        if selected_relay.get('snr', 0.0) > direct_link_snr:
            return selected_relay, True
        else:
            return None, False


class AdaptiveRelaySelector:
    """
    Adaptive relay selector with real-time relay switching.
    Combines optimal stopping with continuous monitoring.
    """
    
    def __init__(self, 
                slot_time: float = 0.1,
                sensing_cost: float = 0.002,
                snr_threshold: float = 10.0,
                hysteresis: float = 2.0):  # 2 dB hysteresis to prevent oscillation
        """
        Initialize the adaptive relay selector.
        
        Args:
            slot_time: Time slot duration in seconds
            sensing_cost: Cost of sensing a relay in seconds
            snr_threshold: SNR threshold for reliable communication in dB
            hysteresis: Hysteresis value to prevent frequent relay switching
        """
        self.optimal_stopping = OptimalStoppingRelaySelector(
            slot_time=slot_time,
            sensing_cost=sensing_cost,
            snr_threshold=snr_threshold
        )
        self.hysteresis = hysteresis
        self.current_relay = None
        self.current_relay_snr = 0.0
        
    def update_relay(self, 
                    candidates: List[Dict[str, Any]], 
                    direct_link_snr: float) -> Tuple[Dict[str, Any], bool, bool]:
        """
        Update relay selection based on current conditions.
        
        Args:
            candidates: List of candidate relays with their properties
            direct_link_snr: SNR of direct link between source and destination
            
        Returns:
            Tuple of (selected relay, use_relay, changed)
            where use_relay is True if relay should be used,
            and changed is True if relay selection changed
        """
        # If direct link is good enough and significantly better than current relay
        if direct_link_snr >= self.optimal_stopping.snr_threshold and \
           (self.current_relay is None or 
            direct_link_snr > self.current_relay_snr + self.hysteresis):
            changed = self.current_relay is not None
            self.current_relay = None
            self.current_relay_snr = 0.0
            return None, False, changed
            
        # Select relay using optimal stopping
        selected_relay, use_relay = self.optimal_stopping.select_relay_with_snr(
            candidates, direct_link_snr
        )
        
        # Check if we should change the current relay
        if use_relay:
            selected_snr = selected_relay.get('snr', 0.0)
            
            # Change relay if:
            # 1. No current relay OR
            # 2. New relay is significantly better
            if self.current_relay is None or \
               selected_snr > self.current_relay_snr + self.hysteresis:
                changed = True
                self.current_relay = selected_relay
                self.current_relay_snr = selected_snr
                return selected_relay, True, changed
        
        # No change
        return self.current_relay, self.current_relay is not None, False
        
    def reset(self):
        """Reset the relay selection state."""
        self.current_relay = None
        self.current_relay_snr = 0.0 