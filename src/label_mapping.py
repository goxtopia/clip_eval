import json
import os
from typing import Dict, List, Set, Tuple

class LabelMapper:
    def __init__(self, mapping_path: str):
        self.equivalents: List[Set[str]] = []
        self.implications: Dict[str, Set[str]] = {}
        
        if os.path.exists(mapping_path):
            self._load_mapping(mapping_path)
        else:
            print(f"Warning: Mapping file not found at {mapping_path}. Using empty mapping.")

    def _load_mapping(self, path: str):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Parse equivalents (bidirectional)
        raw_eqs = data.get("equivalents", [])
        # We'll merge sets of equivalents
        # Using a Disjoint Set Union (DSU) approach or simple propagation would be ideal,
        # but for simplicity we'll just store sets.
        # Actually, simpler: map each word to a canonical ID or set of synonyms.
        # Let's just store the sets of synonyms.
        
        # Naive approach: merge overlapping sets
        sets = [set(group) for group in raw_eqs]
        merged = True
        while merged:
            merged = False
            new_sets = []
            while sets:
                current = sets.pop(0)
                # Check against remaining
                overlap_idx = -1
                for i, other in enumerate(new_sets):
                    if not current.isdisjoint(other):
                        overlap_idx = i
                        break
                
                if overlap_idx != -1:
                    new_sets[overlap_idx].update(current)
                    merged = True
                else:
                    new_sets.append(current)
            sets = new_sets
        
        self.equivalent_map = {}
        for group in sets:
            for word in group:
                self.equivalent_map[word] = group

        # Parse implications (directional)
        raw_imps = data.get("implications", {})
        # We need to compute transitive closure
        # A -> B, B -> C  => A -> {B, C}
        
        # First load direct implications
        adj = {k: set(v) for k, v in raw_imps.items()}
        
        # Transitive closure
        # Since the graph shouldn't be too large or deep, a simple loop is fine.
        # Or just use DFS/BFS for each node.
        
        # Ensure we handle all nodes
        nodes = set(adj.keys())
        for v in adj.values():
            nodes.update(v)
            
        self.closure = {}
        
        # Memoized DFS
        def get_closure(node):
            if node in self.closure:
                return self.closure[node]
            
            res = set()
            # Direct children
            children = adj.get(node, set())
            res.update(children)
            
            # Recursive
            for child in children:
                res.update(get_closure(child))
            
            self.closure[node] = res
            return res

        for node in nodes:
            get_closure(node)
            
    def expand_label(self, label: str) -> Set[str]:
        """
        Returns a set containing the label, its equivalents, and all implied labels.
        """
        result = {label}
        
        # 1. Add equivalents
        # If label is in an equivalent group, add all of them
        if label in self.equivalent_map:
            result.update(self.equivalent_map[label])
        
        # 2. Add implications for all current labels
        # (Implications might trigger more equivalents? 
        #  User said 'person' <-> 'people'. 'woman' -> 'person'.
        #  Does 'woman' -> 'people'? Yes, via 'person'.
        #  So we should check implications for everything in result so far.)
        
        # Iteratively expand until stable
        changed = True
        while changed:
            changed = False
            current_snapshot = list(result)
            for l in current_snapshot:
                # Add implications
                if l in self.closure:
                    imps = self.closure[l]
                    for imp in imps:
                        if imp not in result:
                            result.add(imp)
                            changed = True
                            
                # Add equivalents
                if l in self.equivalent_map:
                    eqs = self.equivalent_map[l]
                    for eq in eqs:
                        if eq not in result:
                            result.add(eq)
                            changed = True
                            
        return result

    def expand_label_set(self, labels: Set[str]) -> Set[str]:
        """Expands a set of labels."""
        result = set()
        for l in labels:
            result.update(self.expand_label(l))
        return result
