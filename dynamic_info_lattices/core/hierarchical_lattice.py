"""
Hierarchical Lattice Construction and Adaptation

Implements Algorithms S3-S4: Hierarchical Lattice Construction and Adaptive Lattice Refinement
from the supplementary material.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class HierarchicalLattice(nn.Module):
    """
    Hierarchical Lattice Construction and Adaptation System
    
    Implements:
    - Algorithm S3: Hierarchical Lattice Construction
    - Algorithm S4: Adaptive Lattice Refinement
    """
    
    def __init__(self, config, data_shape: Tuple[int, ...]):
        super().__init__()
        self.config = config
        self.data_shape = data_shape
        
        # Lattice parameters
        self.max_scales = config.max_scales
        self.base_resolution = config.base_resolution
        self.refinement_threshold = config.refinement_threshold
        self.coarsening_threshold = config.coarsening_threshold
        
        # Gradient computation parameters
        self.gradient_refinement_threshold = 0.1
        
    def construct_hierarchical_lattice(self, y_obs: torch.Tensor) -> Dict:
        """
        Construct hierarchical lattice structure
        
        Implements Algorithm S3: Hierarchical Lattice Construction
        
        Args:
            y_obs: Observed data [batch_size, length, channels]
            
        Returns:
            lattice: Hierarchical lattice structure
        """
        L, C = y_obs.shape[1], y_obs.shape[2]

        # Determine if we need frequency dimension based on data characteristics
        # For 1D time series, we only use (t,s) coordinates
        if len(self.data_shape) == 1 or C == 1:
            F = 1  # No frequency dimension for 1D time series
            use_frequency = False
        else:
            F = self._next_power_of_2(min(L, self.base_resolution[1]))
            use_frequency = True
        
        # Initialize lattice hierarchy
        lattice_hierarchy = {}
        all_nodes = []
        
        # Construct lattice at each scale
        for s in range(self.max_scales + 1):
            stride_t = 2 ** s
            stride_f = 2 ** s

            scale_nodes = []

            # Generate nodes for this scale - use proper coordinate system
            # At scale s, we have fewer nodes with larger regions
            num_t_nodes = max(1, L // stride_t)
            num_f_nodes = max(1, F // stride_f) if use_frequency else 1

            for t_idx in range(num_t_nodes):
                # Convert index to spatial coordinate with proper bounds checking
                t = t_idx * stride_t

                # CUDA-safe coordinate validation with strict bounds checking
                if t >= L or t < 0:  # Skip if beyond bounds
                    continue

                # Additional safety: ensure t + scale_factor doesn't exceed bounds
                scale_factor = 2 ** s
                if t + scale_factor > L:
                    # Adjust t to fit within bounds, but ensure it's still valid
                    t = max(0, L - scale_factor)
                    if t < 0 or t >= L:  # If scale_factor > L, skip this node
                        continue

                # Final validation: ensure t is within valid range
                if t < 0 or t >= L:
                    continue

                if use_frequency:
                    # 2D case: use both t and f dimensions
                    for f_idx in range(num_f_nodes):
                        f = f_idx * stride_f
                        if f >= F:  # Skip if beyond bounds
                            continue
                        # Additional safety for frequency dimension
                        if f + scale_factor > F:
                            f = max(0, F - scale_factor)
                            if f < 0:
                                f = 0
                        node = (t, f, s)
                        scale_nodes.append(node)
                        all_nodes.append(node)
                else:
                    # 1D case: only t dimension, f=0 as placeholder
                    node = (t, 0, s)
                    scale_nodes.append(node)
                    all_nodes.append(node)
            
            lattice_hierarchy[s] = scale_nodes
            
            logger.debug(f"Scale {s}: {len(scale_nodes)} nodes, stride=({stride_t}, {stride_f})")
        
        # Create lattice structure
        lattice = {
            'hierarchy': lattice_hierarchy,
            'active_nodes': all_nodes,
            'resolution': (L, F),
            'max_scales': self.max_scales,
            'use_frequency': use_frequency,
            'parent_child_map': self._build_parent_child_map(lattice_hierarchy),
            'node_coordinates': self._build_coordinate_map(all_nodes)
        }
        
        logger.info(f"Constructed hierarchical lattice with {len(all_nodes)} total nodes")
        
        return lattice
    
    def adapt_lattice(
        self,
        current_lattice: Dict,
        entropy_map: torch.Tensor,
        k: int,
        entropy_gradients: torch.Tensor = None
    ) -> Dict:
        """
        Adapt lattice structure based on entropy patterns

        Implements Algorithm S4: Adaptive Lattice Refinement with CUDA-safe bounds checking

        Args:
            current_lattice: Current lattice structure
            entropy_map: Entropy values for each node
            k: Current diffusion step

        Returns:
            adapted_lattice: Updated lattice structure
        """
        active_nodes = current_lattice['active_nodes']
        hierarchy = current_lattice['hierarchy'].copy()

        # CUDA-safe size limits to prevent memory/indexing issues
        MAX_LATTICE_SIZE = 50000  # Prevent lattices from growing too large
        if len(active_nodes) > MAX_LATTICE_SIZE:
            logger.warning(f"Lattice size ({len(active_nodes)}) exceeds safe limit ({MAX_LATTICE_SIZE}), skipping adaptation")
            return current_lattice

        # Ensure entropy_map and active_nodes are compatible sizes
        if len(entropy_map) != len(active_nodes):
            logger.warning(f"Entropy map size ({len(entropy_map)}) != active nodes size ({len(active_nodes)}), adjusting")
            # Safely resize entropy_map to match active_nodes
            if len(entropy_map) > len(active_nodes):
                entropy_map = entropy_map[:len(active_nodes)]
            else:
                # Pad with zeros if entropy_map is smaller
                padding = torch.zeros(len(active_nodes) - len(entropy_map), device=entropy_map.device, dtype=entropy_map.dtype)
                entropy_map = torch.cat([entropy_map, padding])

        # Compute entropy gradients for refinement decisions if not provided
        if entropy_gradients is None:
            entropy_gradients = self._compute_entropy_gradients(
                entropy_map, active_nodes, current_lattice
            )
        
        # Phase 1: Refinement with CUDA-safe bounds checking
        refinement_candidates = []
        for i, (t, f, s) in enumerate(active_nodes):
            # CUDA-safe bounds checking for all array accesses
            if i >= len(entropy_map) or i >= len(entropy_gradients):
                continue

            if s < self.max_scales:  # Can only refine if not at maximum scale
                try:
                    entropy_val = float(entropy_map[i]) if i < len(entropy_map) else 0.0
                    gradient_val = float(entropy_gradients[i]) if i < len(entropy_gradients) else 0.0

                    # Additional validation for tensor values
                    if torch.isnan(entropy_map[i]) or torch.isinf(entropy_map[i]):
                        entropy_val = 0.0
                    if torch.isnan(entropy_gradients[i]) or torch.isinf(entropy_gradients[i]):
                        gradient_val = 0.0

                    # Refinement criterion with safe value checks
                    if (entropy_val > self.refinement_threshold and
                        gradient_val > self.gradient_refinement_threshold and
                        entropy_val < 1000.0 and gradient_val < 1000.0):  # Sanity bounds

                        # Validate coordinates before adding to candidates
                        if self._validate_coordinates(t, f, s):
                            refinement_candidates.append((t, f, s))
                except Exception as e:
                    logger.warning(f"Error processing refinement candidate {i}: {e}")
                    continue
        
        # Apply refinements
        new_nodes = []
        for t, f, s in refinement_candidates:
            children = self._subdivide_node(t, f, s)
            new_nodes.extend(children)
            
            # Add to hierarchy
            child_scale = s + 1
            if child_scale not in hierarchy:
                hierarchy[child_scale] = []
            hierarchy[child_scale].extend(children)
        
        # Phase 2: Coarsening with CUDA-safe bounds checking
        coarsening_candidates = []
        for s in range(1, self.max_scales + 1):  # Start from scale 1
            if s in hierarchy:
                scale_nodes = hierarchy[s]

                # Group siblings for coarsening evaluation
                try:
                    sibling_groups = self._group_siblings(scale_nodes)
                except Exception as e:
                    logger.warning(f"Error grouping siblings at scale {s}: {e}")
                    continue

                for siblings in sibling_groups:
                    # Check if all siblings have low entropy with CUDA-safe indexing
                    all_low_entropy = True
                    valid_siblings = []

                    for sibling in siblings:
                        try:
                            if sibling in active_nodes:
                                idx = active_nodes.index(sibling)
                                # CUDA-safe bounds checking
                                if 0 <= idx < len(entropy_map):
                                    entropy_val = float(entropy_map[idx])

                                    # Validate tensor value
                                    if torch.isnan(entropy_map[idx]) or torch.isinf(entropy_map[idx]):
                                        entropy_val = float('inf')  # Treat invalid values as high entropy

                                    if entropy_val >= self.coarsening_threshold:
                                        all_low_entropy = False
                                        break
                                    valid_siblings.append(sibling)
                                else:
                                    # Index out of bounds, treat as high entropy
                                    all_low_entropy = False
                                    break
                        except Exception as e:
                            logger.warning(f"Error processing sibling {sibling}: {e}")
                            all_low_entropy = False
                            break

                    if all_low_entropy and len(valid_siblings) > 1:
                        coarsening_candidates.append(valid_siblings)
        
        # Apply coarsening
        nodes_to_remove = []
        nodes_to_add = []
        
        for siblings in coarsening_candidates:
            # Get parent node
            if siblings:
                t, f, s = siblings[0]
                parent = self._get_parent_node(t, f, s)
                nodes_to_add.append(parent)
                nodes_to_remove.extend(siblings)
                
                # Update hierarchy
                parent_scale = s - 1
                if parent_scale not in hierarchy:
                    hierarchy[parent_scale] = []
                if parent not in hierarchy[parent_scale]:
                    hierarchy[parent_scale].append(parent)
                
                # Remove siblings from hierarchy
                for sibling in siblings:
                    if sibling in hierarchy[s]:
                        hierarchy[s].remove(sibling)
        
        # Update active nodes
        updated_active_nodes = [
            node for node in active_nodes 
            if node not in nodes_to_remove
        ]
        updated_active_nodes.extend(new_nodes)
        updated_active_nodes.extend(nodes_to_add)
        
        # Remove duplicates while preserving order
        seen = set()
        final_active_nodes = []
        for node in updated_active_nodes:
            if node not in seen:
                seen.add(node)
                final_active_nodes.append(node)
        
        # Create updated lattice
        adapted_lattice = {
            'hierarchy': hierarchy,
            'active_nodes': final_active_nodes,
            'resolution': current_lattice['resolution'],
            'max_scales': self.max_scales,
            'parent_child_map': self._build_parent_child_map(hierarchy),
            'node_coordinates': self._build_coordinate_map(final_active_nodes)
        }
        
        logger.debug(f"Lattice adaptation: {len(refinement_candidates)} refinements, "
                    f"{len(coarsening_candidates)} coarsenings, "
                    f"{len(final_active_nodes)} total nodes")
        
        return adapted_lattice
    
    def synchronize_scales(
        self,
        z: torch.Tensor,
        lattice: Dict
    ) -> torch.Tensor:
        """Synchronize values across different scales to maintain consistency"""
        # Implement cross-scale synchronization using interpolation
        synchronized_z = z.clone()
        
        hierarchy = lattice['hierarchy']
        
        # Synchronize from coarse to fine scales
        for s in range(self.max_scales):
            if s in hierarchy and s + 1 in hierarchy:
                coarse_nodes = hierarchy[s]
                fine_nodes = hierarchy[s + 1]
                
                # For each coarse node, interpolate to its children
                for t_c, f_c, s_c in coarse_nodes:
                    # Find children
                    children = self._get_children_nodes(t_c, f_c, s_c)
                    
                    if children:
                        # Get coarse value
                        coarse_region = self._extract_region(synchronized_z, t_c, f_c, s_c)
                        
                        # Interpolate to children
                        for t_f, f_f, s_f in children:
                            if (t_f, f_f, s_f) in fine_nodes:
                                fine_region = self._interpolate_region(
                                    coarse_region, t_c, f_c, s_c, t_f, f_f, s_f
                                )
                                synchronized_z = self._update_region(
                                    synchronized_z, fine_region, t_f, f_f, s_f
                                )
        
        return synchronized_z
    
    def _next_power_of_2(self, x: int) -> int:
        """Find next power of 2 greater than or equal to x"""
        return 1 << (x - 1).bit_length()
    
    def _build_parent_child_map(self, hierarchy: Dict) -> Dict:
        """Build mapping between parent and child nodes"""
        parent_child_map = {}
        
        for s in range(self.max_scales):
            if s in hierarchy and s + 1 in hierarchy:
                for t, f, scale in hierarchy[s]:
                    children = self._get_children_nodes(t, f, scale)
                    parent_child_map[(t, f, scale)] = children
        
        return parent_child_map
    
    def _build_coordinate_map(self, nodes: List[Tuple[int, int, int]]) -> Dict:
        """Build mapping from nodes to their spatial coordinates

        FIXED: Use consistent coordinate system with bounds checking
        """
        coordinate_map = {}

        for t, f, s in nodes:
            scale_factor = 2 ** s
            # t,f are already spatial coordinates, define region based on scale
            # Add bounds checking to ensure coordinates are valid
            t_start = max(0, t)
            t_end = t + scale_factor
            f_start = max(0, f)
            f_end = f + scale_factor

            coordinate_map[(t, f, s)] = {
                't_range': (t_start, t_end),
                'f_range': (f_start, f_end),
                'scale_factor': scale_factor
            }

        return coordinate_map
    
    def _compute_entropy_gradients(
        self,
        entropy_map: torch.Tensor,
        active_nodes: List[Tuple[int, int, int]],
        lattice: Dict
    ) -> torch.Tensor:
        """Compute spatial gradients of entropy for refinement decisions with CUDA-safe bounds checking"""
        gradients = torch.zeros_like(entropy_map)

        for i, (t, f, s) in enumerate(active_nodes):
            # CUDA-safe bounds checking
            if i >= len(entropy_map) or i >= len(gradients):
                continue

            try:
                # Validate current entropy value
                if torch.isnan(entropy_map[i]) or torch.isinf(entropy_map[i]):
                    gradients[i] = 0.0
                    continue

                # Find neighboring nodes at the same scale
                neighbors = self._find_neighbors(t, f, s, active_nodes)

                if neighbors:
                    neighbor_entropies = []
                    for nt, nf, ns in neighbors:
                        try:
                            if (nt, nf, ns) in active_nodes:
                                neighbor_idx = active_nodes.index((nt, nf, ns))
                                # CUDA-safe bounds checking for neighbor access
                                if 0 <= neighbor_idx < len(entropy_map):
                                    neighbor_entropy = entropy_map[neighbor_idx]
                                    # Validate neighbor entropy value
                                    if not (torch.isnan(neighbor_entropy) or torch.isinf(neighbor_entropy)):
                                        neighbor_entropies.append(neighbor_entropy)
                        except Exception as e:
                            logger.debug(f"Error processing neighbor ({nt}, {nf}, {ns}): {e}")
                            continue

                    if len(neighbor_entropies) > 0:
                        try:
                            neighbor_tensor = torch.stack(neighbor_entropies)
                            current_entropy = entropy_map[i]

                            # Compute gradient magnitude with safe operations
                            diff_tensor = neighbor_tensor - current_entropy
                            gradient_mag = torch.std(diff_tensor)

                            # Validate gradient value before assignment
                            if not (torch.isnan(gradient_mag) or torch.isinf(gradient_mag)):
                                gradients[i] = gradient_mag
                            else:
                                gradients[i] = 0.0
                        except Exception as e:
                            logger.debug(f"Error computing gradient for node {i}: {e}")
                            gradients[i] = 0.0
                    else:
                        gradients[i] = 0.0
                else:
                    gradients[i] = 0.0

            except Exception as e:
                logger.warning(f"Error processing entropy gradient for node {i}: {e}")
                if i < len(gradients):
                    gradients[i] = 0.0
                continue

        return gradients
    
    def _validate_coordinates(self, t: int, f: int, s: int) -> bool:
        """Validate that coordinates are within safe bounds"""
        try:
            # Check for reasonable coordinate values
            if t < 0 or f < 0 or s < 0:
                return False
            if s > self.max_scales:
                return False

            # Check against data shape bounds
            L, F = self.data_shape[:2] if len(self.data_shape) >= 2 else (self.data_shape[0], 1)
            scale_factor = 2 ** s

            # Ensure the region defined by these coordinates fits within data bounds
            if t + scale_factor > L or f + scale_factor > F:
                return False

            return True
        except Exception:
            return False

    def _subdivide_node(self, t: int, f: int, s: int) -> List[Tuple[int, int, int]]:
        """Subdivide a node into its children with CUDA-safe bounds checking"""
        children = []
        child_scale = s + 1

        # Validate parent node first
        if not self._validate_coordinates(t, f, s) or child_scale > self.max_scales:
            return children

        # Each node subdivides into 4 children (2x2) with bounds checking
        L, F = self.data_shape[:2] if len(self.data_shape) >= 2 else (self.data_shape[0], 1)

        for dt in [0, 1]:
            for df in [0, 1]:
                child_t = 2 * t + dt
                child_f = 2 * f + df

                # Validate child coordinates before adding
                if self._validate_coordinates(child_t, child_f, child_scale):
                    children.append((child_t, child_f, child_scale))

        return children
    
    def _get_children_nodes(self, t: int, f: int, s: int) -> List[Tuple[int, int, int]]:
        """Get children of a node"""
        return self._subdivide_node(t, f, s)
    
    def _get_parent_node(self, t: int, f: int, s: int) -> Tuple[int, int, int]:
        """Get parent of a node"""
        if s == 0:
            return (t, f, s)  # Root level has no parent
        
        parent_scale = s - 1
        parent_t = t // 2
        parent_f = f // 2
        
        return (parent_t, parent_f, parent_scale)
    
    def _group_siblings(self, nodes: List[Tuple[int, int, int]]) -> List[List[Tuple[int, int, int]]]:
        """Group nodes that are siblings (share the same parent)"""
        sibling_groups = {}
        
        for t, f, s in nodes:
            parent = self._get_parent_node(t, f, s)
            if parent not in sibling_groups:
                sibling_groups[parent] = []
            sibling_groups[parent].append((t, f, s))
        
        return list(sibling_groups.values())
    
    def _find_neighbors(
        self,
        t: int,
        f: int,
        s: int,
        active_nodes: List[Tuple[int, int, int]]
    ) -> List[Tuple[int, int, int]]:
        """Find neighboring nodes at the same scale"""
        neighbors = []
        
        # Check 8-connected neighbors
        for dt in [-1, 0, 1]:
            for df in [-1, 0, 1]:
                if dt == 0 and df == 0:
                    continue
                
                neighbor = (t + dt, f + df, s)
                if neighbor in active_nodes:
                    neighbors.append(neighbor)
        
        return neighbors
    
    def _extract_region(
        self,
        z: torch.Tensor,
        t: int,
        f: int,
        s: int
    ) -> torch.Tensor:
        """Extract spatial region corresponding to lattice node

        FIXED: Use consistent coordinate system - t,f are spatial positions
        """
        scale_factor = 2 ** s

        # t,f are already spatial coordinates, define region size based on scale
        t_start = t
        t_end = min(t + scale_factor, z.shape[1])

        if len(z.shape) >= 3:
            f_start = f
            f_end = min(f + scale_factor, z.shape[2])
            return z[:, t_start:t_end, f_start:f_end]
        else:
            return z[:, t_start:t_end]
    
    def _interpolate_region(
        self,
        coarse_region: torch.Tensor,
        t_c: int, f_c: int, s_c: int,
        t_f: int, f_f: int, s_f: int
    ) -> torch.Tensor:
        """Interpolate coarse region to fine resolution"""
        # Simple bilinear interpolation
        # In practice, this could be more sophisticated
        target_size = (2 ** s_f, 2 ** s_f)
        
        interpolated = F.interpolate(
            coarse_region.unsqueeze(0),
            size=target_size,
            mode='bilinear',
            align_corners=False
        ).squeeze(0)
        
        return interpolated
    
    def _update_region(
        self,
        z: torch.Tensor,
        region: torch.Tensor,
        t: int,
        f: int,
        s: int
    ) -> torch.Tensor:
        """Update spatial region in global tensor"""
        z_updated = z.clone()
        
        scale_factor = 2 ** s
        t_start = t * scale_factor
        t_end = min((t + 1) * scale_factor, z.shape[1])
        f_start = f * scale_factor
        f_end = min((f + 1) * scale_factor, z.shape[2])
        
        # Ensure region fits
        region_h = min(region.shape[1], t_end - t_start)
        region_w = min(region.shape[2], f_end - f_start)
        
        z_updated[:, t_start:t_start+region_h, f_start:f_start+region_w] = \
            region[:, :region_h, :region_w]
        
        return z_updated


# Fix missing import
import torch.nn.functional as F
