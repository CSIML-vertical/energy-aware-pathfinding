"""
Multi-Objective Energy-Aware A* Pathfinder (Mihir)
Finds optimal routes dynamically balancing traversal time and predicted energy expenditure:
Cost(e) = lambda * Time(e) + (1 - lambda) * Energy(e)
"""
import heapq
from typing import Dict, List, Tuple, Optional


def euclidean_distance(pos1: Tuple[float, float], pos2: Tuple[float, float]) -> float:
    return ((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)**0.5


class EnergyAStarRouter:
    def __init__(self, network, energy_predictor):
        self.network = network
        self.predictor = energy_predictor

    def _get_node_pos(self, node_id: str) -> Tuple[float, float]:
        """Safely fetch node (x, y) coordinates if available, fallback to (0, 0)."""
        if hasattr(self.network, "nodes") and isinstance(self.network.nodes, dict):
            val = self.network.nodes.get(node_id)
            if isinstance(val, (tuple, list)) and len(val) >= 2:
                return (float(val[0]), float(val[1]))
        if hasattr(self.network, "positions") and isinstance(self.network.positions, dict):
            val = self.network.positions.get(node_id)
            if isinstance(val, (tuple, list)) and len(val) >= 2:
                return (float(val[0]), float(val[1]))
        return (0.0, 0.0)

    def heuristic(
        self,
        current_node: str,
        goal_node: str,
        lambda_weight: float,
        max_speed_kmph: float = 100.0,
        min_kwh_per_km: float = 0.10,
    ) -> float:
        """Admissible multi-objective heuristic."""
        pos1 = self._get_node_pos(current_node)
        pos2 = self._get_node_pos(goal_node)
        if pos1 == (0.0, 0.0) and pos2 == (0.0, 0.0):
            return 0.0  # Safe Dijkstra fallback if coordinates aren't stored
        dist = euclidean_distance(pos1, pos2)
        h_time_min = (dist / max_speed_kmph) * 60.0
        h_energy_kwh = dist * min_kwh_per_km
        return lambda_weight * h_time_min + (1.0 - lambda_weight) * h_energy_kwh

    def find_path(
        self,
        start_node: str,
        goal_node: str,
        segment_roughness_map: Dict[str, float],
        lambda_weight: float = 0.5,
    ) -> Optional[dict]:
        all_nodes = (
            set(self.network.nodes.keys())
            if isinstance(self.network.nodes, dict)
            else set(self.network.nodes)
        )
        if start_node not in all_nodes or goal_node not in all_nodes:
            raise ValueError(f"Start ({start_node}) or Goal ({goal_node}) not in network.")

        queue = []
        # Priority Queue state: (f_cost, g_time, g_energy, curr_node, node_path, seg_path)
        heapq.heappush(queue, (0.0, 0.0, 0.0, start_node, [start_node], []))
        visited_cost = {}

        while queue:
            f_cost, g_time, g_energy, curr_node, node_path, seg_path = heapq.heappop(queue)

            if curr_node == goal_node:
                return {
                    "nodes": node_path,
                    "segments": seg_path,
                    "total_time_min": round(g_time, 2),
                    "total_energy_kwh": round(g_energy, 2),
                    "total_cost": round(
                        lambda_weight * g_time + (1.0 - lambda_weight) * g_energy, 4
                    ),
                }

            if curr_node in visited_cost and visited_cost[curr_node] <= f_cost:
                continue
            visited_cost[curr_node] = f_cost

            for neighbor, seg_id in self.network.get_neighbors(curr_node):
                seg = self.network.get_segment(seg_id)
                roughness = segment_roughness_map.get(seg_id, seg.initial_roughness)

                # Segment travel time (rough roads slow vehicles down)
                effective_speed = max(5.0, seg.speed_limit_kmph * (1.0 - 0.35 * roughness))
                edge_time_min = (seg.length_km / effective_speed) * 60.0

                # ML Predicted segment energy consumption
                edge_energy_kwh = self.predictor.predict_segment_energy(
                    seg.length_km, seg.speed_limit_kmph, roughness
                )

                edge_cost = (
                    lambda_weight * edge_time_min + (1.0 - lambda_weight) * edge_energy_kwh
                )
                next_g_time = g_time + edge_time_min
                next_g_energy = g_energy + edge_energy_kwh
                current_g_cost = (
                    lambda_weight * g_time + (1.0 - lambda_weight) * g_energy + edge_cost
                )

                h_val = self.heuristic(neighbor, goal_node, lambda_weight)
                f_next = current_g_cost + h_val

                heapq.heappush(
                    queue,
                    (
                        f_next,
                        next_g_time,
                        next_g_energy,
                        neighbor,
                        node_path + [neighbor],
                        seg_path + [seg_id],
                    ),
                )

        return None
    