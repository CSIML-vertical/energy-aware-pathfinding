from dataclasses import dataclass
from random import Random
from typing import Dict, List, Tuple
import math
#ROAD CLASS SETTINGS
ROAD_CLASSES = {
    "highway":{
        "roughness_range": (0.05, 0.35),
        "initial_estimate": 0.15,
        "speed_limit_kmph": 100,},
    "main_road":{
        "roughness_range": (0.20, 0.65),
        "initial_estimate": 0.40,
        "speed_limit_kmph": 50,},
    "small_road":{
        "roughness_range": (0.35, 0.90),
        "initial_estimate": 0.60,
        "speed_limit_kmph": 25,},
}
#ROAD SEGMENT DATA STRUCTUR
@dataclass
class RoadSegment:
    segment_id: str
    start_node: str
    end_node: str
    road_class: str
    length_km: float
    initial_roughness: float
    true_roughness: float
    speed_limit_kmph: float
    
#ROAD NETWORK GRAPH
class RoadNetwork:
    def __init__(self):
        self.nodes = set()
        self.segments: Dict[str, RoadSegment] = {}
        self.adjacency: Dict[str,List[Tuple[str,str]]] = {}
    def add_node(self, node_id: str) -> None:
        if not node_id:
            raise ValueError("Node ID cannot be empty.")
        self.nodes.add(node_id)
        self.adjacency.setdefault(node_id, [])
    def add_segment(self, segment: RoadSegment) -> None:
        if segment.segment_id in self.segments:
            raise ValueError("Duplicate segment ID.")

        if segment.start_node not in self.nodes:
            raise ValueError("Start node does not exist.")

        if segment.end_node not in self.nodes:
            raise ValueError("End node does not exist.")

        if segment.start_node == segment.end_node:
            raise ValueError("Self-loops are not allowed.")

        if segment.road_class not in ROAD_CLASSES:
            raise ValueError("Invalid road class.")

        if segment.length_km <= 0:
            raise ValueError("Road length must not be negative.")
        for value in (
            segment.initial_roughness,
            segment.true_roughness,
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError("Roughness must be in [0, 1].")
        if segment.speed_limit_kmph <= 0:
            raise ValueError("Speed limit must be positive.")

            
        self.segments[segment.segment_id] =segment
        self.adjacency[segment.start_node].append((segment.end_node, segment.segment_id))
        self.adjacency[segment.end_node].append((segment.start_node, segment.segment_id))
    def get_segment(self, segment_id: str) -> RoadSegment:
        return self.segments[segment_id]
    
    def get_observation_context(self, segment_id: str) -> dict:
        """
        Return road attributes that can be used by the
        vehicle-observation simulator.

        Deliberately excludes true_roughness.
        """
        segment = self.get_segment(segment_id)

        return {
            "segment_id": segment.segment_id,
            "road_class": segment.road_class,
            "length_km": segment.length_km,
            "speed_limit_kmph": segment.speed_limit_kmph,
            "initial_roughness": segment.initial_roughness,
        }

    def get_simulation_truth(self, segment_id: str) -> float:
        """
        Simulation-only access to hidden ground truth.

        Use this inside the synthetic observation generator,
        never as an estimator feature.
        """
        return self.get_segment(segment_id).true_roughness

    def get_neighbors(self, node_id: str):
        if node_id not in self.nodes:
            raise KeyError(f"Unknown node: {node_id}")

        return list(self.adjacency[node_id])
    def get_segments_by_class(self, road_class: str):
        return [
            segment
            for segment in self.segments.values()
            if segment.road_class== road_class
        ]
    def is_connected(self) -> bool:
        if not self.nodes:
            return True
        start = next(iter(self.nodes))
        visited = {start}
        stack = [start]

        while stack:
            current = stack.pop()
            for neighbor, _ in self.adjacency[current]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    stack.append(neighbor)

        return visited == self.nodes
    def validate(self) -> bool:
        for segment_id, segment in self.segments.items():
            if segment_id != segment.segment_id:
                return False
            if segment.start_node not in self.nodes:
                return False
            if segment.end_node not in self.nodes:
                return False

        for node, neighbors in self.adjacency.items():
            if node not in self.nodes:
                return False
            for neighbor, segment_id in neighbors:
                if neighbor not in self.nodes:
                    return False
                if segment_id not in self.segments:
                    return False

        return True

    def summary(self) -> dict:
        return {
            "number_of_nodes": len(self.nodes),
            "number_of_segments": len(self.segments),
            "connected": self.is_connected(),
            "valid": self.validate(),
            "road_class_counts": {
                road_class: len(
                    self.get_segments_by_class(road_class)
                )
                for road_class in ROAD_CLASSES
            },
        }
#CONFIGURABLE NETWORK GENERATOR

def generate_road_network(
    num_nodes: int = 50,
    num_segments: int = 80,
    seed: int = 42,
    initial_noise: float = 0.15,
    class_weights: dict = None,
) -> RoadNetwork:
    """
    Generate a connected, undirected road network.

    num_nodes:
        Number of intersections.
    num_segments:
        Number of road segments. Must be at least num_nodes - 1.
    seed:
        Random seed for reproducible experiments.
    initial_noise:
        Maximum absolute deviation from the class-based
        initial estimate.
    class_weights:
        Optional relative probabilities for road classes.
        Example: {"highway": 0.2, "main_road": 0.5,
                  "small_road": 0.3}.
    """

    if num_nodes < 1:
        raise ValueError("num_nodes must be at least 1.")

    if num_segments < num_nodes - 1:
        raise ValueError(
            "A connected network requires at least num_nodes - 1 "
            "segments."
        )
    max_segments = num_nodes * (num_nodes - 1) // 2
    if num_segments > max_segments:
        raise ValueError(
            "Too many segments for a simple undirected graph.")
    if initial_noise < 0:
        raise ValueError("initial_noise cannot be negative.")
    rng = Random(seed)
    if class_weights is None:
        class_weights = {
            "highway": 0.25,
            "main_road": 0.45,
            "small_road": 0.30,
        }

    if (
        set(class_weights) != set(ROAD_CLASSES)
        or any(weight < 0 for weight in class_weights.values())
        or sum(class_weights.values()) <= 0
    ):
        raise ValueError("Invalid road-class weights.")

    class_names = list(ROAD_CLASSES)
    total_weight = sum(class_weights.values())
    probabilities = [
        class_weights[name] / total_weight
        for name in class_names
    ]

    network = RoadNetwork()
    for i in range(num_nodes):
        network.add_node(f"N{i:04d}")

    node_ids = sorted(network.nodes)
    candidate_edges = set()

    for i in range(num_nodes - 1):
        candidate_edges.add((node_ids[i], node_ids[i + 1]))
    all_possible_edges = [
        (node_ids[i], node_ids[j])
        for i in range(num_nodes)
        for j in range(i + 1, num_nodes)
    ]

    remaining_edges = list(
        set(all_possible_edges) - candidate_edges
    )
    rng.shuffle(remaining_edges)

    needed = num_segments - len(candidate_edges)
    candidate_edges.update(remaining_edges[:needed])
    sorted_edges = sorted(candidate_edges)

    for index, (start, end) in enumerate(sorted_edges, start=1):
        road_class = rng.choices(
            class_names, weights=probabilities, k=1
        )[0]

        settings = ROAD_CLASSES[road_class]
        low, high = settings["roughness_range"]

        # Hidden simulation ground truth.
        true_roughness = rng.uniform(low, high)
        # Imperfect initial estimate.
        initial_roughness = (
            settings["initial_estimate"]
            + rng.uniform(-initial_noise, initial_noise)
        )
        initial_roughness = max(
            0.0, min(1.0, initial_roughness))
        # Example length: 0.5 to 10 km.
        length_km = round(rng.uniform(0.5, 10.0), 2)

        segment = RoadSegment(
            segment_id=f"R{index:05d}",
            start_node=start,
            end_node=end,
            road_class=road_class,
            length_km=length_km,
            initial_roughness=initial_roughness,
            true_roughness=true_roughness,
            speed_limit_kmph=settings["speed_limit_kmph"],
        )

        network.add_segment(segment)

    return network
if __name__ == "__main__":
    network = generate_road_network(
        num_nodes=50,
        num_segments=80,
        seed=42,
    )

    print("NETWORK SUMMARY")
    print(network.summary())

    print("\nFIRST FIVE ROAD SEGMENTS")

    for segment in list(network.segments.values())[:5]:
        print(segment)

    print("\nNEIGHBORS OF N0000")
    print(network.get_neighbors("N0000"))