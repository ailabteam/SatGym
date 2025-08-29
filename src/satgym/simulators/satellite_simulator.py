# src/satgym/simulators/satellite_simulator.py
import os
import contextlib
import logging
from pathlib import Path
from typing import List, Dict, Any

import h5py
import networkx as nx
import numpy as np
from skyfield.api import load, wgs84

try:
    # Import với 'src.' vì đó là cấu trúc gốc của StarPerf package
    from src.constellation_generation.by_XML import constellation_configuration
    from src.XML_constellation.constellation_connectivity import connectivity_mode_plugin_manager
    from src.XML_constellation.constellation_entity.user import user as User
    from src.XML_constellation.constellation_entity.satellite import satellite as Satellite
    from src.XML_constellation.constellation_evaluation.exists_ISL.delay import distance_between_satellite_and_user
except ImportError as e:
    logging.critical(f"Failed to import from StarPerf. Ensure sys.path is correctly set by the calling script. Error: {e}")
    raise

logger = logging.getLogger(__name__)

class SatelliteSimulator:
    """Provides a clean API to interact with the StarPerf satellite simulation."""
    def __init__(self, starperf_path: Path, config: Dict[str, Any]):
        self.starperf_dir = starperf_path
        self.config = config
        self.ts = load.timescale()

        self.constellation = self._initialize_constellation()
        self.shell = self.constellation.shells[0]
        self.num_satellites = self.shell.number_of_satellites
        self.sat_id_map: Dict[int, Satellite] = {sat.id: sat for orbit in self.shell.orbits for sat in orbit.satellites}

        sample_sat = self.sat_id_map[1]
        self.simulation_steps = len(sample_sat.longitude)
        logger.info(f"SatelliteSimulator initialized with {self.simulation_steps} time steps.")

        self._graph_cache: Dict[int, nx.Graph] = {}

    @contextlib.contextmanager
    def _as_current_dir(self):
        prev_cwd = Path.cwd()
        os.chdir(self.starperf_dir)
        try: yield
        finally: os.chdir(prev_cwd)

    def _initialize_constellation(self) -> Any:
        h5_filepath = self.starperf_dir / "data" / "XML_constellation" / f"{self.config['constellation_name']}.h5"
        with self._as_current_dir():
            logger.info(f"Loading constellation: {self.config['constellation_name']}...")
            constellation = constellation_configuration.constellation_configuration(
                dT=self.config['dT'], constellation_name=self.config['constellation_name']
            )
            logger.info(f"Augmenting HDF5 file to ensure '/delay' group exists...")
            with h5py.File(h5_filepath, 'a') as file:
                if 'delay' not in file:
                    delay_group = file.create_group('delay')
                    if 'position' in file:
                        for shell_name in file['position'].keys():
                            if shell_name not in delay_group:
                                delay_group.create_group(shell_name)
            logger.info("Building connectivity and writing link data to HDF5...")
            connection_manager = connectivity_mode_plugin_manager.connectivity_mode_plugin_manager()
            connection_manager.execute_connection_policy(constellation=constellation, dT=self.config['dT'])
        return constellation

    # ... (Dán các hàm API public và helper private khác vào đây)
    # (get_network_graph, get_satellite_position, _build_network_graph, ...)

    def get_network_graph(self, time_step: int) -> nx.Graph:
        """
        Returns a cached or newly built NetworkX graph for a specific time step.
        The graph contains nodes (satellite IDs) and edges with 'delay' attribute.
        """
        if time_step in self._graph_cache:
            return self._graph_cache[time_step]
        graph = self._build_network_graph(time_step)
        self._graph_cache[time_step] = graph
        return graph

    def get_satellite_position(self, sat: Satellite, time_step: int) -> np.ndarray:
        """Returns satellite position as a NumPy array [lon, lat, alt_km]."""
        safe_idx = min(time_step, self.simulation_steps) - 1
        alt = sat.altitude[safe_idx] if isinstance(sat.altitude, list) else sat.altitude
        return np.array([sat.longitude[safe_idx], sat.latitude[safe_idx], alt])
    
    def find_nearest_satellite(self, user: User, time_step: int) -> Satellite:
        """Finds the closest satellite to a ground user."""
        return min(
            self.sat_id_map.values(),
            key=lambda sat: distance_between_satellite_and_user(user, sat, time_step)
        )

    def get_elevation_angle(self, user: User, satellite: Satellite, time_step: int) -> float:
        """Calculates the elevation angle of a satellite in degrees."""
        sat_skyfield_obj = satellite.true_satellite 
        time = self.ts.utc(2023, 10, 1, 0, 0, (time_step - 1) * self.config['dT'])
        user_location = wgs84.latlon(user.latitude, user.longitude)
        difference = sat_skyfield_obj - user_location
        topocentric = difference.at(time)
        alt, az, distance = topocentric.altaz()
        return alt.degrees

    # --- Private Helper Methods ---

    def _build_network_graph(self, time_step: int) -> nx.Graph:
        """Constructs the network graph for a given time step from ISL data."""
        G = nx.Graph()
        G.add_nodes_from(self.sat_id_map.keys())
        for sat_id, current_sat in self.sat_id_map.items():
            if not hasattr(current_sat, 'ISL'): continue
            for isl_link in current_sat.ISL:
                neighbor_id = isl_link.satellite2 if isl_link.satellite1 == sat_id else isl_link.satellite1
                if sat_id < neighbor_id:
                    neighbor_sat = self.sat_id_map[neighbor_id]
                    dist = self._distance_between_sats(current_sat, neighbor_sat, time_step)
                    delay = dist / 299792.458 # Speed of light in km/s
                    G.add_edge(sat_id, neighbor_id, delay=delay)
        return G
            
    def _distance_between_sats(self, sat1: Satellite, sat2: Satellite, time_step: int) -> float:
        """Calculates the great-circle distance in kilometers between two satellites."""
        safe_idx = min(time_step, self.simulation_steps) - 1
        lon1, lat1 = sat1.longitude[safe_idx], sat1.latitude[safe_idx]
        lon2, lat2 = sat2.longitude[safe_idx], sat2.latitude[safe_idx]
        lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
        dlon, dlat = lon2 - lon1, lat2 - lat1
        a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
        c = 2 * np.arcsin(np.sqrt(a))
        return 6371 * c
