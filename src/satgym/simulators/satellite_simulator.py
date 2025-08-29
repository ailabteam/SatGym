# src/satgym/simulators/satellite_simulator.py

import os
import sys
import contextlib
import logging
from pathlib import Path
from typing import List, Dict, Any
from itertools import islice

# --- KHỐI IMPORT AN TOÀN VÀ TỰ LỰC ---
# Kỹ thuật này đảm bảo module có thể import dependency của nó
# mà không cần dựa vào script chạy bên ngoài để sửa sys.path.

# 1. Xác định đường dẫn gốc của StarPerf một cách đáng tin cậy
_STARPERF_ROOT = Path(__file__).resolve().parent.parent.parent / "deps" / "StarPerf_Simulator"

# 2. Tạm thời thêm thư mục gốc của StarPerf vào sys.path
if str(_STARPERF_ROOT) not in sys.path:
    sys.path.insert(0, str(_STARPERF_ROOT))
    _PATH_ADDED = True
else:
    _PATH_ADDED = False

try:
    # 3. Thực hiện các import cần thiết.
    # Bây giờ, 'from src...' sẽ hoạt động vì Python sẽ tìm thấy thư mục 'src'
    # bên trong đường dẫn gốc mà chúng ta đã thêm vào.
    import h5py
    import networkx as nx
    import numpy as np
    from skyfield.api import load, wgs84
    from src.constellation_generation.by_XML import constellation_configuration
    from src.XML_constellation.constellation_connectivity import connectivity_mode_plugin_manager
    from src.XML_constellation.constellation_entity.user import user as User
    from src.XML_constellation.constellation_entity.satellite import satellite as Satellite
    from src.XML_constellation.constellation_evaluation.exists_ISL.delay import distance_between_satellite_and_user

finally:
    # 4. Quan trọng: Dọn dẹp sys.path sau khi đã import xong.
    # Điều này ngăn chặn các tác dụng phụ không mong muốn cho các module khác.
    if _PATH_ADDED:
        sys.path.pop(0)

# ----------------------------------------

logger = logging.getLogger(__name__)

class SatelliteSimulator:
    """
    Provides a clean API to interact with the StarPerf satellite simulation.
    Handles the creation and state management of the satellite constellation.
    """
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
        """Context manager to temporarily change the working directory for StarPerf calls."""
        prev_cwd = Path.cwd()
        os.chdir(self.starperf_dir)
        try:
            yield
        finally:
            os.chdir(prev_cwd)

    def _initialize_constellation(self) -> Any:
        """
        Initializes StarPerf, creating/augmenting the HDF5 data file.
        This is a heavy operation that runs only once.
        """
        h5_filepath = self.starperf_dir / "data" / "XML_constellation" / f"{self.config['constellation_name']}.h5"
        
        with self._as_current_dir():
            logger.info(f"Loading constellation: {self.config['constellation_name']}...")
            constellation = constellation_configuration.constellation_configuration(
                dT=self.config['dT'], constellation_name=self.config['constellation_name']
            )
            
            logger.info(f"Augmenting HDF5 file to ensure '/delay' group exists...")
            try:
                with h5py.File(h5_filepath, 'a') as file:
                    if 'delay' not in file:
                        delay_group = file.create_group('delay')
                        if 'position' in file:
                            for shell_name in file['position'].keys():
                                if shell_name not in delay_group:
                                    delay_group.create_group(shell_name)
            except Exception as e:
                logger.error(f"Failed to augment HDF5 file: {e}", exc_info=True)
                raise
            
            logger.info("Building connectivity and writing link data to HDF5...")
            connection_manager = connectivity_mode_plugin_manager.connectivity_mode_plugin_manager()
            connection_manager.execute_connection_policy(constellation=constellation, dT=self.config['dT'])
            
        return constellation

    # --- Public API Methods ---

    def get_network_graph(self, time_step: int) -> nx.Graph:
        """
        Returns a cached or newly built NetworkX graph for a specific time step.
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
                    delay = dist / 299792.458
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
        return 6731 * c # Bán kính Trái Đất ~ 6371 km
