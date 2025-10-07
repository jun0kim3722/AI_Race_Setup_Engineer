import os
from datetime import datetime

from mcp.server.fastmcp import FastMCP
from analysis_engine import SetupEngineer
from dotenv import load_dotenv

load_dotenv(".env")

# Create an MCP server
mcp = FastMCP(
    name="Race Engineer Server",
    host="0.0.0.0",  # only used for SSE transport (localhost)
    port=8050,  # only used for SSE transport (set this to any port)
)

def validate_lap_index(lap_idx: int) -> str:
    """Helper function to validate the requested lap index."""
    if lap_idx < 0:
        return f"Error: lap_idx must be a non-negative integer. Received {lap_idx}."
    if lap_idx >= len(se.lap_data):
        return f"Error: lap_idx {lap_idx} is out of bounds. Only {len(se.lap_data)} laps available (0 to {len(se.lap_data) - 1})."
    return ""

# ************************************ Initial Tools ************************************
@mcp.tool()
def get_telemetry_file_info() -> str:
    """
    List all the session files and info. This tool is for user to choose correct session to analyze.
    **Do not call this.**
    """
    files_info = []
    files = os.listdir(os.getenv("TELEMETRY_DIR"))
    for f in files:
        base, ext = os.path.splitext(f)
        if ext in (".ld", ".ldx") and base not in files_info:
            files_info.append(base)

    global telemetry_files
    telemetry_files= []
    for file in files_info:
        try:
            track, car, num_laps, date_str, time_str = file.split("-")
            dt = datetime.strptime(f"{date_str} {time_str}", "%Y.%m.%d %H.%M.%S")
            telemetry_files.append((dt, track, car, date_str, time_str, file))
        except ValueError:
            continue

    telemetry_files.sort(key=lambda x: x[0], reverse=True)

    return_str = f"{'No.':<4} {'Track Name':<20} {'Car Name':<20} {'Date':<12} {'Time':<8}\n"
    for i, (_, track, car, date_str, time_str, _) in enumerate(telemetry_files, start=1):
        return_str += f"{i:<4} {track:<20} {car:<20} {date_str:<12} {time_str:<8}\n"
    
    return return_str

@mcp.tool()
def get_setup_file_info(ld_ldx_idx: int):
    """
    Reads the setup files. This tool is for user to choose correct setup to analyze.
    **Do not call this.**
    """
    dir = os.getenv("SETUP_DIR")
    car = telemetry_files[ld_ldx_idx][2]
    track = telemetry_files[ld_ldx_idx][1]
    print(f"READING: {dir}/{car}/{track}")

    global setup_files
    try:
        setup_files = os.listdir(f"{dir}/{car}/{track}")
        setup_files.sort(key=lambda f: os.path.getmtime(os.path.join(f"{dir}/{car}/{track}", f)), reverse=True)
    except:
        raise FileNotFoundError(f"Setup file not found. Dir: {dir}/{car}/{track}")

    return_str = f"{'No.':<4} {'Setup Name':<20}\n"
    for i, file_name in enumerate(setup_files, start=1):
        return_str += f"{i:<4} {file_name}\n"
    
    return return_str

@mcp.tool()
def get_session(ld_ldx_idx: int, setup_idx: int) -> dict:
    """
    Get session data, read telemetry. This tool is to load all the data user wish to analyze.
    **Do not call this.**

    Return:
        File data, Track Name, Vehicle Name, Vehicle Weight, Number of laps 
    """
    try:
        telemetry_dir = os.getenv("TELEMETRY_DIR")
        ld_ldx_file = f"{telemetry_dir}/{telemetry_files[ld_ldx_idx][5]}.ldx"

        setup_dir = os.getenv("SETUP_DIR")
        car = telemetry_files[ld_ldx_idx][2]
        track = telemetry_files[ld_ldx_idx][1]
        setup_file = f"{setup_dir}/{car}/{track}/{setup_files[setup_idx]}"

        global se
        se = SetupEngineer(ld_ldx_file, setup_file)
    
    except Exception as e:
        return f"Error: {str(e)} ld: {ld_ldx_file} setup: {setup_file}"
    
@mcp.tool()
def init_session_info() -> dict:
    """
    Get initial session info. This data is preloaded before user's query.
    """
    return se.session.get_session_info()


@mcp.tool()
def read_processed_lap_time() -> dict:
    """
    Use this to find a lap you want to analyize by looking at lap times.
    This contain laps from the session that excluded outliers.
    Contain low 1% to 95% of the lap times.

    Return:
        A dictionary with {lap idx : lap time}
    """

    return se.get_processed_lap_time()

# ************************************ Long Run Analysis Tools ************************************
@mcp.tool()
def eval_long_run() -> dict:
    """
    Evaluate long-run performance metrics for tires and brakes across all laps.
    For each lap, computes average, maximum, and standard deviation values of 
    tire temperatures, tire pressures, and brake temperatures for all four corners 
    (LF, RF, LR, RR).

    Returns:
        A dictionary containing lap-level metrics:
            - 'tire_temp_info': dictionary with lists of per-lap metrics
                - 'tire_temp_avg (°C)': average tire temperature per lap.
                - 'tire_temp_max (°C)': maximum tire temperature per lap.
                - 'tire_temp_std (°C)': standard deviation of tire temperature per lap.
            - 'tire_psi_info': dictionary with lists of per-lap metrics
                - 'tire_psi_avg (psi)': average tire pressure per lap.
                - 'tire_psi_max (psi)': maximum tire pressure per lap.
                - 'tire_psi_std (psi)': standard deviation of tire pressure per lap.
            - 'brake_temp_info': dictionary with lists of per-lap metrics
                - 'brake_temp_avg (°C)': average brake temperature per lap.
                - 'brake_temp_max (°C)': maximum brake temperature per lap.
                - 'brake_temp_std (°C)': standard deviation of brake temperature per lap.
            - 'ride_hight_info': dictionary with lists of per-lap metrics
                - 'ride_hight_avg (mm)': average right hight per lap.
                - 'ride_hight_max (mm)': maximum right hight per lap.
                - 'ride_hight_std (mm)': standard deviation of right hight per lap.
    """
    return se.eval_long_run()

# ************************************ Lap Analysis Tools ************************************
@mcp.tool()
def get_entire_lap_data(lap_idx: int) -> dict:
    """
    Retrieve complete telemetry data, freq, and unit for a specified lap.
    Do not call this unless you want entire lap data. Try to use other tools for specific tesks.

    Args:
        lap_idx (int): Index of the lap to analyze.

    Returns:
        A dictionary containing lap-level telemetry data for all channels:
            - Each key corresponds to a telemetry channel name.
            - Each value is a sub-dictionary with:
                - 'data': numpy array of channel values
                - 'freq': sampling frequency (Hz) of the channel
                - 'unit': unit of the give data
    """
    error_message = validate_lap_index(lap_idx)
    if error_message:
        return {"status": "error", "message": error_message}
    
    lap = se.get_lap(lap_idx)
    return {key: {"data" : chann.data, "feq" : chann.freq, "unit" : chann.unit} for key, chann in lap.channs.items()}

@mcp.tool()
def eval_tire_load(lap_idx: int) -> dict:
    """
    Analyze tire load metrics across the entire lap, including average and maximum
    loads for each tire, as well as front-to-rear and left-to-right load distribution.

    Args:
        Lap idx that you want to analys.
 
    Returns:
        A dictionary containing lap-level metrics:
            - 'avg_load_wheel (N)': average load for each tire ('avg_load_lf', 'avg_load_rf', etc.).
            - 'max_load_wheel (N)': average load for each tire ('max_load_lf', 'max_load_rf', etc.).
            - 'front_to_rear_load_split (N)': ratio of front to total load.
            - 'left_to_right_load_split (N)': ratio of left to total load.
    """
    error_message = validate_lap_index(lap_idx)
    if error_message:
        return {"status": "error", "message": error_message}

    lap_telemetry = se.get_lap(lap_idx)
    return se.eval_tire_load(lap_telemetry)

@mcp.tool()
def eval_suspension(lap_idx: int) -> dict:
    """
    Summarize suspension travel statistics for each wheel across the entire lap,
    including mean, standard deviation, and range.

    Args:
        Lap idx that you want to analys.
 
    Returns:
        A dictionary containing lap-level suspension travel metrics for each wheel ('lf', 'rf', 'lr', 'rr'),
        where each key maps to a sub-dictionary with:
            - 'mean (mm)': average suspension travel
            - 'std (mm)': standard deviation of suspension travel
            - 'range (mm)': peak-to-peak suspension travel
    """
    error_message = validate_lap_index(lap_idx)
    if error_message:
        return {"status": "error", "message": error_message}
    
    lap_telemetry = se.get_lap(lap_idx)
    return se.eval_suspension(lap_telemetry)

@mcp.tool()
def eval_damper(lap_idx: int) -> dict:
    """
    Perform histogram-based frequency analysis of damper velocity for each wheel
    across the entire lap.

    Args:
        Lap idx that you want to analys.
 
    Returns:
        A dictionary containing lap-level damper velocity metrics for each wheel ('lf', 'rf', 'lr', 'rr'),
        where each key maps to a sub-dictionary with:
            - 'freq (Hz)': histogram counts of damper velocity occurrences
            - 'damp_vel (mm/s)': corresponding damper velocity bin edges
    """
    error_message = validate_lap_index(lap_idx)
    if error_message:
        return {"status": "error", "message": error_message}

    lap_telemetry = se.get_lap(lap_idx)
    return se.eval_damper(lap_telemetry)

@mcp.tool()
def eval_rake(lap_idx: int) -> dict:
    """
    Analyze the car's rake behavior across the lap, including overall, braking,
    and acceleration-specific average angles.

    Args:
        Lap idx that you want to analys.
 
    Returns:
        A dictionary containing lap-level rake metrics:
            - 'overall_avg_rake (rad)': average rake angle across the lap
            - 'avg_rake_under_braking (rad)': average rake angle during braking
            - 'avg_rake_under_accelerating (rad)': average rake angle during acceleration
            - 'rake_data (rad)': full array of rake measurements
    """
    error_message = validate_lap_index(lap_idx)
    if error_message:
        return {"status": "error", "message": error_message}
    
    lap_telemetry = se.get_lap(lap_idx)
    return se.eval_rake(lap_telemetry)

# ************************************ Lap Corner Analysis Tools ************************************
@mcp.tool()
def eval_bumpstop(lap_idx: int) -> dict:
    """
    Evaluate bump stop engagement and detect hard stops for each corner in the lap.

    Args:
        Lap idx that you want to analys.
 
    Returns:
        A dictionary containing corner-level bump stop metrics, with corner indices as keys:
            - 'bumpstop front ratio': ratio of front wheels hitting bump stops
            - 'bumpstop rear ratio': ratio of rear wheels hitting bump stops
            - 'hardstop': dictionary with keys 'lf', 'rf', 'lr', 'rr' boolean indicating hard stop events per wheel
    """
    error_message = validate_lap_index(lap_idx)
    if error_message:
        return {"status": "error", "message": error_message}
    
    lap_telemetry = se.get_lap(lap_idx)
    return se.eval_bumpstop(lap_telemetry)

@mcp.tool()
def eval_cornering_performance(lap_idx: int) -> dict:
    """
    Analyze key cornering metrics for each identified turn in the lap,
    including apex speed, lateral and longitudinal g-forces, and corner duration.

    Args:
        Lap idx that you want to analys.
 
    Returns:
        A dictionary containing corner-level metrics, with corner indices as keys:
            - 'min_speed_at_apex_kmh': minimum speed at apex in km/h
            - 'avg_lateral_g': average lateral g-force
            - 'avg_longitudinal_g': average longitudinal g-force
            - 'duration_s': duration of the corner in seconds
    """
    error_message = validate_lap_index(lap_idx)
    if error_message:
        return {"status": "error", "message": error_message}
    
    lap_telemetry = se.get_lap(lap_idx)
    return se.eval_cornering_performance(lap_telemetry)

@mcp.tool()
def eval_oversteer(lap_idx: int) -> dict:
    """
    Compute oversteer and slip angle metrics for each corner in a lap.

    Args:
        lap (LapTelemertyData): Telemetry data for a single lap.

    Returns:
        dict: A dictionary with two main sections:
            - Entry metrics (per corner, keyed by corner index).
            - Exit metrics (per corner, keyed by corner index).

        Each corner entry contains:
            - 'oversteer' (float): Oversteer angle in radians
            (positive = oversteer, negative = understeer).
            - 'slip_angle_f' (float): Front wheel slip angle in radians
            (positive = slipping right, negative = slipping left).
            - 'slip_angle_r' (float): Rear wheel slip angle in radians
            (positive = slipping right, negative = slipping left).
    """
    error_message = validate_lap_index(lap_idx)
    if error_message:
        return {"status": "error", "message": error_message}
    
    lap_telemetry = se.get_lap(lap_idx)
    return se.eval_oversteer(lap_telemetry)

# Run the server
if __name__ == "__main__":
    mcp.run(transport="stdio")