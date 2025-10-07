"""
This module contains the LLM-based Telemetry Analysis Engine. It uses a Large Language
Model (LLM) to diagnose problems from processed telemetry data and generates a
structured, human-readable report.
"""
import sys
import asyncio
import numpy as np
import matplotlib.pyplot as plt
from data_processing import SessionData, LoadSetup, LapTelemertyData
from scipy.fft import rfft, rfftfreq

# ************************ Helper Functions ************************
def remove_outliers(lap_times:list, lower_pct:int=1, upper_pct:int=95) -> list:
    lower = np.percentile(lap_times, lower_pct)
    upper = np.percentile(lap_times, upper_pct)
    return [i for i, lap_time in enumerate(lap_times) if lower <= lap_time <= upper]

def detect_hardstop(bs_up, bs_dw, bs_froce, bs_rate, damper_vel):
    F_th  = 1500.0   # N   bumpstop force threshold
    alpha = 100.0    # N/s minimum bumpstop force-rate
    return (
        ((bs_up.astype(bool) | bs_dw.astype(bool)) == 1) & 
        (bs_froce >= F_th) & (np.abs(damper_vel) <= 1.0) & (bs_rate >= alpha)
    )

def get_freq_match(chann_list: list) -> dict:
    min_freq = min([chann.freq for chann in chann_list])
    min_len = min([chann.data.size for chann in chann_list])
    return_data = []
    for chann in chann_list:
        data = chann.data
        freq = chann.freq
        if freq != min_freq:
            index = freq // min_freq
            return_data.append(data[::index][:min_len])
        else:
            return_data.append(data)

    return return_data

# ************************ Analysis Engine ************************
class SetupEngineer:
    def __init__(self, ld_file, setup_file):
        self.session = SessionData(ld_file, LoadSetup(setup_file))
        valid_id_list = remove_outliers(self.session.lap_times)
        self.lap_data = np.array([self.session.lap_data[id] for id in valid_id_list])
        self.best_lap_idx = np.argmin(self.session.lap_times)

        # get track info
        best_lap = self.get_best_lap_data()
        self.track_x, self.track_y, self.track_freq = best_lap.gen_track()
        # self.plot_track(best_lap)
        # self.plot_damper_histogram(best_lap)

        # TODO give them a better function discrption
        # self.eval_long_run()
        # self.eval_tire_load(best_lap)
        # self.eval_suspension(best_lap)
        # self.eval_damper(best_lap)
        # self.eval_rake(best_lap)
        # self.eval_bumpstop(best_lap)
        # self.eval_cornering_performance(best_lap)
        # self.eval_oversteer(best_lap)
        
    def get_processed_lap_time(self):
        return {i : lap.time for i, lap in enumerate(self.lap_data)}

    def get_lap(self, lap_idx):
        return self.lap_data[lap_idx]
    
    def get_best_lap_data(self):
        return self.lap_data[self.best_lap_idx]
    
    def eval_long_run(self):
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
                - 'ride_height_info': dictionary with lists of per-lap metrics
                    - 'ride_height_avg (mm)': average right hight per lap.
                    - 'ride_height_max (mm)': maximum right hight per lap.
                    - 'ride_height_std (mm)': standard deviation of right hight per lap.
        """
        tire_temp_info = {"tire_temp_avg": [], "tire_temp_max": [], "tire_temp_std": []}
        tire_psi_info = {"tire_psi_avg": [], "tire_psi_max": [], "tire_psi_std": []}
        brake_temp_info = {"brake_temp_avg": [], "brake_temp_max": [], "brake_temp_std": []}
        ride_height_info = {"ride_height_avg": [], "ride_height_max": [], "ride_height_std": []}
        for lap in self.lap_data:
            # Tire temp
            tire_temp_avg = {
                "lf" : np.mean(lap.channs["TYRE_TAIR_LF"].data),
                "rf" : np.mean(lap.channs["TYRE_TAIR_RF"].data),
                "lr" : np.mean(lap.channs["TYRE_TAIR_LR"].data),
                "rr" : np.mean(lap.channs["TYRE_TAIR_RR"].data),
            }
            tire_temp_max = {
                "lf" : np.max(lap.channs["TYRE_TAIR_LF"].data),
                "rf" : np.max(lap.channs["TYRE_TAIR_RF"].data),
                "lr" : np.max(lap.channs["TYRE_TAIR_LR"].data),
                "rr" : np.max(lap.channs["TYRE_TAIR_RR"].data),
            }
            tire_temp_std = {
                "lf" : np.std(lap.channs["TYRE_TAIR_LF"].data),
                "rf" : np.std(lap.channs["TYRE_TAIR_RF"].data),
                "lr" : np.std(lap.channs["TYRE_TAIR_LR"].data),
                "rr" : np.std(lap.channs["TYRE_TAIR_RR"].data),
            }
            tire_temp_info["tire_temp_avg"].append(tire_temp_avg)
            tire_temp_info["tire_temp_max"].append(tire_temp_max)
            tire_temp_info["tire_temp_std"].append(tire_temp_std)

            # Tire psi
            tire_psi_avg = {
                "lf" : np.mean(lap.channs["TYRE_PRESS_LF"].data),
                "rf" : np.mean(lap.channs["TYRE_PRESS_RF"].data),
                "lr" : np.mean(lap.channs["TYRE_PRESS_LR"].data),
                "rr" : np.mean(lap.channs["TYRE_PRESS_RR"].data),
            }
            tire_psi_max = {
                "lf" : np.max(lap.channs["TYRE_PRESS_LF"].data),
                "rf" : np.max(lap.channs["TYRE_PRESS_RF"].data),
                "lr" : np.max(lap.channs["TYRE_PRESS_LR"].data),
                "rr" : np.max(lap.channs["TYRE_PRESS_RR"].data),
            }
            tire_psi_std = {
                "lf" : np.std(lap.channs["TYRE_PRESS_LF"].data),
                "rf" : np.std(lap.channs["TYRE_PRESS_RF"].data),
                "lr" : np.std(lap.channs["TYRE_PRESS_LR"].data),
                "rr" : np.std(lap.channs["TYRE_PRESS_RR"].data),
            }
            tire_psi_info["tire_psi_avg"].append(tire_psi_avg)
            tire_psi_info["tire_psi_max"].append(tire_psi_max)
            tire_psi_info["tire_psi_std"].append(tire_psi_std)

            # Brake temp
            brake_temp_avg = {
                "lf" : np.mean(lap.channs["BRAKE_TEMP_LF"].data),
                "rf" : np.mean(lap.channs["BRAKE_TEMP_RF"].data),
                "lr" : np.mean(lap.channs["BRAKE_TEMP_LR"].data),
                "rr" : np.mean(lap.channs["BRAKE_TEMP_RR"].data),
            }
            brake_temp_max = {
                "lf" : np.max(lap.channs["BRAKE_TEMP_LF"].data),
                "rf" : np.max(lap.channs["BRAKE_TEMP_RF"].data),
                "lr" : np.max(lap.channs["BRAKE_TEMP_LR"].data),
                "rr" : np.max(lap.channs["BRAKE_TEMP_RR"].data),
            }
            brake_temp_std = {
                "lf" : np.std(lap.channs["BRAKE_TEMP_LF"].data),
                "rf" : np.std(lap.channs["BRAKE_TEMP_RF"].data),
                "lr" : np.std(lap.channs["BRAKE_TEMP_LR"].data),
                "rr" : np.std(lap.channs["BRAKE_TEMP_RR"].data),
            }
            brake_temp_info["brake_temp_avg"].append(brake_temp_avg)
            brake_temp_info["brake_temp_max"].append(brake_temp_max)
            brake_temp_info["brake_temp_std"].append(brake_temp_std)

            # Ride hight
            ride_height_avg = {
                "lf" : np.mean(lap.channs["ride_height_lf"].data),
                "rf" : np.mean(lap.channs["ride_height_rf"].data),
                "lr" : np.mean(lap.channs["ride_height_lr"].data),
                "rr" : np.mean(lap.channs["ride_height_rr"].data),
            }
            ride_height_max = {
                "lf" : np.max(lap.channs["ride_height_lf"].data),
                "rf" : np.max(lap.channs["ride_height_rf"].data),
                "lr" : np.max(lap.channs["ride_height_lr"].data),
                "rr" : np.max(lap.channs["ride_height_rr"].data),
            }
            ride_height_std = {
                "lf" : np.std(lap.channs["ride_height_lf"].data),
                "rf" : np.std(lap.channs["ride_height_rf"].data),
                "lr" : np.std(lap.channs["ride_height_lr"].data),
                "rr" : np.std(lap.channs["ride_height_rr"].data),
            }
            ride_height_info["ride_height_avg"].append(ride_height_avg)
            ride_height_info["ride_height_max"].append(ride_height_max)
            ride_height_info["ride_height_std"].append(ride_height_std)
    
        return {
            "tire_temp_info": tire_temp_info,
            "tire_psi_info": tire_psi_info,
            "brake_temp_info": brake_temp_info,
            "ride_height_info": ride_height_info
        }
    
    def eval_tire_load(self, lap: LapTelemertyData) -> dict:
        """
        Analyze tire load metrics across the entire lap, including average and maximum
        loads for each tire, as well as front-to-rear and left-to-right load distribution.

        Returns:
            A dictionary containing lap-level metrics:
                - 'avg_load_wheel (N)': average load for each tire ('avg_load_lf', 'avg_load_rf', etc.).
                - 'max_load_wheel (N)': average load for each tire ('max_load_lf', 'max_load_rf', etc.).
                - 'front_to_rear_load_split (N)': ratio of front to total load.
                - 'left_to_right_load_split (N)': ratio of left to total load.
        """
        tire_load = {
            "load_lf": lap.channs["tire_load_lf"].data,
            "load_rf": lap.channs["tire_load_rf"].data,
            "load_lr": lap.channs["tire_load_lr"].data,
            "load_rr": lap.channs["tire_load_rr"].data,
        }

        avg_loads = {f"avg_{wheel}": np.mean(data) for wheel, data in tire_load.items()}
        max_loads = {f"max_{wheel}": np.max(data) for wheel, data in tire_load.items()}

        avg_front_load = (avg_loads["avg_load_lf"] + avg_loads["avg_load_rf"])
        avg_rear_load = (avg_loads["avg_load_lr"] + avg_loads["avg_load_rr"])
        front_to_rear_split = avg_front_load / (avg_front_load + avg_rear_load)

        avg_left_load = (avg_loads["avg_load_lf"] + avg_loads["avg_load_lr"])
        avg_right_load = (avg_loads["avg_load_rf"] + avg_loads["avg_load_rr"])
        left_to_right_split = avg_left_load / (avg_left_load + avg_right_load)

        metrics = {
            **avg_loads,
            **max_loads,
            "front_to_rear_load_split": front_to_rear_split,
            "left_to_right_load_split": left_to_right_split
        }
        
        return metrics
    
    def eval_suspension(self, lap: LapTelemertyData) -> dict:
        """
        Summarize suspension travel statistics for each wheel across the entire lap,
        including mean, standard deviation, and range.

        Returns:
            A dictionary containing lap-level suspension travel metrics for each wheel ('lf', 'rf', 'lr', 'rr'),
            where each key maps to a sub-dictionary with:
                - 'mean (mm)': average suspension travel
                - 'std (mm)': standard deviation of suspension travel
                - 'range (mm)': peak-to-peak suspension travel
        """
        suspension_data = {
            "lf": lap.channs["SUS_TRAVEL_LF"].data,
            "rf": lap.channs["SUS_TRAVEL_RF"].data,
            "lr": lap.channs["SUS_TRAVEL_LR"].data,
            "rr": lap.channs["SUS_TRAVEL_RR"].data,
        }
        
        travel_metrics = {}
        for wheel, data in suspension_data.items():
            travel_metrics[wheel] = {
                "mean": np.mean(data),
                "std": np.std(data),
                "range": np.ptp(data) # Peak-to-peak
            }
        
        return travel_metrics

    def eval_damper(self, lap: LapTelemertyData) -> dict:
        """
        Perform histogram-based frequency analysis of damper velocity for each wheel
        across the entire lap.

        Returns:
            A dictionary containing lap-level damper velocity metrics for each wheel ('lf', 'rf', 'lr', 'rr'),
            where each key maps to a sub-dictionary with:
                - 'freq (Hz)': histogram counts of damper velocity occurrences
                - 'damp_vel (mm/s)': corresponding damper velocity bin edges
        """
        damper_vel_data = {
            "lf": lap.channs["damper_vel_lf"].data,
            "rf": lap.channs["damper_vel_rf"].data,
            "lr": lap.channs["damper_vel_lr"].data,
            "rr": lap.channs["damper_vel_rr"].data,
        }

        bins = np.linspace(-300, 300, 10)
        damper_hist = {}
        for wheel, data in damper_vel_data.items():
            freq, damp_vel = np.histogram(data, bins=bins)
            damper_hist[wheel] = {"freq" : freq, "damp_vel" : damp_vel}
        
        return damper_hist

    def eval_rake(self, lap: LapTelemertyData, accelerate_threshold: float=0.5, braking_threshold: float=-1.0) -> dict:
        """
        Analyze the car's rake behavior across the lap, including overall, braking,
        and acceleration-specific average angles.

        Returns:
            A dictionary containing lap-level rake metrics:
                - 'overall_avg_rake (rad)': average rake angle across the lap
                - 'avg_rake_under_braking (rad)': average rake angle during braking
                - 'avg_rake_under_accelerating (rad)': average rake angle during acceleration
                - 'rake_data (rad)': full array of rake measurements
        """
        rake_data, g_lon_data = get_freq_match([lap.channs["rake_angle"], lap.channs["G_LON"]])
        
        avg_rake = np.mean(rake_data)
        braking_rake = rake_data[g_lon_data < braking_threshold]
        accelerate_rake = rake_data[g_lon_data > accelerate_threshold]
        avg_braking_rake = np.mean(braking_rake) if braking_rake.size > 0 else 0
        avg_accelerate_rake = np.mean(accelerate_rake) if accelerate_rake.size > 0 else 0
        
        return {
            "overall_avg_rake": avg_rake,
            "avg_rake_under_braking": avg_braking_rake,
            "avg_rake_under_accelerating": avg_accelerate_rake,
            "rake_data" : rake_data
        }

    def eval_bumpstop(self, lap: LapTelemertyData) -> dict:
        """
        Evaluate bump stop engagement and detect hard stops for each corner in the lap.

        Returns:
            A dictionary containing corner-level bump stop metrics, with corner indices as keys:
                - 'bumpstop front ratio': ratio of front wheels hitting bump stops
                - 'bumpstop rear ratio': ratio of rear wheels hitting bump stops
                - 'hardstop': dictionary with keys 'lf', 'rf', 'lr', 'rr' boolean indicating hard stop events per wheel
        """
        turn_list = lap.all_turns()
        corner_metrics = {}
        for turn_id, turn in enumerate(turn_list):
            bsu_lf = turn['BUMPSTOPUP_RIDE_LF'].data
            bsu_rf = turn['BUMPSTOPUP_RIDE_RF'].data
            bsu_lr = turn['BUMPSTOPUP_RIDE_LR'].data
            bsu_rr = turn['BUMPSTOPUP_RIDE_RR'].data

            bsd_lf = turn['BUMPSTOPDN_RIDE_LF'].data
            bsd_rf = turn['BUMPSTOPDN_RIDE_RF'].data
            bsd_lr = turn['BUMPSTOPDN_RIDE_LR'].data
            bsd_rr = turn['BUMPSTOPDN_RIDE_RR'].data

            bsf_lf = turn['BUMPSTOP_FORCE_LF'].data
            bsf_rf = turn['BUMPSTOP_FORCE_RF'].data
            bsf_lr = turn['BUMPSTOP_FORCE_LR'].data
            bsf_rr = turn['BUMPSTOP_FORCE_RR'].data

            bsr_lf = turn['bs_force_rate_lf'].data
            bsr_rf = turn['bs_force_rate_rf'].data
            bsr_lr = turn['bs_force_rate_lr'].data
            bsr_rr = turn['bs_force_rate_rr'].data

            dmv_lf = turn['damper_vel_lf'].data
            dmv_rf = turn['damper_vel_rf'].data
            dmv_lr = turn['damper_vel_lr'].data
            dmv_rr = turn['damper_vel_rr'].data

            # hard stop detection
            hardstop_lf = detect_hardstop(bsu_lf, bsd_lf, bsf_lf, bsr_lf, dmv_lf)
            hardstop_rf = detect_hardstop(bsu_rf, bsd_rf, bsf_rf, bsr_rf, dmv_rf)
            hardstop_lr = detect_hardstop(bsu_lr, bsd_lr, bsf_lr, bsr_lr, dmv_lr)
            hardstop_rr = detect_hardstop(bsu_rr, bsd_rr, bsf_rr, bsr_rr, dmv_rr)
            
            hardstop = {
                "lf": hardstop_lf,
                "rf": hardstop_rf,
                "lr": hardstop_lr,
                "rr": hardstop_rr,
            }

            f_bsu_comb =  np.logical_or.reduce([bsu_lf, bsu_rf])
            r_bsu_comb =  np.logical_or.reduce([bsu_lr, bsu_rr])
            f_ratio = sum(f_bsu_comb) / len(f_bsu_comb)
            r_ratio = sum(r_bsu_comb) / len(r_bsu_comb)

            corner_data = {
                "bumpstop front ratio": f_ratio,
                "bumpstop rear ratio": r_ratio,
                "hardstop": hardstop
            }
            corner_metrics[turn_id + 1] = corner_data

        return corner_metrics
    
    def eval_cornering_performance(self, lap: LapTelemertyData) -> dict:
        """
        Analyze key cornering metrics for each identified turn in the lap,
        including apex speed, lateral and longitudinal g-forces, and corner duration.

        Returns:
            A dictionary containing corner-level metrics, with corner indices as keys:
                - 'min_speed_at_apex_kmh': minimum speed at apex in km/h
                - 'avg_lateral_g': average lateral g-force
                - 'avg_longitudinal_g': average longitudinal g-force
                - 'duration_s': duration of the corner in seconds
        """
        turn_list = lap.all_turns()
        data_freq = lap.channs["SPEED"].freq
        corner_metrics = {}
        for turn_id, turn in enumerate(turn_list):
            turn_speed = turn["SPEED"].data
            turn_g_lat = turn["g_lat"].data
            turn_g_lon = turn["g_lon"].data
            
            # Calculate the metrics
            min_speed = np.min(turn_speed)
            avg_g_lat = np.mean(turn_g_lat)
            avg_g_lon = np.mean(turn_g_lon)
            duration = turn_speed.size / data_freq

            # Store the metrics in a dictionary
            corner_data = {
                "min_speed_at_apex_kmh": min_speed * 3.6, # Convert from m/s to km/h
                "avg_lateral_g": avg_g_lat,
                "avg_longitudinal_g": avg_g_lon,
                "duration_s": duration
            }
            corner_metrics[turn_id + 1] = corner_data
        
        return corner_metrics

    def eval_oversteer(self, lap: LapTelemertyData) -> dict:
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
        turn_list = lap.all_entry()
        entry_metrics = {}
        for turn_id, turn in enumerate(turn_list):
            oversteer = turn['oversteer'].data
            slip_angle_f = turn['slip_angle_f'].data
            slip_angle_r = turn['slip_angle_r'].data

            corner_data = {
                "oversteer" : oversteer,
                "slip_angle_f" : slip_angle_f,
                "slip_angle_r" : slip_angle_r,
            }
            entry_metrics[turn_id + 1] = corner_data
        
        turn_list = lap.all_exit()
        exit_metrics = {}
        for turn_id, turn in enumerate(turn_list):
            oversteer = turn['oversteer'].data
            slip_angle_f = turn['slip_angle_f'].data
            slip_angle_r = turn['slip_angle_r'].data

            corner_data = {
                "oversteer" : oversteer,
                "slip_angle_f" : slip_angle_f,
                "slip_angle_r" : slip_angle_r,
            }
            exit_metrics[turn_id + 1] = corner_data
        
        return {"entry_metrics" : entry_metrics, "exit_metrics" : exit_metrics}

    def plot_track(self, lap: LapTelemertyData, plot_turn:bool=True, plot_bs:bool=False):
        plt.figure(figsize=(20, 20))
        plt.axis('equal')
        plt.plot(self.track_x, self.track_y, color="black", linewidth=15)

        if plot_turn:
            turns_limit = lap.turns_limits[self.track_freq]
            for start, apex, exit in turns_limit:
                plt.plot(self.track_x[start:apex], self.track_y[start:apex], color="red", linewidth=15)
                plt.plot(self.track_x[apex:exit], self.track_y[apex:exit], color="green", linewidth=15)

        plt.show()
    
    def plot_damper_histogram(self, lap: LapTelemertyData):
        damper_data = {
            'Front-Left': lap.channs["damper_vel_lf"].data,
            'Front-Right': lap.channs["damper_vel_rf"].data,
            'Rear-Left': lap.channs["damper_vel_lr"].data,
            'Rear-Right': lap.channs["damper_vel_rr"].data
        }

        filtered_data = [np.atleast_1d(d) for d in damper_data.values() if np.array(d).size > 0]
        plt.style.use('dark_background')
        fig, axs = plt.subplots(2, 2, figsize=(12, 10))
        axs_flat = axs.flatten()

        # Determine the bins based on the full range of data for better consistency
        all_data = np.concatenate(filtered_data)
        
        # Use percentiles to set a more robust x-axis range, filtering outliers
        x_min = np.percentile(all_data, 1)
        x_max = np.percentile(all_data, 99)
        bins = np.linspace(x_min, x_max, 50)
        
        # Plot histogram for each wheel provided
        max_freq = 0
        for idx, (wheel, data) in enumerate(damper_data.items()):
            ax = axs_flat[idx]
            if np.array(data).size > 0:
                # Generate the histogram counts and bins to find the max frequency
                counts, _ = np.histogram(data, bins=bins)
                max_freq = max(max_freq, counts.max())
                
                ax.hist(data, bins=bins, alpha=0.7, color='cyan', histtype='stepfilled')
                ax.set_title(f'Damper Velocity: {wheel}', fontsize=14, color='white')
                ax.set_xlabel('Damper Velocity (m/s)', fontsize=10, color='white')
                ax.set_ylabel('Frequency', fontsize=10, color='white')
                ax.axvline(0, color='white', linestyle='--', linewidth=1.5)
                ax.grid(True, linestyle=':', alpha=0.6)
                
                # Customize ticks and spines for a clean look
                ax.tick_params(axis='x', colors='white')
                ax.tick_params(axis='y', colors='white')
                for spine in ax.spines.values():
                    spine.set_edgecolor('white')
        
        # Now set consistent x and y limits for all subplots
        for ax in axs_flat:
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(0, max_freq * 1.3)
        
        # Hide any unused subplots
        for i in range(len(damper_data), len(axs_flat)):
            axs_flat[i].axis('off')

        fig.suptitle('Damper Velocity Histograms', fontsize=18, color='white', y=0.95)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

def main():
    if len(sys.argv)!=3:
        print("Usage: data_processing.py /path/to/file.ld /path/to/setup.json")
        sys.exit(1)

    analyzer = SetupEngineer(sys.argv[1], sys.argv[2])
    
# Example Usage
if __name__ == "__main__":
    main()
