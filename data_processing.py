import os
import sys
import numpy as np
import yaml
import json

import pandas as pd

from ldparser.ldparser import ldData
from scipy.signal import savgol_filter
import xml.etree.ElementTree as ET

def smooth(data: np.array, freq: int, time_step: float=0.2, polyorder: int=3):
    window = time_step // (1.0 / freq)
    return savgol_filter(data, window_length=window, polyorder=polyorder)

def derivative(data: np.array, freq: int) -> np.array:
    time_step = 1.0 / freq
    diff = np.concatenate(([0], data[:-1] - data[1:]))
    deriv = diff / time_step
    return deriv

class LoadSetup:
    def __init__(self, setup_file):
        try:
            self.setup_data = json.load(open(setup_file))
            setup_conv = yaml.safe_load(open("ACC_data/setup_conv.yaml"))[self.setup_data["carName"]]

            # ride height
            add_height = self.setup_data["advancedSetup"]["aeroBalance"]["rideHeight"]
            ride_height_f = setup_conv["rideHeight"]["front"] + add_height[0]
            ride_height_r = setup_conv["rideHeight"]["rear"] + add_height[2]

            # wheel rate
            wheelRate_idx = self.setup_data["advancedSetup"]["mechanicalBalance"]["wheelRate"]
            wheel_rate_lf = setup_conv["wheelRate"]["front"][wheelRate_idx[0]]
            wheel_rate_rf = setup_conv["wheelRate"]["front"][wheelRate_idx[1]]
            wheel_rate_lr = setup_conv["wheelRate"]["rear"][wheelRate_idx[2]]
            wheel_rate_rr = setup_conv["wheelRate"]["rear"][wheelRate_idx[3]]

            self.setup_values = {
                "RIDE_HEIGHT_F" : ride_height_f,
                "RIDE_HEIGHT_R" : ride_height_r,

                "WHEEL_RATE_LF" : wheel_rate_lf,
                "WHEEL_RATE_RF" : wheel_rate_rf,
                "WHEEL_RATE_LR" : wheel_rate_lr,
                "WHEEL_RATE_RR" : wheel_rate_rr,
            }
        except:
            raise FileNotFoundError("Setup File is not found")

class SessionData:
    def __init__(self, motec_file, setup):
        if '.ldx' in motec_file:
            ldx_file = motec_file
            ld_file = motec_file.split(".ldx")[0] + '.ld'
        
        elif '.ld' in motec_file:
            ld_file = motec_file
            ldx_file = motec_file.split(".ld")[0] + '.ldx'
        else:
            raise FileNotFoundError("Wrong file type given. Try again with .ld or .ldx files")

        # get lap data
        ld = ldData.fromfile(ld_file)
        self.lap_info = self.get_laps(ldx_file)
        self.get_laps_limits(ld)
        self.lap_times = self.get_laps_times(self.lap_info)
        self.num_laps = len(self.lap_times)

        self.vehicle_name = setup.setup_data["carName"]
        self.vehicle_weight = ld.head.event.venue.vehicle.weight
        self.track_name = ld.head.event.venue.name
        self.file_date = ld.head.datetime

        # TODO: change print to logs
        # print("READING: ", motec_file)
        # print("File date: ", file_date)
        # print("Track name: ", track_name)
        # print("Car name: ", vehicle_name)
        # print("Car weight: ", vehicle_weight)
        # for i, lap_time in enumerate(self.lap_times):
        #     print(f"Lap time {i}:\t{int(lap_time//60)}:{lap_time%60}")
        
        self.setup = setup.setup_data
        self.vd_ = yaml.safe_load(open("ACC_data/vehicle_data.yaml"))[self.vehicle_name]
        self.sd_ = setup.setup_values
        self.rd_ = {chann.name : TelemertyData(chann.name, chann.data, chann.freq, chann.unit) for chann in ld.channs}
        
        self.get_advanced_data()
        self.lap_data = self.split_lap_data()
    
    def get_session_info(self):
        return {"File data": self.file_date, "Track Name": self.track_name, "Vehicle_name": self.vehicle_name,
                "Vehicle Weight": self.vehicle_weight, "Number of laps": self.num_laps}
    
    def get_laptimes(self):
        return {i : lap_time for i, lap_time in enumerate(self.lap_times)}

    def get_advanced_data(self):
        self.rd_["g_lat"] = TelemertyData("g_lat", self.rd_['G_LAT'].data / 9.80665, self.rd_['G_LAT'].freq, "G")
        self.rd_["g_lon"] = TelemertyData("g_lon", self.rd_['G_LON'].data / 9.80665, self.rd_['G_LON'].freq, "G")

        # driving
        oversteer_freq = self.update_freq_map(["SPEED", "g_lat", "STEERANGLE"])
        oversteer = np.where(self.rd_['SPEED'].fm() < 13.88, 0, 1) * np.sign(self.rd_["g_lat"].fm()) * \
                        ((self.vd_["WHEELBASE"] * self.rd_["g_lat"].fm() / self.rd_['SPEED'].fm()**2) - \
                        np.sign(np.mean(self.rd_["STEERANGLE"].fm() * self.rd_["g_lat"].fm())) * \
                        self.rd_["STEERANGLE"].fm())

        slip_angle_f_freq = self.update_freq_map(["STEERANGLE", "ROTY", "SPEED"])
        slip_angle_f = self.rd_["STEERANGLE"].fm() - np.arctan(self.vd_["FRONT_AXLE"] * self.rd_["ROTY"].fm() / 
                                                                    self.rd_["SPEED"].fm())
        slip_angle_r_freq = self.update_freq_map(["ROTY", "SPEED"])
        slip_angle_r = - np.arctan(self.vd_["REAR_AXLE"] * self.rd_["ROTY"].fm() / self.rd_["SPEED"].fm())

        # tire data
        tire_load_lf_freq = self.update_freq_map(["SUS_TRAVEL_LF", "BUMPSTOP_FORCE_LF"])
        tire_load_lf = self.sd_["WHEEL_RATE_LF"] * self.rd_["SUS_TRAVEL_LF"].fm() + self.rd_["BUMPSTOP_FORCE_LF"].fm()

        tire_load_rf_freq = self.update_freq_map(["SUS_TRAVEL_RF", "BUMPSTOP_FORCE_RF"])
        tire_load_rf = self.sd_["WHEEL_RATE_RF"] * self.rd_["SUS_TRAVEL_RF"].fm() + self.rd_["BUMPSTOP_FORCE_RF"].fm()

        tire_load_lr_freq = self.update_freq_map(["SUS_TRAVEL_LR", "BUMPSTOP_FORCE_LR"])
        tire_load_lr = self.sd_["WHEEL_RATE_LR"] * self.rd_["SUS_TRAVEL_LR"].fm() + self.rd_["BUMPSTOP_FORCE_LR"].fm()

        tire_load_rr_freq = self.update_freq_map(["SUS_TRAVEL_RR", "BUMPSTOP_FORCE_RR"])
        tire_load_rr = self.sd_["WHEEL_RATE_RR"] * self.rd_["SUS_TRAVEL_RR"].fm() + self.rd_["BUMPSTOP_FORCE_RR"].fm()

        # suspension calc
        suspension_freq = self.rd_["SUS_TRAVEL_LF"].freq
        ride_height_lf = self.sd_["RIDE_HEIGHT_F"] - self.rd_["SUS_TRAVEL_LF"].data
        ride_height_rf = self.sd_["RIDE_HEIGHT_F"] - self.rd_["SUS_TRAVEL_RF"].data
        ride_height_lr = self.sd_["RIDE_HEIGHT_R"] - self.rd_["SUS_TRAVEL_LR"].data
        ride_height_rr = self.sd_["RIDE_HEIGHT_R"] - self.rd_["SUS_TRAVEL_RR"].data
        ride_height_split_f = self.rd_["SUS_TRAVEL_RF"].data - self.rd_["SUS_TRAVEL_LF"].data
        ride_height_split_r = self.rd_["SUS_TRAVEL_RR"].data - self.rd_["SUS_TRAVEL_LR"].data

        damper_vel_lf = derivative(self.rd_["SUS_TRAVEL_LF"].data, self.rd_["SUS_TRAVEL_LF"].freq)
        damper_vel_rf = derivative(self.rd_["SUS_TRAVEL_RF"].data, self.rd_["SUS_TRAVEL_RF"].freq)
        damper_vel_lr = derivative(self.rd_["SUS_TRAVEL_LR"].data, self.rd_["SUS_TRAVEL_LR"].freq)
        damper_vel_rr = derivative(self.rd_["SUS_TRAVEL_RR"].data, self.rd_["SUS_TRAVEL_RR"].freq)

        bs_force_rate_freq = self.rd_["BUMPSTOP_FORCE_LF"].freq
        bs_force_rate_lf = derivative(self.rd_["BUMPSTOP_FORCE_LF"].data, self.rd_["BUMPSTOP_FORCE_LF"].freq)
        bs_force_rate_rf = derivative(self.rd_["BUMPSTOP_FORCE_RF"].data, self.rd_["BUMPSTOP_FORCE_RF"].freq)
        bs_force_rate_lr = derivative(self.rd_["BUMPSTOP_FORCE_LR"].data, self.rd_["BUMPSTOP_FORCE_LR"].freq)
        bs_force_rate_rr = derivative(self.rd_["BUMPSTOP_FORCE_RR"].data, self.rd_["BUMPSTOP_FORCE_RR"].freq)

        # rake
        rake = (ride_height_lr + ride_height_rr - ride_height_lf - ride_height_rf)/2
        rake_angle = np.arctan(rake / self.vd_["WHEELBASE"])
        rake_left = ride_height_lr - ride_height_lf
        rake_right = ride_height_rr - ride_height_rf
        rake_left_right_split = rake_left - rake_right

        self.ad_ = {
            # driving
            "oversteer" : TelemertyData("oversteer", oversteer, oversteer_freq, 'rad'),
            "slip_angle_f" : TelemertyData("slip_angle_f", slip_angle_f, slip_angle_f_freq, 'rad'),
            "slip_angle_r" : TelemertyData("slip_angle_r", slip_angle_r, slip_angle_r_freq, 'rad'),

            # tire data
            "tire_load_lf" : TelemertyData("tire_load_lf", tire_load_lf, tire_load_lf_freq, 'N'),
            "tire_load_rf" : TelemertyData("tire_load_rf", tire_load_rf, tire_load_rf_freq, 'N'),
            "tire_load_lr" : TelemertyData("tire_load_lr", tire_load_lr, tire_load_lr_freq, 'N'),
            "tire_load_rr" : TelemertyData("tire_load_rr", tire_load_rr, tire_load_rr_freq, 'N'),
            
            # suspension calc
            "ride_height_lf" : TelemertyData("ride_height_lf", ride_height_lf, suspension_freq, 'mm'),
            "ride_height_rf" : TelemertyData("ride_height_rf", ride_height_rf, suspension_freq, 'mm'),
            "ride_height_lr" : TelemertyData("ride_height_lr", ride_height_lr, suspension_freq, 'mm'),
            "ride_height_rr" : TelemertyData("ride_height_rr", ride_height_rr, suspension_freq, 'mm'),
            "ride_height_split_f" : TelemertyData("ride_height_split_f", ride_height_split_f, suspension_freq, 'mm'),
            "ride_height_split_r" : TelemertyData("ride_height_split_r", ride_height_split_r, suspension_freq, 'mm'),

            # damper calc
            "damper_vel_lf" : TelemertyData("damper_vel_lf", damper_vel_lf, suspension_freq, 'mm/s'),
            "damper_vel_rf" : TelemertyData("damper_vel_rf", damper_vel_rf, suspension_freq, 'mm/s'),
            "damper_vel_lr" : TelemertyData("damper_vel_lr", damper_vel_lr, suspension_freq, 'mm/s'),
            "damper_vel_rr" : TelemertyData("damper_vel_rr", damper_vel_rr, suspension_freq, 'mm/s'),

            "bs_force_rate_lf" : TelemertyData("bs_force_rate_lf", bs_force_rate_lf, bs_force_rate_freq, 'N/s'),
            "bs_force_rate_rf" : TelemertyData("bs_force_rate_rf", bs_force_rate_rf, bs_force_rate_freq, 'N/s'),
            "bs_force_rate_lr" : TelemertyData("bs_force_rate_lr", bs_force_rate_lr, bs_force_rate_freq, 'N/s'),
            "bs_force_rate_rr" : TelemertyData("bs_force_rate_rr", bs_force_rate_rr, bs_force_rate_freq, 'N/s'),

            # rake
            "rake" : TelemertyData("rake", rake, suspension_freq, 'rad'),
            "rake_angle" : TelemertyData("rake_angle", rake_angle, suspension_freq, 'rad'),
            "rake_left_right_split" : TelemertyData("rake_left_right_split", rake_left_right_split, suspension_freq, 'ratio'),
        }
    
    def update_freq_map(self, name_list: list) -> dict:
        freq_set = {self.rd_[name].freq for name in name_list}
        min_freq = min(freq_set)
        freq_map = {}
        for freq in freq_set:
            skip = freq // min_freq
            freq_map[freq] = skip

        for name in name_list:
            self.rd_[name].freq_map = freq_map

        return min_freq

    def get_laps(self, ldx_file):
        laps = []
        try:
            tree = ET.parse(ldx_file)
            root = tree.getroot()

            # read lap times
            for lap in root[0][0][0][0]:
                laps.append(float(lap.attrib['Time'])*1e-6)

        except:
            raise FileNotFoundError(".ldx file is not found")

        return np.array(laps)

    def get_laps_limits(self, ld):
        """
        find the start/end indizes of the data for each lap
        """
        freq_list = {chan.freq for chan in ld.channs}
        lap_beacon = len(ld.channs[0].data)
        lap_freq = ld.channs[0].freq
        self.laps_limits = {}
        for freq in freq_list:
            n = freq * lap_beacon // lap_freq
            laps_limits = []
            if self.lap_info[0]!=0:
                laps_limits = [0]
            laps_limits.extend((np.array(self.lap_info)*freq).astype(int))
            laps_limits.extend([n])
            self.laps_limits[freq] = list(zip(laps_limits[:-1], laps_limits[1:]))

    def get_laps_times(self, laps):
        lap_times = []
        if len(laps) == 0: return lap_times
        if laps[0] != 0:  lap_times = np.array([laps[0]])
        lap_times = np.concatenate((lap_times, laps[1:]-laps[:-1]))

        return lap_times
    
    def split_lap_data(self):
        laps = []
        for lap_id in range(self.num_laps):
            lap_data = {}

            # split raw data
            for chann in self.rd_.values():
                limits = self.laps_limits[chann.freq][lap_id]
                data = chann.data[limits[0]:limits[1]]
                lap_data[chann.name] = TelemertyData(chann.name, data, chann.freq, chann.unit)
            
            # split raw data
            for chann in self.ad_.values():
                limits = self.laps_limits[chann.freq][lap_id]
                data = chann.data[limits[0]:limits[1]]
                lap_data[chann.name] = TelemertyData(chann.name, data, chann.freq, chann.unit)
            
            laps.append(LapTelemertyData(lap_id, self.lap_times[lap_id], lap_data))
        
        return laps

class LapTelemertyData:
    def __init__(self, lap_id, lap_time, lap_data):
        self.lap_id = lap_id
        self.time = float(lap_time)
        self.channs = lap_data
        self.get_turn_limits()
        self.split_turn_data()

    def get_turn_limits(self, g_threshold: float=0.5, min_sec: float=1.5) -> list:
        abs_g_lat = np.abs(self.channs["G_LAT"].data)
        g_lat_freq = self.channs["G_LAT"].freq
        turn_len = int(min_sec * g_lat_freq)
        turn_mask = abs_g_lat > g_threshold
        
        # find start and end
        transitions = np.diff(np.concatenate(([0], turn_mask.astype(int), [0])))
        start_indices = np.where(transitions == 1)[0]
        end_indices = np.where(transitions == -1)[0]

        # filter the turns based on the minimum duration
        self.num_turns = 0
        self.turns_limits = {chann.freq : [] for chann in self.channs.values()}
        for start, end in zip(start_indices, end_indices):
            if (end - start) >= turn_len:
                self.num_turns += 1
                turn_slice = abs_g_lat[start:end]
                apex = start + np.argmax(turn_slice)

                for freq in self.turns_limits.keys():
                    ext = freq // g_lat_freq
                    self.turns_limits[freq].append((start * ext, apex * ext, end * ext))

    def split_turn_data(self):
        self.turns = []
        self.turns_entry = []
        self.turns_exit = []
        for turn_id in range(self.num_turns):
            turn = {}
            turn_en = {}
            turn_ex = {}

            for name, chann in self.channs.items():
                limits = self.turns_limits[chann.freq][turn_id]
                entry_data = chann.data[limits[0]:limits[1]]
                turn_en[name] = TelemertyData(chann.name, entry_data, chann.freq, chann.unit)

                exit_data = chann.data[limits[1]:limits[2]]
                turn_ex[name] = TelemertyData(chann.name, exit_data, chann.freq, chann.unit)

                data = chann.data[limits[0]:limits[2]]
                turn[name] = TelemertyData(chann.name, data, chann.freq, chann.unit)
            
            self.turns.append(turn)
            self.turns_entry.append(turn_en)
            self.turns_exit.append(turn_ex)
    
    def get_entry(self, turn_idx):
        return self.turns_entry[turn_idx]
    
    def get_exit(self, turn_idx):
        return self.turns_exit[turn_idx]

    def get_turn(self, turn_idx):
        return self.turns[turn_idx]
    
    def all_turns(self):
        return self.turns
    
    def all_entry(self):
        return self.turns_entry
    
    def all_exit(self):
        return self.turns_exit
        
    def gen_track(self):
        """
        Generates a 2D track map from lap telemetry data.
        
        Args:
            lap_telemetry: A dictionary containing telemetry data channels. 
                        Must include 'SPEED' (m/s) and 'ROTY' (deg/s).
            freq: The data frequency in Hertz (Hz).
            
        Returns:
            track_map: (x_coordinates, y_coordinates).
        """
        speed = self.channs['SPEED']
        roty = self.channs['ROTY']
        
        min_freq = min(speed.freq, roty.freq)
        speed_idx = speed.freq // min_freq
        roty_idx = roty.freq // min_freq
        time_step = 1.0 / min_freq

        heading = np.cumsum(roty.data[::roty_idx] * time_step)
        distance = speed.data[::speed_idx] * time_step

        min_len = min(distance.size, heading.size)
        dx = distance[:min_len] * np.cos(heading[:min_len])
        dy = distance[:min_len] * np.sin(heading[:min_len])

        x_coords = np.cumsum(dx)
        y_coords = np.cumsum(dy)

        # making sure the complete closer
        final_x_error = x_coords[-1]
        final_y_error = y_coords[-1]
        correction_x = np.linspace(0, final_x_error, len(x_coords))
        correction_y = np.linspace(0, final_y_error, len(y_coords))
        
        map_x = x_coords - correction_x
        map_y = y_coords - correction_y

        return map_x, map_y, min_freq

class TelemertyData:
    def __init__(self, name: str, data: np.array, freq: int, unit: str):
        self.name = name
        self.data = data
        self.freq = freq
        self.unit = unit
        if unit == 'deg': self.to_rad()

        self.freq_map = None
        self.series = None
    
    def to_rad(self):
        self.data = np.deg2rad(self.data)
        self.unit = 'rad'
    
    def fm(self):
        slice = self.freq_map[self.freq]
        return self.data[::slice]

def main():
    if len(sys.argv)!=3:
        print("Usage: data_processing.py </Motec/file_name> </path/to/setup.json>")
        sys.exit(1)

    SessionData(sys.argv[1], LoadSetup(sys.argv[2]))


if __name__ == '__main__':
    main()