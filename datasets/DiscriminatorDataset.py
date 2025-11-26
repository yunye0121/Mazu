from pathlib import Path
import xarray as xr
import numpy as np
import pandas as pd
import torch

class ERA5TWDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_root_dir: str,
        start_date_hour: pd.Timestamp,
        end_date_hour: pd.Timestamp,
        upper_variables: list[str],
        surface_variables: list[str],
        static_variables: list[str],
        levels: list[int],
        latitude: tuple[int, int],
        longitude: tuple[int, int],
        get_datetime: bool = True,
    ) -> None:
        super().__init__()
        self.data_root_dir = data_root_dir
        self.start_date_hour = pd.Timestamp(start_date_hour)
        self.end_date_hour = pd.Timestamp(end_date_hour)
        self.upper_variables = upper_variables
        self.surface_variables = surface_variables
        self.static_variables = static_variables
        self.levels = levels
        self.latitude = latitude
        self.longitude = longitude
        self.get_datetime = get_datetime    

    def map_var_name_for_Aurora(self, var_name: str) -> str:
        var_name_mapping = {
            "t2m": "2t",
            "u10": "10u",
            "v10": "10v",
            "msl": "msl",
        }
        if var_name in var_name_mapping:
            return var_name_mapping[var_name]
        else:
            return var_name

    def get_latitude_longitude(self):
        upper_path, sfc_path = self._dt_to_path(self.start_date_hour)
        upper_nc = xr.open_dataset(upper_path).load()
        latitude, longitude = \
            upper_nc.latitude.sel(latitude = slice(*self.latitude)).values, \
            upper_nc.longitude.sel(longitude = slice(*self.longitude)).values
        upper_nc.close()
        return torch.Tensor(latitude), torch.Tensor(longitude)
    
    def get_levels(self):
        upper_path, _ = self._dt_to_path(self.start_date_hour)
        upper_nc = xr.open_dataset(upper_path).load()
        levels = upper_nc.pressure_level.values
        upper_nc.close()
        return tuple(levels)
    
    def get_static_vars_ds(self):
        _ds = xr.open_dataset(self.data_root_dir + "/static/static_vars.nc").load()
        _d = {
            "static_vars": {
                v: torch.Tensor(
                    _ds[v].sel(
                        latitude = slice(*self.latitude), longitude = slice(*self.longitude)
                    ).values
                ).squeeze() for v in self.static_variables
            }
        }
        return _d

    def _dt_to_path(self, date_hour: pd.Timestamp) -> str:
        dir_path = Path(self.data_root_dir) / date_hour.strftime(r"%Y/%Y%m/%Y%m%d")
        name = date_hour.strftime(r"%Y%m%d%H")

        return str(dir_path / f"{name}_upper.nc"), str(dir_path / f"{name}_sfc.nc")

    def __len__(self) -> int:
        duration = self.end_date_hour - self.start_date_hour + pd.Timedelta(hours = 1)
        return round(duration.total_seconds()) // (60 * 60)

    def _nc_to_dict(self, upper_nc, sfc_nc) -> dict:
        _d = {
            "surf_vars": {
                self.map_var_name_for_Aurora(v): torch.Tensor(
                    sfc_nc[v].sel(
                        latitude = slice(*self.latitude),
                        longitude = slice(*self.longitude)
                    ).values,
                ) for v in self.surface_variables
            },
            "atmos_vars": {
                v: torch.Tensor(
                    upper_nc[v].sel(
                        pressure_level = self.levels,
                        latitude = slice(*self.latitude),
                        longitude = slice(*self.longitude),
                    ).values
                ) for v in self.upper_variables
            },
        }
        return _d
        
    def __getitem__(self, index: int) -> tuple:
        cur = self.start_date_hour + pd.Timedelta(hours = index)
        upper_path, sfc_path = self._dt_to_path(cur)
        upper_nc = xr.open_dataset(upper_path).load()
        sfc_nc = xr.open_dataset(sfc_path).load()
        _d = self._nc_to_dict(upper_nc, sfc_nc)
        upper_nc.close()
        sfc_nc.close()

        if self.get_datetime:
            return _d, cur.strftime("%Y-%m-%d %H:%M:%S")
        
        return _d

class AuroraPredictionDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_root_dir: str,
        start_date_hour: pd.Timestamp,
        end_date_hour: pd.Timestamp,
        upper_variables: list[str],
        surface_variables: list[str],
        static_variables: list[str],
        latitude: tuple[int, int],
        longitude: tuple[int, int],
        levels: list[int],
        forecast_hour: int = 1,
        get_datetime: bool = True,
    ) -> None:
        super().__init__()
        self.data_root_dir = data_root_dir
        self.start_date_hour = pd.Timestamp(start_date_hour)
        self.end_date_hour = pd.Timestamp(end_date_hour)
        self.upper_variables = upper_variables
        self.surface_variables = surface_variables
        self.static_variables = static_variables
        self.latitude = latitude
        self.longitude = longitude
        self.levels = levels
        self.forecast_hour = forecast_hour
        self.get_datetime = get_datetime

    def __len__(self):
        total_hours = int((self.end_date_hour - self.start_date_hour).total_seconds() // 3600)
        return (total_hours + 1)

    def _nc_to_dict(self, pred_nc) -> dict:

        _d = {
            "surf_vars": {
                k.removeprefix("surf_"): torch.Tensor(v.values) for k, v in pred_nc.items() if k.startswith("surf_")
            },
            "atmos_vars": {
                k.removeprefix("atmos_"): torch.Tensor(v.values) for k, v in pred_nc.items() if k.startswith("atmos_")
            },
        }
        return _d

    def get_latitude_longitude(self):
        ds_path = str(Path(self.data_root_dir) / \
            f"{ (self.start_date_hour).strftime('%Y%m%d_%H0000')}+{self.forecast_hour}hr.nc",)
        ds = xr.open_dataset(
            ds_path,
        )
        latitude, longitude = \
            torch.Tensor(ds["latitude"].values), torch.Tensor(ds["longitude"].values)
        return latitude, longitude

    def get_levels(self):
        ds_path = str(Path(self.data_root_dir) / \
            f"{ (self.start_date_hour).strftime('%Y%m%d_%H0000')}+{self.forecast_hour}hr.nc",)
        ds = xr.open_dataset(
            ds_path,
        )
        level = torch.Tensor( ds["level"].values )
        return level

    def __getitem__(self, index: int):
        
        ds_dir = str(Path(self.data_root_dir) / \
            f"{ (self.start_date_hour + pd.Timedelta(hours = index * 1)).strftime('%Y%m%d_%H0000')}+{self.forecast_hour}hr.nc",)
        ds = xr.open_dataset(
            ds_dir,
        )

        _d = self._nc_to_dict( ds )
        dt = ds["time"].values[0]
        ds.close()

        if self.get_datetime:
            return _d, pd.Timestamp(dt).strftime("%Y-%m-%d %H:%M:%S")

        return _d

class DiscriminatorDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        ERA5TWDataset: ERA5TWDataset,
        AuroraTWDataset: AuroraPredictionDataset,
    ) -> None:
        super().__init__()
        
        if not isinstance(ERA5TWDataset, list):
            ERA5TWDataset = [ERA5TWDataset]
        if not isinstance(AuroraTWDataset, list):
            AuroraTWDataset = [AuroraTWDataset]

        self.ERA5_datasets = ERA5TWDataset
        self.Aurora_datasets = AuroraTWDataset

        self.ERA5_lengths = [len(ds) for ds in self.ERA5_datasets]
        self.Aurora_lengths = [len(ds) for ds in self.Aurora_datasets]

        self.ERA5_cum = np.cumsum([0] + self.ERA5_lengths)
        self.Aurora_cum = np.cumsum([0] + self.Aurora_lengths)

        self.total_ERA5 = sum(self.ERA5_lengths)
        self.total_Aurora = sum(self.Aurora_lengths)
        self.total_len = self.total_ERA5 + self.total_Aurora

    def __len__(self):
        return self.total_len

    def get_latitude_longitude(self):
        return self.ERA5_datasets[0].get_latitude_longitude()

    def get_levels(self):
        return self.ERA5_datasets[0].get_levels()

    def get_static_vars_ds(self):
        return self.ERA5_datasets[0].get_static_vars_ds()

    def __getitem__(self, index):
        if index < 0 or index >= self.total_len:
            raise IndexError(f"Index {index} out of range for dataset of length {self.total_len}")

        if index < self.total_ERA5:
            ds_index = np.searchsorted(self.ERA5_cum, index, side = "right") - 1
            sample_index = index - self.ERA5_cum[ds_index]
            _x = self.ERA5_datasets[ds_index][sample_index]
            label = 0 

        else:
            adjusted_index = index - self.total_ERA5
            ds_index = np.searchsorted(self.Aurora_cum, adjusted_index, side = "right") - 1
            sample_index = adjusted_index - self.Aurora_cum[ds_index]
            _x = self.Aurora_datasets[ds_index][sample_index]
            label = 1

        return _x, label

if __name__ == "__main__":
    from ERA5TWDatasetforAurora import ERA5TWDatasetforAurora
    from AuroraTWandERA5TWDatasetforAurora import AuroraTWandERA5TWDatasetforAurora

    data_root_dir = '/work/yunye0121/era5_tw'
    start_date_hour = '2013-01-01 01:00:00'
    end_date_hour = '2013-01-30 20:00:00'
    upper_variables = ['u', 'v', 't', 'q', 'z']
    surface_variables = ['t2m', "u10", "v10", "msl"]
    static_variables = ['z', 'lsm', 'slt']
    latitude = (39.75, 5)
    longitude = (100, 144.75)
    levels = [1000, 925, 850, 700, 500, 300, 150, 50]
    lead_time = 0
    input_time_window = 1
    rollout_step = 0

    Aurora_input_dir = "/work/yunye0121/ar_96hrs_2013-2018_result"
    forecast_hour = 1
    # use_Aurora_input_len = 1

    start = pd.Timestamp(start_date_hour)
    end = pd.Timestamp(end_date_hour)
    era5_start = (start + pd.Timedelta(hours=forecast_hour)).strftime('%Y-%m-%d %H:%M:%S')
    era5_end = (end + pd.Timedelta(hours=forecast_hour)).strftime('%Y-%m-%d %H:%M:%S')

    era5_ds= ERA5TWDataset(
        data_root_dir=data_root_dir,
        start_date_hour=era5_start,      # shifted!
        end_date_hour=era5_end,          # shifted!
        upper_variables=upper_variables,
        surface_variables=surface_variables,
        static_variables=static_variables,
        latitude=latitude,
        longitude=longitude,
        levels=levels,
        # get_datetime=False, 
        get_datetime=True,
        # flatten=False,
    )

    aurora_ds = AuroraPredictionDataset(
        data_root_dir=Aurora_input_dir,
        start_date_hour=start,
        end_date_hour=end,
        upper_variables=upper_variables,
        surface_variables=surface_variables,
        static_variables=static_variables,
        latitude=latitude,
        longitude=longitude,
        levels=levels,
        forecast_hour=forecast_hour,
        get_datetime = True,
    )

    discriminator_ds = DiscriminatorDataset(
        ERA5TWDataset = era5_ds,
        AuroraTWDataset = aurora_ds,
    )

    print(f"{len(discriminator_ds)=}")

    print("DiscriminatorDataset length:", len(discriminator_ds))
    print("\nIterating over DiscriminatorDataset samples:")
    for i in range(3):
        (x, input_dates), label = discriminator_ds[i]
        print(f"\nDiscriminator sample[{i}]:")
        print("  Sample:")
        print(f"    surf_vars['2t'] shape: {x['surf_vars']['2t'].shape}")
        print(f"    atmos_vars['u'] shape: {x['atmos_vars']['u'].shape}")
        # print(f"    static_vars['z'] shape: {x['static_vars']['z'].shape}")
        print(f"{input_dates=}")
        print(f"{label=}")
    
    for i in range(len(discriminator_ds)-3, len(discriminator_ds)):
        (x, input_dates), label = discriminator_ds[i]
        print(f"\nDiscriminator sample[{i}]:")
        print("  Sample:")
        print(f"    surf_vars['2t'] shape: {x['surf_vars']['2t'].shape}")
        print(f"    atmos_vars['u'] shape: {x['atmos_vars']['u'].shape}")
        # print(f"    static_vars['z'] shape: {x['static_vars']['z'].shape}")
        print(f"{input_dates=}")
        print(f"{label=}")

    print("\nTest complete. You are ready to train your model using this paired dataset!")

