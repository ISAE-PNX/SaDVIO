The default folder structure for the experiment data and the estimator results should resemble this:

```
data/
└── <exp_folder>/
    └── <exp_date>/
        ├── traj_<exp_date>.txt
        ├── data_ubx_<exp_date>.mat
        ├── ublox/
        │   └── (ROS2 bag data)
        ├── rinex/
        │   └── (RINEX data files)
        └── _results/
            ├── context/
            │   └── context_pred_<exp_date>.mat
            ├── vio/
            │   └── <vo_bag_folder>/
            │       ├── (ROS2 bag data for VO)
            │       └── fig/
            │           └── (generated figures)
            └── pg/
                └── <pg_bag_folder>/
                    ├── (ROS2 bag data for PG)
                    └── fig/
                        └── (generated figures)
```