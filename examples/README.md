# Notes on Examples

## Getting Started
### 00
- 3 yaw angles provided for 1 wind turbine, 1 wind speed, and 1 wind direction combination, so downsized to 1
- Will not run because we can't calculate a custom set of points currently

### 01
- Created a control flow to better find the input JSON that will now be applied to all examples so it can run from anywhere within floris, floris/examples, or floris/examples/example-folder
- Will not run due to the same issue as 00

### 02
- Updated the method to change turbine parameters and enforce using reinitialize_flow_field
- Gets caught at same point as 00, and 01

### 03
- Runs!

### 04
- TurbOPark, Multizone, and Gauss Legacy were commented out because they're not implemented
- A ton of changes to FLORIS functionality to align everything, and get the examples up and running up to the common failure point
- Gets caught at same point as 00, 01, and 02


## AEP Calculations
### AEP No Class
- Updated the way the turbine parameters are retrieved to match the new API
- If you uncomment lines 78-84, the example will run just fine, otherwise, the you get stuck with the same `NotImplementedError` as in getting started

### AEP
- Updated the way the turbine parameters are retrieved to match the new API
- Lines 78-84 will not run, per all previous comments
- Line 124, `YawOptimizationWindRose`, gets stuck at tools/optimization/scipy/yaw_wind_rose:576 where we loop through the turbine's yaw angles
    - Changed this to just use the farm.farm_controller.yaw_angles....
- Updated set `FarmController.set_yaw_angles` control flow to allow for nested inputs of the correct size to be used
- Aside from 78-84, the code runs with seemingly the same results as the previous example


## Change Turbine
### Change diameter
- Added a new function to create a copy of the turbine mapping, then update based on that
- `FlorisInterface.change_turbine` now lets you map to a turbine that already exists in the `Floris.farm.turbine_map`.
- Applied vectorization to the wake calculation, but now there is no power being produced
