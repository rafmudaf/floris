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
