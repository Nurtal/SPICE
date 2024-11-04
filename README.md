# SPICE
## Spatial Pattern Interaction for Cell Exploration
This program computes a matrix of neighbors from one or multiple FCS files.

### Arguments:
* input: Path to the FCS file or a directory containing multiple FCS files.
* output: Path to the output directory where results will be saved.
* radius_min: Minimum radius used to compute the neighborhood.
* radius_max: Maximum radius used to compute the neighborhood.
    
### Example Usage:
        
        spice.py -i 'path/to/my/fcs_file' -o '/path/to/my/output/dir' -r 1 -R 10



