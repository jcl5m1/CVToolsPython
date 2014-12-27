Python script for testing depth noise of flat wall and corner datasets. 

It will auto-generate a test file if none is provided.
=======
Python script for testing depth noise of flat wall and corner datasets.  
It can parse a binary little endian PLY file of vertices.  It will do a best fit plane analysis.

While the script attempts to flip the Z-axis of the data as needed, it natively prefers -Z 
as the camera direction. It will auto-generate a test PLY file if none is provided.

It will attempt to autodetect if it is a corner dataset or of a flat wall. 
It will split the corner left and right at X=0.  Use the -x (offset) parameter to shift the dataset.
Press any key to toggle between othogonal top down view, and the rotating perspective view.
It only renders a subsample of points for fast rendering, but does that calculations using the full dataset.

-h will print the help screen

This script requires the following python additional libraries:
numpy
pyOpenGL
