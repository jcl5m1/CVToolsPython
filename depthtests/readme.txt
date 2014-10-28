Python script for testing depth noise of flat wall and corner datasets.  
It can parse a binary little endian PLY file of vertices.  It will do a best fit plane analysis.
It will attempts to autodetect if it is a corner dataset or of a flat wall.
While the script attempts to flip the Z-axis of the data as needed, it natively prefers -Z 
as the camera direction. It will auto-generate a test file if none is provided.

-h will print the help screen

This script requires the following python additional libraries:
numpy
pyOpenGL
