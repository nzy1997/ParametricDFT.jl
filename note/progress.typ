= Progress Record 12/7 

1. Introduce a Dataset and update the training for parametric DFT. 
2. Add Analysis code based on the img_process analysis. Training on multiple images, and result is not significant based on the analysis.
3. Bring some other examples using the parametric DFT. Edge Detection, and others.  
4. Add some notes for the fundamentals of the Manopt.jl, how it turns to gradient manifold optimization. 
5. Add some note for tensor contraction used in the process creating the quantum fourier transform circuit. 
6. Minor: Fix original documentation error.

== Some questions

1. The original code is trying to compress the image by 95%, which might not be a representative case. Should we also test the other numbers for compression, like 70%? 60% ? To check the effect of the function and its correlation with the compression value next. 
2. Selection of the optimization process. L1-norm is probably not a good options, some other possible analysis process or optimization method could work better?  
3. Do we have some existing reference that could help this work better? 