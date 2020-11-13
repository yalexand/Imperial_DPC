clear all;

addpath_DPC;

ALYtools_dir = 'C:\Users\alexany\ALYtools';
addpath(ALYtools_dir);
addpath_ALYtools;

    dpc = DPC_processor;
    dpc.use_gpu = true;

    % Authors' example
    dpc.na            = 0.4;                        % numerical aperture of the imaging system
    dpc.na_illum      = 0.4;                        % numerical aperture of the illumination
    dpc.magnification = 20*2;                       % magnification of the imaging system
    dpc.ps            = 6.5/dpc.magnification;      % pixel size in micron    
    dpc.lambda        = 0.514;                      % wavelength in micron
    dpc.illu_rotation = [0, 180, 90, 270];
    dpc.load_images('dataset_DPC_MCF10A.mat',true); % do permute
              
    dpc.setCoordinates;
    dpc.genSource_no_aberration_correction;
    dpc.zernike_poly = genZernikePoly(dpc.Fx, dpc.Fy, dpc.na, dpc.lambda, dpc.num_Zernike);        
    dpc.generate_pupil;
    dpc.normalize_images([],[]);    
        
    t0=tic;    
        reg_L2 = 1.0*[1e-1, 5e-3]; % regularization coeffs
        [amplitude, phase] = dpc.DPC_L2(reg_L2);
    toc(t0)

icy_imshow(phase);
