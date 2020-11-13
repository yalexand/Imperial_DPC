clear all;

addpath_DPC;

ALYtools_dir = 'C:\Users\alexany\ALYtools';
addpath(ALYtools_dir);
addpath_ALYtools;

    dpc = DPC_processor;
    dpc.use_gpu = true;

    % Authors' aberration correction example
    dpc.na            = 0.4;                       % numerical aperture of the imaging system
    dpc.na_illum      = 0.4;                       % numerical aperture of the illumination
    dpc.magnification = 20*2;                      % magnification of the imaging system
    dpc.ps            = 6.5/dpc.magnification;     % pixel size in micron    
    dpc.lambda        = 0.514;                     % wavelength in micron
    dpc.illu_rotation = [0, 180, 90];
    %
    dpc.load_images('dataset_DPC_with_aberration.mat',false); % don't permute
           
    max_iter    = 20;
    use_tv      = false; % change to true if want TV
    verbose     = true;
    [amplitude,phase] = dpc.restore_with_aberrations_correction(max_iter,use_tv,verbose);
   
icy_imshow(amplitude);
icy_imshow(phase);
