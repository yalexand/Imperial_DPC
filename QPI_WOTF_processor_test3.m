clear all;

addpath_DPC;

ALYtools_dir = 'C:\Users\alexany\ALYtools';
addpath(ALYtools_dir);
addpath_ALYtools;

    dpc = DPC_processor;
    dpc.use_gpu = true;
    
%     10x example
    dpc.na            = 0.3;                       % numerical aperture of the imaging system
    dpc.na_illum      = 0.3;                       % numerical aperture of the illumination
    dpc.magnification = 10;                        % magnification of the imaging system
    dpc.ps            = 0.571;                     % pixel size in micron    
    dpc.lambda        = 0.560;                     % wavelength in micron
    dpc.illu_rotation = [0, 180, 90, 270];   
    img_dir = 'D:\Users\alexany\QPI_Spring_2019\November_2020\20201112\HEK\semicircles_22p';
         
%     20x example
%       sigma              = 0.9325;                    % partial coherence factor – the ratio (illumination NA)/(objective NA)
%       dpc.na            = 0.4;                        % numerical aperture of the imaging system
%       dpc.na_illum      = 0.4; %sigma*dpc.na;         % numerical aperture of the illumination
%       dpc.lambda        = 0.560;                      % wavelength in micron
%       dpc.magnification = 22.9;                       % magnification of the imaging system
%       dpc.ps            = 6.5/dpc.magnification;      % pixel size in micron
%       dpc.illu_rotation = [0, 180, 90, 270];                      
%       img_dir = 'D:\Users\alexany\QPI_Spring_2019\November_2020\20201112\HEK\20x\set1\semicircles_28p';    
          
    fnames = {'Top.tif','Bottom.tif','Left.tif','Right.tif'};
    for k=1:numel(fnames)
        dpc.load_image([img_dir filesep fnames{k}],k);
    end
          
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
